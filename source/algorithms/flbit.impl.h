#include "flbit.h"
#include "flbit.tmpl.h"
#include "math/float.h"
//
#include "config/settings.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "general/iter.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/num.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "qm/lbit.h"
#include "qm/spin.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/common/prof.h"
#include "tools/common/split.h"
#include "tools/finite/h5.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/entanglement_entropy.h"
#include "tools/finite/measure/expectation_value.h"
#include "tools/finite/measure/number_entropy.h"
#include "tools/finite/mps.h"
#include "tools/finite/ops.h"
#include "tools/finite/print.h"
#include <complex>
#include <fmt/ranges.h>
#include <h5pp/h5pp.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

template<typename Scalar>
flbit<Scalar>::flbit(std::shared_ptr<h5pp::File> h5file_) : AlgorithmFinite<Scalar>(std::move(h5file_), OptRitz::NONE, AlgorithmType::fLBIT) {
    tools::log->trace("Constructing class_flbit");
    tensors.state->set_name("state_real");
}

template<typename Scalar>
void flbit<Scalar>::resume() {
    // Resume can imply many things
    // 1) Resume a simulation which terminated prematurely
    // 2) Resume a previously successful simulation. This may be desireable if the config
    //    wants something that is not present in the file.
    //      a) A certain number of states
    //      b) A state inside a particular energy window
    //      c) The ground or "roof" states
    // To guide the behavior, we check the setting ResumePolicy.

    auto state_name = settings::storage::file_resume_name;
    if(state_name.empty()) state_name = "state_real";

    auto state_prefixes = tools::common::h5::resume::find_state_prefixes(*h5file, status.algo_type, state_name);
    if(state_prefixes.empty()) throw except::state_error("no resumable states were found");
    for(const auto &state_prefix : state_prefixes) {
        tools::log->info("Resuming state [{}]", state_prefix);
        tools::finite::h5::load::simulation(*h5file, state_prefix, tensors, status, status.algo_type);

        // Our first task is to decide on a state name for the newly loaded state
        // The simplest is to inferr it from the state prefix itself
        auto name = tools::common::h5::resume::extract_state_name(state_prefix);
        tensors.state->set_name(name);
        clear_convergence_status();

        // Create an initial state in the real basis
        switch(settings::strategy::initial_state) {
            case StateInit::PRODUCT_STATE_DOMAIN_WALL:
            case StateInit::PRODUCT_STATE_PATTERN:
            case StateInit::PRODUCT_STATE_NEEL:
            case StateInit::PRODUCT_STATE_NEEL_SHUFFLED:
            case StateInit::PRODUCT_STATE_NEEL_DISLOCATED: break;
            default:
                tools::log->warn("Expected initial_state: "
                                 "PRODUCT_STATE_DOMAIN_WALL|"
                                 "PRODUCT_STATE_PATTERN|"
                                 "PRODUCT_STATE_NEEL|"
                                 "PRODUCT_STATE_NEEL_SHUFFLED|"
                                 "PRODUCT_STATE_NEEL_DISLOCATED"
                                 ". Got {}",
                                 enum2sv(settings::strategy::initial_state));
        }

        if(settings::strategy::initial_axis.find("z") == std::string::npos)
            tools::log->warn("Expected initial_axis == z. Got {}", settings::strategy::initial_axis);

        tensors.initialize_state(ResetReason::INIT, settings::strategy::initial_state, StateInitType::REAL, settings::strategy::initial_axis,
                                 settings::strategy::use_eigenspinors, settings::get_bond_min(status.algo_type), settings::strategy::initial_pattern);

        tensors.move_center_point_to_inward_edge();

        // Load the unitary circuit
        unitary_gates_2site_layers = qm::lbit::read_unitary_2site_gate_layers(*h5file, "/fLBIT/model/unitary_circuit");

        // Generate the corresponding state in lbit basis
        transform_to_lbit_basis();

        // Initialize a custom task list
        std::deque<flbit_task> task_list;

        if(status.algorithm_has_finished) {
            task_list.emplace_back(flbit_task::POST_PRINT_RESULT);
        } else {
            // This could be a savepoint state
            // Simply "continue" the algorithm until convergence
            if(name.find("real") != std::string::npos) {
                task_list.emplace_back(flbit_task::INIT_TIME);
                task_list.emplace_back(flbit_task::INIT_GATES);
                task_list.emplace_back(flbit_task::TRANSFORM_TO_LBIT);
                task_list.emplace_back(flbit_task::TIME_EVOLVE);
            } else
                throw except::runtime_error("Unrecognized state name for flbit: [{}]", name);
            task_list.emplace_back(flbit_task::POST_DEFAULT);
        }
        run_task_list(task_list);
    }

    // If we reached this point the current state has finished for one reason or another.
    // TODO: We may still have some more things to do, e.g. the config may be asking for more states
}

template<typename Scalar> void flbit<Scalar>::run_task_list(std::deque<flbit_task> &task_list) {
    using namespace settings::strategy;
    while(not task_list.empty()) {
        auto task = task_list.front();
        switch(task) {
            case flbit_task::INIT_RANDOMIZE_MODEL: initialize_model(); break;
            case flbit_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE: initialize_state(ResetReason::INIT, StateInit::RANDOM_PRODUCT_STATE); break;
            case flbit_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE_NEEL_SHUFFLED:
                initialize_state(ResetReason::INIT, StateInit::PRODUCT_STATE_NEEL_SHUFFLED);
                break;
            case flbit_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE_NEEL_DISLOCATED:
                initialize_state(ResetReason::INIT, StateInit::PRODUCT_STATE_NEEL_DISLOCATED);
                break;
            case flbit_task::INIT_RANDOMIZE_INTO_ENTANGLED_STATE: initialize_state(ResetReason::INIT, StateInit::RANDOM_ENTANGLED_STATE); break;
            case flbit_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE_PATTERN: initialize_state(ResetReason::INIT, StateInit::PRODUCT_STATE_PATTERN); break;
            case flbit_task::INIT_RANDOMIZE_INTO_MIDCHAIN_SINGLET_NEEL_STATE:
                initialize_state(ResetReason::INIT, StateInit::MIDCHAIN_SINGLET_NEEL_STATE);
                break;
            case flbit_task::INIT_BOND_LIMITS: init_bond_dimension_limits(); break;
            case flbit_task::INIT_TRNC_LIMITS: init_truncation_error_limits(); break;
            case flbit_task::INIT_WRITE_MODEL: write_to_file(StorageEvent::MODEL); break;
            case flbit_task::INIT_CLEAR_STATUS: status.clear(); break;
            case flbit_task::INIT_CLEAR_CONVERGENCE: clear_convergence_status(); break;
            case flbit_task::INIT_DEFAULT: run_preprocessing(); break;
            case flbit_task::INIT_TIME: {
                create_time_points();
                update_time_step();
                break;
            }
            case flbit_task::INIT_GATES: {
                create_unitary_circuit_gates();
                create_hamiltonian_gates();
                update_time_evolution_gates();
                break;
            }
            case flbit_task::TIME_EVOLVE:
                tensors.state->set_name("state_real");
                if(settings::flbit::run_iter_in_parallel)
                    run_algorithm_parallel();
                else
                    run_algorithm();
                break;
            case flbit_task::TRANSFORM_TO_LBIT: transform_to_lbit_basis(); break;
            case flbit_task::TRANSFORM_TO_REAL: transform_to_real_basis(); break;
            case flbit_task::POST_WRITE_RESULT: write_to_file(StorageEvent::FINISHED); break;
            case flbit_task::POST_PRINT_RESULT: print_status_full(); break;
            case flbit_task::POST_PRINT_TIMERS: tools::common::timer::print_timers(); break;
            case flbit_task::POST_DEFAULT: run_postprocessing(); break;
            case flbit_task::TIMER_RESET: tid::reset("fLBIT"); break;
        }
        task_list.pop_front();
    }
}

template<typename Scalar> void flbit<Scalar>::run_default_task_list() {
    std::deque<flbit_task> default_task_list = {
        flbit_task::INIT_DEFAULT,
        flbit_task::TIME_EVOLVE,
        flbit_task::POST_DEFAULT,
    };

    run_task_list(default_task_list);
    if(not default_task_list.empty()) {
        for(auto &task : default_task_list) tools::log->critical("Unfinished task: {}", enum2sv(task));
        throw except::runtime_error("Simulation ended with unfinished tasks");
    }
}

template<typename Scalar> void flbit<Scalar>::run_preprocessing() {
    tools::log->info("Running {} preprocessing", status.algo_type_sv());
    auto t_pre = tid::tic_scope("pre");
    status.clear();

    initialize_model(); // First use of random!
    init_bond_dimension_limits();
    init_truncation_error_limits();

    // Create an initial state in the real basis
    tensors.state->set_name("state_real");
    initialize_state(ResetReason::INIT, settings::strategy::initial_state);
    tensors.move_center_point_to_inward_edge();
    state_real_init = std::make_unique<StateFinite<Scalar>>(tensors.get_state());
    tools::finite::print::model(tensors.get_model());

    create_unitary_circuit_gates();

    // Store the model
    write_to_file(StorageEvent::MODEL, CopyPolicy::TRY);

    create_time_points();
    update_time_step();
    create_hamiltonian_gates();
    update_time_evolution_gates();

    if(not tensors.position_is_inward_edge()) throw except::logic_error("Put the state on an edge!");

    // Generate the corresponding state in lbit basis
    transform_to_lbit_basis();
    if constexpr(settings::debug) {
        if(tensors.get_length() <= 8) {
            // We know that we should get back to the initial state in the real basis, if we apply
            // the unitary transformation on the lbit state : Ψ = U|Ψ'⟩
            auto svd_cfg           = svd::config(status.bond_lim, status.trnc_lim);
            svd_cfg.svd_lib        = svd::lib::lapacke;
            svd_cfg.svd_rtn        = svd::rtn::geauto;
            StateFinite state_real = tensors.get_state();
            if(settings::flbit::use_mpo_circuit) {
                state_real = qm::lbit::transform_to_real_basis(*state_lbit, unitary_gates_mpo_layers, ledge, redge, svd_cfg);

            } else {
                state_real = qm::lbit::transform_to_real_basis(*state_lbit, unitary_gates_2site_layers, svd_cfg);
            }
            auto psi_lbit_t  = tools::finite::mps::mps2tensor<Scalar>(*state_lbit);
            auto psi_real1_t = tools::finite::mps::mps2tensor<Scalar>(tensors.get_state());
            auto psi_real2_t = tools::finite::mps::mps2tensor<Scalar>(state_real);
            auto u_circuit_t =
                qm::lbit::get_unitary_circuit_as_tensor(unitary_gates_2site_layers); // TODO: For some reason this is a transpose away from our typical circuit
            auto        psi_lbit_v            = tenx::VectorMap(psi_lbit_t);
            auto        psi_real1_v           = tenx::VectorMap(psi_real1_t);
            auto        psi_real2_v           = tenx::VectorMap(psi_real2_t);
            auto        u_circuit_m           = tenx::MatrixMap(u_circuit_t);
            cx64        overlap_real1_real2   = psi_real1_v.dot(psi_real2_v);
            cx64        overlap_real1_u_lbit  = psi_real1_v.dot(u_circuit_m * psi_lbit_v);
            cx64        overlap_lbit_ua_real1 = psi_lbit_v.dot(u_circuit_m.adjoint() * psi_real1_v);
            cx64        overlap_lbit_ua_real2 = psi_lbit_v.dot(u_circuit_m.adjoint() * psi_real2_v);
            auto        uadjoint_u            = Eigen::MatrixXcd(u_circuit_m.adjoint() * u_circuit_m);
            std::string err;
            if(std::abs(overlap_real1_real2.real() + overlap_real1_real2.imag() - 1) > status.trnc_lim * 10)
                err += fmt::format("overlap_real1_real2: {:.16f}\n", overlap_real1_real2);
            if(std::abs(overlap_real1_u_lbit.real() + overlap_real1_u_lbit.imag() - 1) > status.trnc_lim * 10)
                err += fmt::format("overlap_real1_u_lbit: {:.16f}\n", overlap_real1_u_lbit);
            if(std::abs(overlap_lbit_ua_real1.real() + overlap_lbit_ua_real1.imag() - 1) > status.trnc_lim * 10)
                err += fmt::format("overlap_lbit_ua_real1: {:.16f}\n", overlap_lbit_ua_real1);
            if(std::abs(overlap_lbit_ua_real2.real() + overlap_lbit_ua_real2.imag() - 1) > status.trnc_lim * 10)
                err += fmt::format("overlap_lbit_ua_real2: {:.16f}\n", overlap_lbit_ua_real2);
            if(not uadjoint_u.isIdentity(status.trnc_lim * 10)) err += fmt::format("uadjoint_u is not an identity matrix");

            if(not err.empty()) throw except::logic_error("Debug error:\n{}", err);
        }
    }
    tools::log->info("Finished {} preprocessing", status.algo_type_sv());
}

template<typename Scalar> void flbit<Scalar>::run_algorithm() {
    if(tensors.state->get_name().empty()) tensors.state->set_name("state_real");
    if(not state_lbit) transform_to_lbit_basis();
    if(time_points.empty()) create_time_points();
    if(cmp_t(status.delta_t.template to_floating_point<cx128>(), 0.0)) update_time_step();
    tools::log->info("Starting {} algorithm with model [{}] for state [{}]", status.algo_type_sv(), enum2sv(settings::model::model_type),
                     tensors.state->get_name());
    if(not tensors.position_is_inward_edge()) throw except::logic_error("Put the state on an edge!");
    if(settings::flbit::run_effective_model) run_algorithm2();
    auto t_run = tid::tic_scope("run", tid::level::normal);
    while(true) {
        update_state();
        check_convergence();
        write_to_file(StorageEvent::ITERATION);
        print_status();
        tools::log->trace("Finished step {}, iter {}, pos {}, dir {}", status.step, status.iter, status.position, status.direction);

        // It's important not to perform the last move, so we break now: that last state would not get optimized
        if(status.algo_stop != AlgorithmStop::NONE) break;
        update_time_evolution_gates();
        status.wall_time = tid::get_unscoped("t_tot").get_time();
        status.algo_time = t_run->get_time();
        t_run->start_lap();
    }
    tools::log->info("Finished {} simulation of state [{}] -- stop reason: {}", status.algo_type_sv(), tensors.state->get_name(), status.algo_stop_sv());
    status.algorithm_has_finished = true;
}

template<typename Scalar> void flbit<Scalar>::run_algorithm_parallel() {
    if(tensors.state->get_name().empty()) tensors.state->set_name("state_real");
    if(not state_lbit) transform_to_lbit_basis();
    if(not state_lbit_init) {
        state_lbit_init = std::make_unique<StateFinite<Scalar>>(*state_lbit);
        tools::finite::mps::normalize_state(*state_lbit_init, std::nullopt, NormPolicy::ALWAYS);
    }
    if(time_points.empty()) create_time_points();
    if(cmp_t(status.delta_t.template to_floating_point<cx128>(), 0.0)) update_time_step();
    tools::log->info("Starting {} algorithm with model [{}] for state [{}]", status.algo_type_sv(), enum2sv(settings::model::model_type),
                     tensors.state->get_name());
    if(not tensors.position_is_inward_edge()) throw except::logic_error("Put the state on an edge!");
    if(settings::flbit::run_effective_model) run_algorithm2();

    auto ham_swap_gates = std::vector<std::vector<qm::SwapGate>>{ham_swap_gates_1body, ham_swap_gates_2body, ham_swap_gates_3body};
    auto status_init    = status;
    auto t_run          = tid::tic_scope("run", tid::level::normal);
#pragma omp parallel for ordered schedule(dynamic, 1)
    for(size_t tidx = 0; tidx < time_points.size(); ++tidx) {
        auto t_step                       = tid::tic_scope("step");
        auto gates_tevo                   = flbit_tmpl::get_time_evolution_gates(time_points[tidx], ham_swap_gates);
        auto state_tevo                   = StateFinite<Scalar>();
        auto status_tevo                  = AlgorithmStatus();
        std::tie(state_tevo, status_tevo) = // Avoid structured binding here due to a bug in clang <= 15
            flbit_tmpl::update_state(tidx, time_points[tidx], *state_lbit_init, gates_tevo, unitary_gates_2site_layers, status_init);
        status_tevo.wall_time = tid::get_unscoped("t_tot").get_time();
        status_tevo.algo_time = t_run->get_time();
        if(not state_tevo.position_is_inward_edge()) throw std::runtime_error("state_tevo is not at the edge! it will not be written to file!");
        flbit_tmpl::print_status(state_tevo, status_tevo);
#pragma omp ordered
        {
            status_tevo.event = StorageEvent::ITERATION;
            tools::finite::h5::save::simulation(*h5file, state_tevo, tensors.get_model(), tensors.get_edges(), status_tevo, CopyPolicy::OFF);
            status_tevo.event = StorageEvent::NONE;
        }
        tools::log->trace("Finished step {}, iter {}, pos {}, dir {}", status_tevo.step, status_tevo.iter, status_tevo.position, status_tevo.direction);
        if(tidx + 1 == time_points.size()) {
            status              = status_tevo;
            tensors.get_state() = state_tevo;
        }
    }
    status.algorithm_has_finished = true;
}

template<typename Scalar> void flbit<Scalar>::run_algorithm2() {
    // Get the interacting Hamiltonian in the diagonal l-bit basis in matrix form
    auto t_run = tid::tic_scope("run");
    auto t_eff = tid::tic_scope("eff");
    auto sites = num::range<size_t>(0, tensors.get_length(), 1);
    tensors.clear_cache();
    tensors.clear_measurements();
    auto svd_cfg             = svd::config();
    svd_cfg.svd_lib          = svd::lib::lapacke;
    svd_cfg.svd_rtn          = svd::rtn::geauto;
    svd_cfg.switchsize_gejsv = 0;
    svd_cfg.switchsize_gesvd = 1;
    svd_cfg.switchsize_gesdd = 16;
    svd_cfg.truncation_limit = status.trnc_lim;
    svd_cfg.rank_max         = status.bond_lim;

    // Define the circuits
    auto circuit_noninteracting = uprop.circuit;
    for(auto &gate : circuit_noninteracting) gate.l = 0; // Set all the lambdas equal to zero to turn off interaction
    const auto  u_and = qm::lbit::get_unitary_2site_gate_layers(circuit_noninteracting);
    const auto &u_mbl = unitary_gates_2site_layers;
    if(u_mbl.size() != 16) throw except::logic_error("u_mbl.size() != 16");
    if(u_and.size() != 16) throw except::logic_error("u_and.size() != 16");
    Eigen::VectorXcd hamiltonian_eff_diagonal;
    {
        const auto mpos = tensors.model->get_compressed_mpos(MposWithEdges::ON);
        //        const auto mpos = tensors.model->get_mpos(MposWithEdges::ON);
        auto t_ham      = tid::tic_scope("ham");
        auto num_states = static_cast<Eigen::Index>(std::pow(2, sites.size()));
        auto bitseqs    = std::vector<size_t>();
        bitseqs.reserve(static_cast<size_t>(num_states) / 2);
        for(auto b : num::range<size_t>(0, num_states)) { bitseqs.emplace_back(b); }
        hamiltonian_eff_diagonal.resize(static_cast<long>(bitseqs.size()));
#pragma omp                      parallel for schedule(dynamic, 16)
        for(size_t idx = 0; idx < bitseqs.size(); ++idx) {
            if(u_mbl.size() != settings::model::lbit::u_depth) throw except::logic_error("u_mbl.size() != u_depth. Has u_mbl been initialized?");
            if(u_and.size() != settings::model::lbit::u_depth) throw except::logic_error("u_and.size() != u_depth. Has u_and been initialized?");

            auto pattern = fmt::format("b{}", tools::get_bitfield(sites.size(), bitseqs[idx],
                                                                                       BitOrder::Reverse)); // Sort states with "1" appearing left to right
            auto state_i = StateFinite<Scalar>(AlgorithmType::fLBIT, sites.size(), 0, 2);
            tools::finite::mps::init::set_product_state_on_axis_using_pattern(state_i, StateInitType::REAL, "z", pattern);
            tools::finite::mps::apply_circuit(state_i, u_and, CircuitOp::NONE, true, GateMove::AUTO, svd_cfg);
            tools::finite::mps::apply_circuit(state_i, u_mbl, CircuitOp::ADJ, true, GateMove::AUTO, svd_cfg);
            hamiltonian_eff_diagonal[static_cast<long>(idx)] = tools::finite::measure::expectation_value<Scalar>(state_i, state_i, mpos);
            if(num::mod<size_t>(idx, bitseqs.size() / 64) == 0)
                tools::log->info("{0:>6}/{1} {2} {3:.3e} s (thread {4:<2}/{5}): ⟨Ψ_i U_and† U_mbl H' U_mbl† U_and |Ψ_i⟩ = {6:.16f}{7:+.16f}", idx,
                                                      bitseqs.size(), pattern, t_eff->get_time(), omp_get_thread_num(), omp_get_num_threads(),
                                                      static_cast<double>(hamiltonian_eff_diagonal[static_cast<long>(idx)].real()),
                                                      static_cast<double>(hamiltonian_eff_diagonal[static_cast<long>(idx)].imag()));
        }
        tools::log->info("Finished calculating hamiltonian_eff_diagonal: {:.3e} s", t_eff->get_last_interval());
                         }
                         {
        tools::log->info("Generating u_and^dagger psi_init");
        auto u_and_adj_state_init = tensors.get_state();
        tools::finite::mps::apply_circuit(u_and_adj_state_init, u_and, CircuitOp::ADJ, true, GateMove::AUTO, svd_cfg);
        const auto u_and_adj_psi_init_tensor = tools::finite::mps::mps2tensor<Scalar>(u_and_adj_state_init);
        const auto u_and_adj_psi_init_vector = tenx::VectorMap(u_and_adj_psi_init_tensor);
        tools::log->info("Starting time evolution");

        auto t_evo = tid::tic_scope("evo");
                     #pragma omp parallel for ordered schedule(dynamic, 1)
        for(size_t tidx = 0; tidx < time_points.size(); ++tidx) {
            auto  time       = time_points[tidx];
            auto  tensor_eff = tensors;
            auto &state_eff  = *tensor_eff.state;
            auto  status_eff = status;

            state_eff.set_name("state_eff");
            status_eff.algo_type = AlgorithmType::fLBIT;
            status_eff.delta_t   = time;
            using namespace std::complex_literals;
            auto t_f64 = std::complex<fp64>(static_cast<fp64>(time.real()), static_cast<fp64>(time.imag()));
            if(t_f64.real() > 1e8 or t_f64.imag() > 1e8) { tools::log->warn("Precision is bad when time > 1e8 | current time == {:.2e}", t_f64); }
            // Generate the time evolution operator in diagonal form
            tools::log->debug("Exponentiating the diagonal Hamiltonians");

            auto tevo_op = [&time](const auto &h) -> cx64 {
#if defined(DMRG_USE_QUADMATH)
                f128_t fmod_th_128 = fmodq(time.real() * fp128(h.real()), atanq(1.0) * 8.0 /* 2 * M_PIq*/);
                return std::exp(-1.0i * static_cast<fp64>(fmod_th_128.value()));
#else
                f128_t fmod_th_128 = fmod(time.real() * fp128(h.real()), 2 * std::numbers::pi_v<fp128>);
#endif
                return std::exp(-1.0i * static_cast<fp64>(fmod_th_128.value()));
            };
            //            Eigen::VectorXcd tevo_eff_diagonal = (-1.0i * t_f64 * hamiltonian_eff_diagonal).array().exp();
            VecType tevo_eff_diagonal = hamiltonian_eff_diagonal.unaryExpr(tevo_op);
            // Time evolve
            tools::log->debug("Time-evolving: {}", t_f64);
            VecType psi_eff = tevo_eff_diagonal.asDiagonal() * u_and_adj_psi_init_vector;
            tools::log->debug("Merging into state");

            auto mps_eff = tenx::TensorCast(psi_eff, psi_eff.size(), 1, 1);
            // Merge these psi into the current state
            tools::finite::mps::merge_multisite_mps(state_eff, mps_eff, sites, 0, MergeEvent::GATE, svd_cfg);
            tools::finite::mps::apply_circuit(state_eff, u_and, CircuitOp::NONE, true, GateMove::ON, svd_cfg);

            // Update stuff
            status_eff.phys_time = abs(time);
            status_eff.iter      = tidx + 1;
            status_eff.step      = status_eff.iter * settings::model::model_size;
            status_eff.position  = state_eff.template get_position<long>();
            status_eff.direction = state_eff.get_direction();
            status_eff.wall_time = tid::get_unscoped("t_tot").get_time();
            status_eff.algo_time = t_eff->get_time();
            if(tidx + 1 >= time_points.size()) status_eff.algo_stop = AlgorithmStop::SUCCESS;

            print_status(status_eff, tensor_eff);
            if(not state_eff.position_is_inward_edge()) throw std::runtime_error("state_eff is not at the edge! it will not be written to file!");
#pragma omp ordered
            {
                status_eff.event = StorageEvent::ITERATION;
                tools::finite::h5::save::simulation(*h5file, state_eff, *tensor_eff.model, *tensor_eff.edges, status_eff, CopyPolicy::TRY);
                status_eff.event = StorageEvent::NONE;
            };
            t_run->start_lap();
        }
    }
}

template<typename Scalar>
void flbit<Scalar>::update_state() {
    /*!
     * \fn void update_state()
     */
    auto delta_t = status.delta_t.template to_floating_point<cx128>();
    tools::log->debug("Starting fLBIT: iter {} | Δt = {:.2e}", status.iter, fp(delta_t));
    if(not state_lbit) throw except::logic_error("state_lbit == nullptr: Set the state in lbit basis before running an flbit step");
    if(not state_lbit_init) {
        state_lbit_init = std::make_unique<StateFinite<Scalar>>(*state_lbit);
        tools::finite::mps::normalize_state(*state_lbit_init, std::nullopt, NormPolicy::ALWAYS);
    }
    *state_lbit = *state_lbit_init;

    // Time evolve from 0 to time_point[iter] here
    auto t_step = tid::tic_scope("step");
    time_evolve_lbit_state();
    transform_to_real_basis();

    tensors.clear_measurements();
    tensors.clear_cache();
    status.phys_time = abs(time_points[std::min(status.iter, time_points.size() - 1)]);

    status.iter += 1;
    status.step += settings::model::model_size;
    status.position  = tensors.template get_position<long>();
    status.direction = tensors.state->get_direction();
}

template<typename Scalar> void flbit<Scalar>::update_time_step() {
    if(time_points.empty()) create_time_points();
    tools::log->trace("Updating time step");
    auto t_updtstep = tid::tic_scope("update_time_step");
    if(status.iter >= time_points.size()) {
        status.delta_t   = time_points.back();
        status.algo_stop = AlgorithmStop::SUCCESS;
        return;
    }
    status.delta_t = time_points[status.iter];
    if(settings::flbit::time_scale == TimeScale::LOGSPACED)
        if(cmp_t(status.delta_t.template to_floating_point<cx128>(), 0.0)) throw except::logic_error("Expected nonzero delta_t after time step update");
    tools::log->debug("Time step iter {} | Δt = {} | t = {:8.2e}", status.iter, status.delta_t.template to_floating_point<cx64>(),
                      status.phys_time.template to_floating_point<fp64>());
}

template<typename Scalar>
void flbit<Scalar>::check_convergence() {
    if(not tensors.position_is_inward_edge()) return;
    auto t_con = tid::tic_scope("check_conv");
    //    check_convergence_entg_entropy();
    if(status.entanglement_saturated_for > 0)
        status.algorithm_saturated_for++;
    else
        status.algorithm_saturated_for = 0;

    status.algorithm_converged_for = status.iter + 1 - std::min(settings::flbit::time_num_steps, status.iter + 1);
    status.algorithm_has_succeeded = status.algorithm_converged_for > 0;
    if(status.algorithm_saturated_for > 0 and status.algorithm_converged_for == 0)
        status.algorithm_has_stuck_for++;
    else
        status.algorithm_has_stuck_for = 0;

    status.algorithm_has_to_stop = false; // Never stop due to saturation

    tools::log->debug("Simulation report: converged {} | saturated {} | stuck {} | succeeded {} | has to stop {}", status.algorithm_converged_for,
                      status.algorithm_saturated_for, status.algorithm_has_stuck_for, status.algorithm_has_succeeded, status.algorithm_has_to_stop);
    status.algo_stop = AlgorithmStop::NONE;
    if(status.iter >= settings::flbit::iter_min) {
        if(status.iter >= settings::flbit::iter_max) status.algo_stop = AlgorithmStop::MAX_ITERS;
        if(status.iter >= settings::flbit::time_num_steps) status.algo_stop = AlgorithmStop::SUCCESS;
        if(status.algorithm_has_to_stop) status.algo_stop = AlgorithmStop::SATURATED;
    }
}

template<typename Scalar>
void flbit<Scalar>::create_time_points() {
    auto  t_crt = tid::tic_scope("create_time_points");
    cx128 time_start(static_cast<fp128>(settings::flbit::time_start_real), static_cast<fp128>(settings::flbit::time_start_imag));
    cx128 time_final(static_cast<fp128>(settings::flbit::time_final_real), static_cast<fp128>(settings::flbit::time_final_imag));
    tools::log->info("Creating time points ({},{}) -> ({},{})", settings::flbit::time_start_real, settings::flbit::time_start_imag,
                     settings::flbit::time_final_real, settings::flbit::time_final_imag);

    cx128 time_diff = time_start - time_final;
    // Check that there will be some time evolution
    if(cmp_t(time_diff, 0.0)) throw except::logic_error("time_start - time_final == 0");
    // Check that the time limits are purely real or imaginary!
    bool time_is_real = abs(time_diff.real()) > 0;
    bool time_is_imag = abs(time_diff.imag()) > 0;
    if(time_is_real and time_is_imag)
        throw except::logic_error("time_start and time_final must both be either purely real or imaginary. Got:\n"
                                  "time_start = {:.8f}\n"
                                  "time_final = {:.8f}",
                                  fp(time_start), fp(time_final));
    time_points.reserve(settings::flbit::time_num_steps);
    if(settings::flbit::time_scale == TimeScale::LOGSPACED) {
        if(time_is_real) {
            for(const auto &t : num::LogSpaced(settings::flbit::time_num_steps, time_start.real(), time_final.real())) { time_points.emplace_back(t); }
        } else {
            for(const auto &t : num::LogSpaced(settings::flbit::time_num_steps, time_start.imag(), time_final.imag())) {
                time_points.emplace_back(cx128(static_cast<fp128>(0.0), t));
            }
        }
    } else if(settings::flbit::time_scale == TimeScale::LINSPACED) {
        if(time_is_real) {
            for(const auto &t : num::LinSpaced(settings::flbit::time_num_steps, time_start.real(), time_final.real())) { time_points.emplace_back(t); }
        } else {
            for(const auto &t : num::LinSpaced(settings::flbit::time_num_steps, time_start.imag(), time_final.imag())) {
                time_points.emplace_back(cx128(static_cast<fp128>(0.0), t));
            }
        }
    }

    //    tools::log->debug("Created {} time points:\n{}", time_points.size(), time_points);
    // Sanity check
    if(time_points.front().real() != static_cast<fp128>(settings::flbit::time_start_real)) throw except::logic_error("Time start real mismatch");
    if(time_points.front().imag() != static_cast<fp128>(settings::flbit::time_start_imag)) throw except::logic_error("Time start imag mismatch");
    if(time_points.back().real() != static_cast<fp128>(settings::flbit::time_final_real)) throw except::logic_error("Time final real mismatch");
    if(time_points.back().imag() != static_cast<fp128>(settings::flbit::time_final_imag)) throw except::logic_error("Time final imag mismatch");
    if(time_points.size() != settings::flbit::time_num_steps)
        throw except::logic_error("Got time_points.size():[{}] != settings::flbit::time_num_steps:[{}]", time_points.size(), settings::flbit::time_num_steps);
}

template<typename Scalar>
void flbit<Scalar>::create_hamiltonian_gates() {
    // Create the hamiltonian gates with n-site terms
    //
    // Note for 2-body terms
    // We want MPO's with only 2-body terms of the Hamiltonian, for distances |i-j| <= J2_span sites.
    // Let's assume we have L==8 and J2_span == 3, then we apply time-evolution
    // L     :  0,1,2,3,4,5,6,7
    // mpo[0]: [0,1,2,3]
    // mpo[1]:   [1,2,3,4]
    // mpo[2]:     [2,3,4,5]
    // mpo[3]:       [3,4,5,6]
    // mpo[4]:         [4,5,6,7]
    // Note that the interaction on sites {1,2} gets applied twice, and {3,4} thrice.
    // To compensate for this, pass a "0" to nbody, which tells the mpo-generator to divide J[i,j] by
    // the number of times it is applied. This ensures that each interaction is time-evolved the right number of times.

    bool has_swap_gates = not ham_swap_gates_1body.empty() or not ham_swap_gates_2body.empty() or not ham_swap_gates_3body.empty();
    bool has_slow_gates = not ham_gates_1body.empty() or not ham_gates_2body.empty() or not ham_gates_3body.empty();

    if(has_swap_gates) tools::log->warn("Hamiltonian swap gates have already been constructed: Normally this is done just once.");
    if(has_slow_gates) tools::log->warn("Hamiltonian gates have already been constructed: Normally this is done just once.");

    ham_gates_1body.clear();
    ham_gates_2body.clear();
    ham_gates_3body.clear();
    ham_gates_Lbody.clear();
    ham_swap_gates_1body.clear();
    ham_swap_gates_2body.clear();
    ham_swap_gates_3body.clear();
    ham_swap_gates_Lbody.clear();

    auto L          = settings::model::model_size;
    auto list_1body = num::range<size_t>(0, L - 0, 1);
    auto list_2body = num::range<size_t>(0, L - 1, 1);
    auto list_3body = num::range<size_t>(0, L - 2, 1);
    if(settings::flbit::use_swap_gates) {
        tools::log->info("Creating Hamiltonian swap gates");
        auto t_swaphamgates = tid::tic_scope("create_ham_gates_swap");
        for(auto pos : list_1body) {
            auto range = 0ul;                                      // Number of site indices
            auto sites = num::range<size_t>(pos, pos + range + 1); // A list of site indices. +1 to include last site (in this case the only site)
            auto nbody = std::vector<size_t>{1};                   // A list of included nbody interaction terms (1: on-site terms, 2: pairwise, and so on)
            auto spins = tensors.state->get_spin_dims(sites);      // A list of spin dimensions for each site (should all be 2 for two-level systems)
            tools::log->debug("Generating {}-body hamiltonian on sites {}", nbody, sites);
            ham_swap_gates_1body.emplace_back(tensors.model->get_multisite_ham_t({pos}, nbody), sites, spins);
        }
        auto J2_ctof = std::min(settings::model::lbit::J2_span,
                                L - 1); // Max distance |i-j| to the furthest interacting site L-1
        for(auto posL : list_2body) {
            auto maxR = std::min<size_t>(posL + J2_ctof, L - 1);
            if(maxR == posL) continue;
            for(auto posR : num::range<size_t>(posL, maxR + 1)) { // maxR+1 to include the last site in range
                if(posL == posR) continue;
                if(posL >= L) throw except::logic_error("posL {} >= L {}", posL, L);
                if(posR >= L) throw except::logic_error("posR {} >= L {}", posR, L);
                auto sites = std::vector<size_t>{posL, posR}; // +1 to include last site
                auto nbody = std::vector<size_t>{2};
                auto spins = tensors.state->get_spin_dims(sites);
                tools::log->debug("Generating {}-body hamiltonian on sites {}", nbody, sites);
                // Accept all swap gates even if all elements are near zero on gates for remote sites,
                // since at large times t these can become relevant again by exp(-itH)
                ham_swap_gates_2body.emplace_back(tensors.model->get_multisite_ham_t(sites, nbody), sites, spins);
            }
            tensors.model->clear_cache();
        }
        // Ignore Hamiltonians with entries smaller than J2_zero: the timescale is too small to resolve them.

        for(auto pos : list_3body) {
            auto range = 2ul;                                                                  // Distance to next-nearest neighbor when 3 sites interact
            auto sites = num::range<size_t>(pos, std::clamp<size_t>(pos + range + 1, 0ul, L)); // +1 to include last site
            auto nbody = std::vector<size_t>{3};
            auto spins = tensors.state->get_spin_dims(sites);
            tools::log->debug("Generating {}-body hamiltonian on sites {}", nbody, sites);
            ham_swap_gates_3body.emplace_back(tensors.model->get_multisite_ham_t(sites, nbody), sites, spins);
        }
        tensors.model->clear_cache();

        if(L <= 6) { // Used for test/debug on small systems
            auto list_Lbody = num::range<size_t>(0, L, 1);
            auto nbody      = std::vector<size_t>{1, 2, 3};
            auto spins      = tensors.state->get_spin_dims(list_Lbody);
            ham_swap_gates_Lbody.emplace_back(tensors.model->get_multisite_ham_t(list_Lbody, nbody), list_Lbody, spins);
        }

        for(const auto &[idx, ham] : iter::enumerate(ham_swap_gates_1body)) {
            if(tenx::isZero(ham.op_t)) tools::log->warn("ham1 swap [{}] is all zeros", idx);
        }
        for(const auto &[idx, ham] : iter::enumerate(ham_swap_gates_2body)) {
            if(tenx::isZero(ham.op_t)) tools::log->warn("ham2 swap [{}] for sites {} is a zero-matrix", idx, ham.pos);
        }
        for(const auto &[idx, ham] : iter::enumerate(ham_swap_gates_3body))
            if(tenx::isZero(ham.op_t)) tools::log->warn("ham3 swap [{}] is all zeros", idx);
    } else {
        tools::log->info("Creating Hamiltonian gates");
        auto t_hamgates = tid::tic_scope("create_ham_gates");
        for(auto pos : list_1body) {
            auto sites = std::vector<size_t>{pos};            // A list of site indices
            auto nbody = std::vector<size_t>{1};              // A list of included nbody interaction terms (1: on-site terms, 2: pairwise, and so on)
            auto spins = tensors.state->get_spin_dims(sites); // A list of spin dimensions for each site (should all be 2 for two-level systems)
            tools::log->debug("Generating {}-body hamiltonian on sites {}", nbody, sites);
            ham_gates_1body.emplace_back(tensors.model->get_multisite_ham_t({pos}, nbody), sites, spins);
        }
        tensors.model->clear_cache();

        auto J2_ctof = std::min(settings::model::lbit::J2_span,
                                L - 1); // Max distance |i-j| to the furthest interacting site L-1
        for(auto posL : list_2body) {
            auto posR = posL + J2_ctof;
            if(J2_ctof == 0) break;
            if(posR >= L) break;
            auto sites = num::range<size_t>(posL, posR + 1); // +1 to include last site, which is otherwise not included in range
            auto nbody = std::vector<size_t>{0, 2};          // zero is a flag to enable compensation for double-counting
            auto spins = tensors.state->get_spin_dims(sites);
            tools::log->debug("Generating {}-body hamiltonian on sites {}", nbody, sites);
            ham_gates_2body.emplace_back(tensors.model->get_multisite_ham_t(sites, nbody), sites, spins);
        }
        tensors.model->clear_cache();

        for(auto posL : list_3body) {
            auto range = 2ul;                                                                    // Distance to next-nearest neighbor when 3 sites interact
            auto sites = num::range<size_t>(posL, std::clamp<size_t>(posL + range + 1, 0ul, L)); // +1 to include last site
            auto nbody = std::vector<size_t>{3};
            auto spins = tensors.state->get_spin_dims(sites);
            tools::log->debug("Generating {}-body hamiltonian on sites {}", nbody, sites);
            ham_gates_3body.emplace_back(tensors.model->get_multisite_ham_t(sites, nbody), sites, spins);
        }
        tensors.model->clear_cache();

        if(L <= 6) { // Used for test/debug on small systems
            auto list_Lbody = num::range<size_t>(0, L - 0, 1);
            auto nbody      = std::vector<size_t>{1, 2, 3};
            auto spins      = tensors.state->get_spin_dims(list_Lbody);
            ham_gates_Lbody.emplace_back(tensors.model->get_multisite_ham_t(list_Lbody, nbody), list_Lbody, spins);
        }
        tensors.model->clear_cache();
        for(const auto &[idx, ham] : iter::enumerate(ham_gates_1body))
            if(tenx::isZero(ham.op_t)) tools::log->warn("ham1[{}] is all zeros", idx);

        for(const auto &[idx, ham] : iter::enumerate(ham_gates_2body))
            if(tenx::isZero(ham.op_t)) tools::log->warn("ham2[{}] is all zeros", idx);

        for(const auto &[idx, ham] : iter::enumerate(ham_gates_3body))
            if(tenx::isZero(ham.op_t)) tools::log->warn("ham3[{}] is all zeros", idx);
    }
    tensors.model->clear_cache();
}

template<typename Scalar>
void flbit<Scalar>::update_time_evolution_gates() {
    // Create the time evolution operators
    if(time_points.empty()) create_time_points();
    update_time_step();
    bool has_swap_gates = not ham_swap_gates_1body.empty() or not ham_swap_gates_2body.empty() or not ham_swap_gates_3body.empty();
    bool has_slow_gates = not ham_gates_1body.empty() or not ham_gates_2body.empty() or not ham_gates_3body.empty();

    if(not has_swap_gates and not has_slow_gates) throw except::logic_error("Hamiltonian gates have not been constructed");
    if(has_swap_gates and has_slow_gates) tools::log->warn("Both swap/non-swap gates have been constructed: Normally only one type should be used");
    auto delta_t = status.delta_t.template to_floating_point<cx128>();
    if(has_swap_gates) {
        auto t_upd = tid::tic_scope("upd_time_evo_swap_gates");
        tools::log->debug("Updating time evolution swap gates to iter {} | Δt = {:.2e}", status.iter, fp(delta_t));
        time_swap_gates_1body = qm::lbit::get_time_evolution_swap_gates(delta_t, ham_swap_gates_1body);
        time_swap_gates_2body = qm::lbit::get_time_evolution_swap_gates(delta_t, ham_swap_gates_2body);
        time_swap_gates_3body = qm::lbit::get_time_evolution_swap_gates(delta_t, ham_swap_gates_3body);
        if(settings::model::model_size <= 6) time_swap_gates_Lbody = qm::lbit::get_time_evolution_swap_gates(delta_t, ham_swap_gates_Lbody);
    }
    if(has_slow_gates) {
        auto t_upd = tid::tic_scope("upd_time_evo_gates");
        tools::log->debug("Updating time evolution gates to iter {} | Δt = {:.2e}", status.iter, fp(delta_t));
        time_gates_1body = qm::lbit::get_time_evolution_gates(delta_t, ham_gates_1body);
        time_gates_2body = qm::lbit::get_time_evolution_gates(delta_t, ham_gates_2body);
        time_gates_3body = qm::lbit::get_time_evolution_gates(delta_t, ham_gates_3body);
        if(settings::model::model_size <= 6) time_gates_Lbody = qm::lbit::get_time_evolution_gates(delta_t, ham_gates_Lbody);
    }
}

template<typename Scalar> void flbit<Scalar>::create_unitary_circuit_gates() {
    if(unitary_gates_2site_layers.size() == settings::model::lbit::u_depth) return;
    if(settings::model::model_type == ModelType::lbit) {
        std::vector<double> fields;
        for(const auto &field : tensors.model->get_parameter("J1_rand")) fields.emplace_back(static_cast<double>(std::any_cast<fp128>(field)));
        unitary_gates_2site_layers.resize(settings::model::lbit::u_depth);
        uprop              = qm::lbit::UnitaryGateProperties(fields);
        uprop.keep_circuit = settings::storage::table::random_unitary_circuit::policy != StoragePolicy::NONE;
        tools::log->info("Creating unitary circuit of 2-site gates {}", uprop.string());
        for(auto &ulayer : unitary_gates_2site_layers) ulayer = qm::lbit::create_unitary_2site_gate_layer(uprop);
        if(settings::flbit::use_mpo_circuit) {
            unitary_gates_mpo_layers.clear();
            for(const auto &ulayer : unitary_gates_2site_layers) unitary_gates_mpo_layers.emplace_back(qm::lbit::get_unitary_mpo_layer(ulayer));
            //        unitary_gates_mpo_layer_full = qm::lbit::merge_unitary_mpo_layers(mpo_layers);
            ledge.resize(1);
            redge.resize(1);
            ledge.setConstant(cx64(1.0, 0.0));
            redge.setConstant(cx64(1.0, 0.0));
        }
    } else {
        // Try to generate the unitary circuit from the microscopic Hamiltonian
    }
}

template<typename Scalar>
void flbit<Scalar>::time_evolve_lbit_state() {
    auto t_evo          = tid::tic_scope("time_evo");
    auto svd_cfg        = svd::config(status.bond_lim, status.trnc_lim);
    svd_cfg.svd_lib     = svd::lib::lapacke;
    svd_cfg.svd_rtn     = svd::rtn::geauto;
    bool has_swap_gates = not time_swap_gates_1body.empty() or not time_swap_gates_2body.empty() or not time_swap_gates_3body.empty();
    bool has_slow_gates = not time_gates_1body.empty() or not time_gates_2body.empty() or not time_gates_3body.empty();
    if(has_swap_gates and has_slow_gates) throw except::logic_error("Both swap and non-swap time evolution gates found");
    if(not has_swap_gates and not has_slow_gates) throw except::logic_error("None of swap or non-swap time evolution gates found");
    auto delta_t = status.delta_t.template to_floating_point<cx128>();
    if(has_swap_gates) {
        tools::log->debug("Applying time evolution swap gates Δt = {:.2e}", fp(delta_t));
        tools::finite::mps::apply_swap_gates(*state_lbit, time_swap_gates_1body, CircuitOp::NONE, GateMove::AUTO, svd_cfg);
        tools::finite::mps::apply_swap_gates(*state_lbit, time_swap_gates_2body, CircuitOp::NONE, GateMove::AUTO, svd_cfg);
        tools::finite::mps::apply_swap_gates(*state_lbit, time_swap_gates_3body, CircuitOp::NONE, GateMove::AUTO, svd_cfg);
    }
    if(has_slow_gates) {
        tools::log->debug("Applying time evolution gates Δt = {:.2e}", fp(delta_t));
        tools::finite::mps::apply_gates(*state_lbit, time_gates_1body, CircuitOp::NONE, true, GateMove::AUTO, svd_cfg);
        tools::finite::mps::apply_gates(*state_lbit, time_gates_2body, CircuitOp::NONE, true, GateMove::AUTO, svd_cfg);
        tools::finite::mps::apply_gates(*state_lbit, time_gates_3body, CircuitOp::NONE, true, GateMove::AUTO, svd_cfg);
    }
    tools::finite::mps::normalize_state(*state_lbit, std::nullopt, NormPolicy::IFNEEDED);

    t_evo.toc();

    if constexpr(settings::debug) {
        // Check that we would get back the original state if we time evolved backwards
        auto state_lbit_debug = *state_lbit;
        if(has_swap_gates) {
            tools::log->debug("Applying time evolution swap gates backward Δt = {:.2e}", fp(delta_t));
            tools::finite::mps::apply_swap_gates(state_lbit_debug, time_swap_gates_3body, CircuitOp::ADJ, GateMove::AUTO, svd_cfg);
            tools::finite::mps::apply_swap_gates(state_lbit_debug, time_swap_gates_2body, CircuitOp::ADJ, GateMove::AUTO, svd_cfg);
            tools::finite::mps::apply_swap_gates(state_lbit_debug, time_swap_gates_1body, CircuitOp::ADJ, GateMove::AUTO, svd_cfg);
        }
        if(has_slow_gates) {
            tools::log->debug("Applying time evolution gates backward Δt = {:.2e}", fp(delta_t));
            tools::finite::mps::apply_gates(state_lbit_debug, time_gates_3body, CircuitOp::ADJ, true, GateMove::AUTO, svd_cfg);
            tools::finite::mps::apply_gates(state_lbit_debug, time_gates_2body, CircuitOp::ADJ, true, GateMove::AUTO, svd_cfg);
            tools::finite::mps::apply_gates(state_lbit_debug, time_gates_1body, CircuitOp::ADJ, true, GateMove::AUTO, svd_cfg);
        }
        tools::finite::mps::normalize_state(state_lbit_debug, std::nullopt, NormPolicy::IFNEEDED);
        auto overlap = tools::finite::ops::overlap<Scalar>(*state_lbit_init, state_lbit_debug);
        tools::log->info("Debug overlap after time evolution: {:.16f}", overlap);
        if(num::gt(std::abs(overlap - RealScalar{1}), 10 * status.trnc_lim))
            throw except::runtime_error("State overlap after backwards time evolution is not 1: Got {:.16f}", fp(overlap));
    }
}

template<typename Scalar>
void flbit<Scalar>::transform_to_real_basis() {
    if(unitary_gates_2site_layers.size() != settings::model::lbit::u_depth) create_unitary_circuit_gates();
    auto t_map      = tid::tic_scope("l2r");
    auto svd_cfg    = svd::config(status.bond_lim, status.trnc_lim);
    svd_cfg.svd_lib = svd::lib::lapacke;
    svd_cfg.svd_rtn = svd::rtn::geauto;
    if(not tensors.state) tensors.state = std::make_unique<StateFinite<Scalar>>(*state_lbit);
    if(settings::flbit::use_mpo_circuit) {
        tensors.get_state() = qm::lbit::transform_to_real_basis(*state_lbit, unitary_gates_mpo_layers, ledge, redge, svd_cfg);

    } else {
        tensors.get_state() = qm::lbit::transform_to_real_basis(*state_lbit, unitary_gates_2site_layers, svd_cfg);
    }
    tensors.clear_measurements();
    tensors.clear_cache();
    status.position  = tensors.template get_position<long>();
    status.direction = tensors.state->get_direction();
}

template<typename Scalar>
void flbit<Scalar>::transform_to_lbit_basis() {
    if(unitary_gates_2site_layers.size() != settings::model::lbit::u_depth) create_unitary_circuit_gates();
    auto t_map      = tid::tic_scope("r2l");
    auto svd_cfg    = svd::config(status.bond_lim, status.trnc_lim);
    svd_cfg.svd_lib = svd::lib::lapacke;
    svd_cfg.svd_rtn = svd::rtn::geauto;
    if(not state_lbit) state_lbit = std::make_unique<StateFinite<Scalar>>(tensors.get_state());
    if(settings::flbit::use_mpo_circuit) {
        *state_lbit = qm::lbit::transform_to_lbit_basis(tensors.get_state(), unitary_gates_mpo_layers, ledge, redge, svd_cfg);
    } else {
        *state_lbit = qm::lbit::transform_to_lbit_basis(tensors.get_state(), unitary_gates_2site_layers, svd_cfg);
    }
    status.position  = tensors.template get_position<long>();
    status.direction = tensors.state->get_direction();
}

template<typename Scalar>
void flbit<Scalar>::write_to_file(StorageEvent storage_event, CopyPolicy copy_policy) {
    AlgorithmFinite<Scalar>::write_to_file(tensors.get_state(), tensors.get_model(), tensors.get_edges(), storage_event, copy_policy);
    //    if(settings::storage::mps::state_lbit::policy != StoragePolicy::NONE) {
    //    if (not state_lbit) transform_to_lbit_basis();
    //    AlgorithmFinite::write_to_file(*state_lbit, tensors.get_model(), tensors.get_edges(), storage_event, copy_policy);
    //    }

    if(h5file and storage_event == StorageEvent::INIT) {
        using namespace settings::flbit;
        auto state_prefix = fmt::format("/fLBIT/{}", tensors.state->get_name());
        h5file->createGroup(state_prefix);
        h5file->writeAttribute(enum2sv(time_scale), state_prefix, "time_scale");
        h5file->writeAttribute(time_num_steps, state_prefix, "time_num_steps");
        h5file->writeAttribute(std::complex(time_start_real, time_start_imag), state_prefix, "time_start");
        h5file->writeAttribute(std::complex(time_final_real, time_final_imag), state_prefix, "time_final");
    }

    // Save the unitaries once
    if(h5file and storage_event == StorageEvent::MODEL) {
        auto t_h5 = tid::tic_scope("h5");
        if(settings::storage::table::random_unitary_circuit::policy == StoragePolicy::INIT) {
            if(uprop.circuit.empty()) throw except::logic_error("The unitary circuit is empty");
            qm::lbit::write_unitary_circuit_parameters(*h5file, "/fLBIT/model/unitary_circuit",
                                                       uprop.circuit); // Writes a table with gate parameters for resuming
            h5file->writeAttribute(uprop.depth, "/fLBIT/model/unitary_circuit", "u_depth");
            h5file->writeAttribute(uprop.fmix, "/fLBIT/model/unitary_circuit", "u_fmix");
            h5file->writeAttribute(uprop.wkind, "/fLBIT/model/unitary_circuit", "u_wkind");
            h5file->writeAttribute(uprop.mkind, "/fLBIT/model/unitary_circuit", "u_mkind");
        }
    }
    //    if(h5file and storage_event == StorageEvent::MODEL) {
    //        auto        t_h5      = tid::tic_scope("h5");
    //        auto        t_event   = tid::tic_scope(enum2sv(storage_event), tid::highest);
    //        auto        t_gates   = tid::tic_scope("gates");
    //        std::string grouppath = "/fLBIT/model/unitary_gates";
    //        if(h5file->linkExists(grouppath)) return;
    //        for(const auto &[idx_layer, layer] : iter::enumerate(unitary_gates_2site_layers)) {
    //            std::string layerpath = fmt::format("{}/layer_{}", grouppath, idx_layer);
    //            for(const auto &[idx_gate, u] : iter::enumerate(layer)) {
    //                std::string gatepath = fmt::format("{}/u_{}", layerpath, idx_gate);
    //                h5file->writeDataset(u.op, gatepath);
    //                h5file->writeAttribute(u.dim, gatepath, "dim");
    //                h5file->writeAttribute(u.pos, gatepath, "pos");
    //            }
    //            h5file->writeAttribute(safe_cast<size_t>(idx_layer), layerpath, "idx_layer");
    //            h5file->writeAttribute(layer.size(), layerpath, "num_gates");
    //        }
    //        h5file->writeAttribute(unitary_gates_2site_layers.size(), grouppath, "num_layers");
    //    }

    // Save the lbit analysis once
    if(h5file and storage_event == StorageEvent::MODEL) {
        auto t_h5    = tid::tic_scope("h5");
        auto t_event = tid::tic_scope(enum2sv(storage_event), tid::highest);
        if(h5file and h5file->linkExists("/fLBIT/model/lbits")) return;
        auto nsamps = settings::flbit::cls::num_rnd_circuits;
        if(nsamps > 0) {
            auto                usites   = std::vector<size_t>{settings::model::model_size};
            auto                udpths   = std::vector<size_t>{settings::model::lbit::u_depth};
            auto                ufmixs   = std::vector<double>{settings::model::lbit::u_fmix};
            auto                ulambdas = std::vector<double>{settings::model::lbit::u_lambda};
            auto                uwkinds  = std::vector<LbitCircuitGateWeightKind>{settings::model::lbit::u_wkind};
            auto                umkinds  = std::vector<LbitCircuitGateMatrixKind>{settings::model::lbit::u_mkind};
            auto                randhf   = settings::flbit::cls::randomize_hfields;
            std::vector<double> fields;
            for(const auto &field : tensors.model->get_parameter("J1_rand")) fields.emplace_back(static_cast<double>(std::any_cast<fp128>(field)));
            auto uprop_default    = qm::lbit::UnitaryGateProperties(fields);
            uprop_default.ulayers = unitary_gates_2site_layers;
            auto lbitSA           = qm::lbit::get_lbit_support_analysis(uprop_default, udpths, ufmixs, ulambdas, uwkinds, umkinds);
            if(h5file and settings::storage::dataset::lbit_analysis::policy == StoragePolicy::INIT) {
                // Put the sample dimension first so that we can collect many simulations in meld along the 0'th dim
                auto label_dist = std::vector<std::string>{"sample", "|i-j|"};
                auto shape_avgs = std::vector<long>{1, lbitSA.corravg.size()};
                auto shape_data =
                    std::vector<long>{safe_cast<long>(nsamps), safe_cast<long>(settings::model::model_size), safe_cast<long>(settings::model::model_size)};

                auto label_data = std::vector<std::string>{"sample", "i", "j"};
                //                if(settings::storage::dataset::lbit_analysis::level >= StorageLevel::LIGHT) {
                h5file->writeDataset(lbitSA.corrmat, "/fLBIT/model/lbits/corrmat", H5D_CHUNKED, shape_data);
                h5file->writeAttribute(label_data, "/fLBIT/model/lbits/corrmat", "dimensions");
                h5file->writeAttribute("The operator support matrix O(i,j) = (1/2^L) Tr(tau_i^z sigma_j^z)", "/fLBIT/model/lbits/corrmat", "description");
                //                }
                //                if(settings::storage::dataset::lbit_analysis::level == StorageLevel::FULL) {
                h5file->writeDataset(lbitSA.cls_avg_fit, "/fLBIT/model/lbits/cls_avg_fit");
                h5file->writeDataset(lbitSA.cls_avg_rms, "/fLBIT/model/lbits/cls_avg_rms");
                h5file->writeDataset(lbitSA.cls_avg_rsq, "/fLBIT/model/lbits/cls_avg_rsq");
                h5file->writeDataset(lbitSA.cls_typ_fit, "/fLBIT/model/lbits/cls_typ_fit");
                h5file->writeDataset(lbitSA.cls_typ_rms, "/fLBIT/model/lbits/cls_typ_rms");
                h5file->writeDataset(lbitSA.cls_typ_rsq, "/fLBIT/model/lbits/cls_typ_rsq");

                h5file->writeDataset(lbitSA.corravg, "/fLBIT/model/lbits/corravg", H5D_CHUNKED, shape_avgs);
                h5file->writeAttribute(label_dist, "/fLBIT/model/lbits/corravg", "dimensions");
                h5file->writeAttribute("Site arithmetic average <<O(|i-j|)>>", "/fLBIT/model/lbits/corravg", "description");

                h5file->writeDataset(lbitSA.corrtyp, "/fLBIT/model/lbits/corrtyp", H5D_CHUNKED, shape_avgs);
                h5file->writeAttribute(label_dist, "/fLBIT/model/lbits/corrtyp", "dimensions");
                h5file->writeAttribute("Site geometric average <<O(|i-j|)>>_typ", "/fLBIT/model/lbits/corrtyp", "description");

                h5file->writeDataset(lbitSA.correrr, "/fLBIT/model/lbits/correrr", H5D_CHUNKED, shape_avgs);
                h5file->writeAttribute(label_dist, "/fLBIT/model/lbits/correrr", "dimensions");
                h5file->writeAttribute("Standard error of <<O(|i-j|)>>", "/fLBIT/model/lbits/correrr", "description");

                h5file->writeDataset(lbitSA.corroff, "/fLBIT/model/lbits/corroff", H5D_CHUNKED, shape_data);
                h5file->writeAttribute(label_data, "/fLBIT/model/lbits/corroff", "dimensions");
                h5file->writeAttribute("The operator support matrix with shifted columns O(i,j) --> O(i,|i-j|)", "/fLBIT/model/lbits/corroff", "description");
                //                }
                h5file->writeAttribute(udpths, "/fLBIT/model/lbits", "u_depth");
                h5file->writeAttribute(ufmixs, "/fLBIT/model/lbits", "u_fmix");
                h5file->writeAttribute(ulambdas, "/fLBIT/model/lbits", "u_lambda");
                h5file->writeAttribute(enum2sv(uwkinds), "/fLBIT/model/lbits", "u_wkind");
                h5file->writeAttribute(enum2sv(umkinds), "/fLBIT/model/lbits", "u_mkind");
                h5file->writeAttribute(nsamps, "/fLBIT/model/lbits", "samples");
                h5file->writeAttribute(randhf, "/fLBIT/model/lbits", "randomize_hfields");
                h5file->writeAttribute(settings::flbit::cls::mpo_circuit_svd_bondlim, "/fLBIT/model/lbits", "u_bond");
                h5file->writeAttribute(settings::flbit::cls::mpo_circuit_svd_trnclim, "/fLBIT/model/lbits", "u_trnc");
                h5file->writeAttribute(settings::flbit::cls::mpo_circuit_switchdepth, "/fLBIT/model/lbits", "mpo_switchdepth");
            }
        }
        if(settings::flbit::cls::exit_when_done) exit(0);
    }
    // Save the lbit density matrix analysis
    auto num_rps = settings::flbit::opdm::num_rps; // Number of random product states
    if(num_rps > 0 and h5file and storage_event == StorageEvent::MODEL) {
        auto length     = safe_cast<long>(settings::model::model_size);
        auto svd_cfg    = svd::config(8192, 1e-12);
        svd_cfg.svd_lib = svd::lib::lapacke;
        svd_cfg.svd_rtn = svd::rtn::gejsv;
        using namespace qm::spin::half::tensor;
        using op_t       = tools::finite::measure::LocalObservableOp<cx64>;
        using opstring_t = std::vector<op_t>;
        auto eigvals_all = Eigen::MatrixXd(num_rps, length);
        for(auto nrps : num::range(0, num_rps)) {
            auto eig_sol        = eig::solver();
            auto pattern        = std::string();
            auto state_lbit_rps = StateFinite<Scalar>(AlgorithmType::fLBIT, settings::model::model_size, 0);
            tools::finite::mps::initialize_state(state_lbit_rps, StateInit::PRODUCT_STATE_NEEL_SHUFFLED, StateInitType::REAL, "+z", false, 1, pattern);
            tools::finite::mps::normalize_state(state_lbit_rps, svd_cfg, NormPolicy::ALWAYS);
            auto state_real_rps = qm::lbit::transform_to_real_basis(state_lbit_rps, unitary_gates_2site_layers,
                                                                    svd_cfg); // Applies U^\dagger
            tools::finite::mps::normalize_state(state_real_rps, svd_cfg, NormPolicy::ALWAYS);

            auto rho = Eigen::Tensor<cx64, 2>(length, length);
            rho.setZero();
            // Now we make measurements on every pair of sites
            for(long pos_i = 0; pos_i < length; ++pos_i) {
                for(long pos_j = pos_i; pos_j < length; ++pos_j) {
                    // Create an operator string from pos_i to pos_j, where
                    //      pos_i has sp,
                    //      pos_j has sm,
                    // insert szm from pos_i (including) to pos_j (excluding).
                    auto opstring = opstring_t{op_t{sp, pos_i}}; // sigma+_i sigma-_j
                    if(pos_i < pos_j) {
                        for(auto pos_x : num::range(pos_i, pos_j)) opstring.emplace_back(op_t{sz, pos_x});
                    }
                    opstring.emplace_back(op_t{sm, pos_j});
                    rho(pos_i, pos_j) = tools::finite::measure::expectation_value<cx64>(state_real_rps, opstring);
                    if(pos_i != pos_j) rho(pos_j, pos_i) = std::conj(rho(pos_i, pos_j));
                }
            }
            auto rho_matrix = tenx::MatrixMap(rho);
            auto rho_trace  = rho_matrix.trace();
            tools::log->debug("rho trace: {:.16f}", rho_trace);
            if(not rho_matrix.isApprox(rho_matrix.adjoint())) throw except::logic_error("rho is not hermitian");
            eig_sol.eig<eig::Form::SYMM>(rho.data(), rho.dimension(0), eig::Vecs::OFF);
            auto eigvals = eig::view::get_eigvals<fp64>(eig_sol.result);
            tools::log->info("opdm eigv {:2} {}: {::.2e} | sum {:.15f}", nrps, pattern, eigvals, eigvals.sum());
            if(std::abs(rho_trace - static_cast<double>(length) / 2.0) > 1e-12) throw except::logic_error("R does not have trace L/2");
            if(eigvals.real().maxCoeff() > 1 + 1e-8) throw except::logic_error("The largest eigenvalue is larger than 1");
            if(eigvals.real().minCoeff() < 0 - 1e-8) throw except::logic_error("The smallest eigenvalue is smaller than 0");
            eigvals_all.row(nrps) = eigvals.cwiseAbs();
        }
        Eigen::VectorXd average = eigvals_all.array().colwise().mean();
        tools::log->info("opdm average: {::.15f}", average.transpose());
        h5file->writeDataset(average, "/fLBIT/model/opdm-eigv");
        h5file->writeAttribute("eigenvalues of the one-paricle densiy matrix: an exact step-function means non-interacting", "/fLBIT/model/opdm-eigv",
                               "description");
        if(settings::flbit::opdm::exit_when_done) exit(0);
    }
}

template<typename Scalar>
void flbit<Scalar>::print_status(const AlgorithmStatus &st, const TensorsFinite<Scalar> &ts) {
    if(num::mod(st.iter, settings::print_freq(st.algo_type)) != 0) return;
    if(settings::print_freq(st.algo_type) == 0) return;
    auto        t_print = tid::tic_scope("print");
    std::string report;
    report += fmt::format("{:<} ", ts.state->get_name());
    report += fmt::format("iter:{:<4} ", st.iter);
    report += fmt::format("step:{:<5} ", st.step);
    report += fmt::format("L:{} ", ts.get_length());
    if(ts.active_sites.empty())
        report += fmt::format("l:{:<2} ", ts.get_position());
    else if(ts.state->get_direction() > 0)
        report += fmt::format("l:[{:>2}-{:<2}] ", ts.active_sites.front(), ts.active_sites.back());
    else if(ts.state->get_direction() < 0)
        report += fmt::format("l:[{:>2}-{:<2}] ", ts.active_sites.back(), ts.active_sites.front());
    //    report += fmt::format("E/L:{:<20.16f} ", tools::finite::measure::energy_per_site(tensors));
    report += fmt::format("ε:{:<8.2e} ", ts.state->get_truncation_error_midchain());
    report += fmt::format("Sₑ(L/2):{:<18.16f} ", tools::finite::measure::entanglement_entropy_midchain(*ts.state));
    report += fmt::format("Sₙ(L/2):{:<18.16f} ", tools::finite::measure::number_entropy_midchain(*ts.state));
    //    if(ts.state->measurements.number_entropy_midchain) // This one is expensive
    //        report += fmt::format("Sₙ(L/2):{:<18.16f} ", ts.state->measurements.number_entropy_midchain.value());

    //    if(state_lbit->measurements.number_entropy_midchain) // This one is expensive
    //        report += fmt::format("Sₙ(L/2):{:<10.8f} ", state_lbit->measurements.number_entropy_midchain.value());

    report += fmt::format("χ:{:<3}|{:<3}|{:<3} ", st.bond_max, st.bond_lim, tools::finite::measure::bond_dimension_midchain(*ts.state));
    if(settings::flbit::time_scale == TimeScale::LOGSPACED)
        report += fmt::format("ptime:{:<} ", fmt::format("{:>.2e}s", st.phys_time.to_floating_point<fp64>()));
    if(settings::flbit::time_scale == TimeScale::LINSPACED)
        report += fmt::format("ptime:{:<} ", fmt::format("{:>.6f}s", st.phys_time.to_floating_point<fp64>()));
    report += fmt::format("wtime:{:<}(+{:<}) ", fmt::format("{:>.1f}s", tid::get_unscoped("t_tot").get_time()),
                          fmt::format("{:>.1f}s", tid::get_unscoped("fLBIT")["run"].get_lap()));
    report += fmt::format("mem[rss {:<.1f}|peak {:<.1f}|vm {:<.1f}]MB ", debug::mem_rss_in_mb(), debug::mem_hwm_in_mb(), debug::mem_vm_in_mb());
    tools::log->info(report);
}

template<typename Scalar> void flbit<Scalar>::print_status() { print_status(status, tensors); }
