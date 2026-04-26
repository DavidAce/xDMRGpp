#pragma once
#include "../fdmrg.h"
#include "../xdmrg.h"
#include "config/enums/AlgorithmStop.h"
#include "config/enums/AlgorithmType.h"
#include "config/enums/CopyPolicy.h"
#include "config/enums/fdmrg_task.h"
#include "config/enums/LogPolicy.h"
#include "config/enums/MergeEvent.h"
#include "config/enums/MpoCompress.h"
#include "config/enums/MposWithEdges.h"
#include "config/enums/OptExit.h"
#include "config/enums/OptRitz.h"
#include "config/enums/OptSolver.h"
#include "config/enums/ResetReason.h"
#include "config/enums/ResumePolicy.h"
#include "config/enums/StateInit.h"
#include "config/enums/StorageEvent.h"
#include "config/enums/xdmrg_task.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "io/fmt_custom.h"
#include "math/eig.h"
#include "math/num.h"
#include "math/svd.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction/contraction_policy.h"
#include "tools/common/contraction/matrix_vector_product.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/common/prof.h"
#include "tools/finite/bex.h"
#include "tools/finite/bex/BondExpansionConfig.h"
#include "tools/finite/bex/BondExpansionResult.h"
#include "tools/finite/h5.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/mps.h"
#include "tools/finite/multisite.h"
#include "tools/finite/ops.h"
#include "tools/finite/opt.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
#include "tools/finite/pos.h"
#include <h5pp/h5pp.h>

template<typename Scalar>
xdmrg<Scalar>::xdmrg(std::shared_ptr<h5pp::File> h5ppFile_) : AlgorithmFinite<Scalar>(std::move(h5ppFile_), settings::xdmrg::ritz, AlgorithmType::xDMRG) {
    tools::log->trace("Constructing class_xdmrg");
    tensors.state->set_name("state_emid");
}

template<typename Scalar>
void xdmrg<Scalar>::resume() {
    // Resume can imply many things
    // 1) Resume a simulation which terminated prematurely
    // 2) Resume a previously successful simulation. This may be desireable if the config
    //    wants something that is not present in the file.
    //      a) A certain number of states
    //      b) A state inside a particular energy window
    //      c) The ground or "roof" states
    // To guide the behavior, we check the setting ResumePolicy.

    auto states_that_may_resume =
        tools::common::h5::resume::find_states_that_may_resume(*h5file, settings::storage::resume_policy, status.algo_type, "state_emid");
    if(states_that_may_resume.empty()) throw except::state_error("no resumable states were found");
    for(const auto &[state_prefix, algo_stop] : states_that_may_resume) {
        tools::log->info("Resuming [{}] | previous stop reason: {} | resume policy: {} ", state_prefix, enum2sv(algo_stop),
                         enum2sv(settings::storage::resume_policy));
        try {
            tools::finite::h5::load::simulation(*h5file, state_prefix, tensors, status, status.algo_type);
        } catch(const except::load_error &le) {
            tools::log->warn("Load error: {}", le.what());
            // continue;
        }

        // Our first task is to decide on a state name for the newly loaded state
        // The simplest is to infer it from the state prefix itself
        auto name = tools::common::h5::resume::extract_state_name(state_prefix);
        tensors.state->set_name(name);

        // Reload the bond and truncation error limits (could be different in the config compared to the status we just loaded)
        double long_max                   = static_cast<double>(std::numeric_limits<long>::max());
        double bond_max                   = std::min(long_max, std::pow(2.0, settings::model::model_size / 2));
        status.bond_max                   = std::min({status.bond_max, safe_cast<long>(bond_max), settings::get_bond_max(status.algo_type)});
        status.bond_min                   = std::max(status.bond_min, settings::get_bond_min(status.algo_type));
        status.bond_lim                   = std::clamp(status.bond_lim, 1l, status.bond_max);
        status.bond_limit_has_reached_max = status.bond_lim == status.bond_max;
        tools::log->info("Initialized bond dimension limits: min {} lim {} max {}", status.bond_min, status.bond_lim, status.bond_max);

        status.trnc_min                   = settings::solvers::svd::truncation_min;
        status.trnc_max                   = settings::solvers::svd::truncation_max;
        status.trnc_lim                   = std::clamp(status.trnc_lim, status.trnc_min, status.trnc_max);
        status.trnc_limit_has_reached_min = status.trnc_lim == status.trnc_min;
        tools::log->info("Initialized truncation error limits: max {:8.2e} lim {:8.2e} min {:8.2e}", status.trnc_max, status.trnc_lim, status.trnc_min);

        status.opt_ritz = settings::xdmrg::ritz;

        // Apply shifts and compress the model
        tools::finite::pos::move_center_point_to_inward_edge(tensors);
        set_parity_shift_mpo();
        set_parity_shift_mpo_squared();
        set_energy_shift_mpo();
        rebuild_tensors(); // Rebuilds and compresses mpos, then rebuilds the environments
        update_precision_limit();
        update_dmrg_blocksize();
        // Initialize a custom task list
        std::deque<xdmrg_task> task_list;

        if(status.algorithm_has_succeeded)
            task_list = {xdmrg_task::POST_PRINT_RESULT};
        else
            task_list = {xdmrg_task::INIT_CLEAR_CONVERGENCE, xdmrg_task::FIND_EXCITED_STATE,
                         xdmrg_task::POST_DEFAULT}; // Probably a savepoint. Simply "continue" the algorithm until convergence
        run_task_list(task_list);
    }
}

template<typename Scalar>
void xdmrg<Scalar>::run_default_task_list() {
    std::deque<xdmrg_task> default_task_list = {
        xdmrg_task::INIT_DEFAULT,
        xdmrg_task::FIND_EXCITED_STATE,
        xdmrg_task::POST_DEFAULT,
    };

    run_task_list(default_task_list);
}

template<typename Scalar>
void xdmrg<Scalar>::run_task_list(std::deque<xdmrg_task> &task_list) {
    while(not task_list.empty()) {
        auto task = task_list.front();
        switch(task) {
            case xdmrg_task::INIT_RANDOMIZE_MODEL: initialize_model(); break;
            case xdmrg_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE: initialize_state(ResetReason::INIT, StateInit::RANDOM_PRODUCT_STATE); break;
            case xdmrg_task::INIT_RANDOMIZE_INTO_ENTANGLED_STATE: initialize_state(ResetReason::INIT, StateInit::RANDOM_ENTANGLED_STATE); break;
            case xdmrg_task::INIT_RANDOMIZE_FROM_CURRENT_STATE: initialize_state(ResetReason::INIT, StateInit::RANDOMIZE_PREVIOUS_STATE); break;
            case xdmrg_task::INIT_BOND_LIMITS: init_bond_dimension_limits(); break;
            case xdmrg_task::INIT_TRNC_LIMITS: init_truncation_error_limits(); break;
            case xdmrg_task::INIT_ENERGY_TARGET: init_energy_target(); break;
            case xdmrg_task::INIT_WRITE_MODEL: write_to_file(StorageEvent::MODEL); break;
            case xdmrg_task::INIT_CLEAR_STATUS: status.clear(); break;
            case xdmrg_task::INIT_CLEAR_CONVERGENCE: clear_convergence_status(); break;
            case xdmrg_task::INIT_DEFAULT: run_preprocessing(); break;

            case xdmrg_task::FIND_ENERGY_RANGE: find_energy_range(); break;
            case xdmrg_task::FIND_EXCITED_STATE:
                tensors.state->set_name("state_emid");
                run_algorithm();
                break;
            case xdmrg_task::POST_WRITE_RESULT: write_to_file(StorageEvent::FINISHED, CopyPolicy::FORCE); break;
            case xdmrg_task::POST_PRINT_RESULT: print_status_full(); break;
            case xdmrg_task::POST_PRINT_TIMERS: tools::common::timer::print_timers(); break;
            case xdmrg_task::POST_RBDS_ANALYSIS: run_rbds_analysis(); break;
            case xdmrg_task::POST_RTES_ANALYSIS: run_rtes_analysis(); break;
            case xdmrg_task::POST_DEFAULT: run_postprocessing(); break;
            case xdmrg_task::TIMER_RESET: tid::reset("xDMRG"); break;
        }
        task_list.pop_front();
    }
    if(not task_list.empty()) {
        for(auto &task : task_list) tools::log->critical("Unfinished task: {}", enum2sv(task));
        throw except::runtime_error("Simulation ended with unfinished tasks");
    }
}

template<typename Scalar>
void xdmrg<Scalar>::init_energy_target(std::optional<double> energy_density_target) {
    switch(status.opt_ritz) {
        case OptRitz::NONE: throw std::logic_error("status.opt_ritz == OptRitz::NONE is invalid under xdmrg");
        case OptRitz::SR: {
            // tools::log->warn("status.opt_ritz == OptRitz::SR should be handled with fdmrg instead of xdmrg");
            status.energy_tgt = 0.0;
            break;
            // throw std::logic_error("status.opt_ritz == OptRitz::SR should be handled with fdmrg instead of xdmrg");
        }
        case OptRitz::LR: {
            status.energy_tgt = 0.0;
            break;
            // throw std::logic_error("status.opt_ritz == OptRitz::LR should be handled with fdmrg instead of xdmrg");
        }
        case OptRitz::LM: {
            status.energy_tgt = 0.0;
            break;
            // throw std::logic_error("status.opt_ritz == OptRitz::LR should be handled with fdmrg instead of xdmrg");
        }
        case OptRitz::SM: {
            // When the Hamiltonian is traceless, the energy level nearest zero is closest to the infinite-temperature limit.
            // Therefore, we expect the energy target to be == 0. However, in some cases we get a symmetric energy spectrum, with
            // every energy level having a counterpart with opposite sign (e.g. Ising-Majorana with g == 0).
            // In this case we can break the degeneracy by setting a tiny shift ~1e-10 to bias xDMRG towards one of the states closest to 0.
            status.energy_tgt = settings::xdmrg::energy_spectrum_shift;
            break;
        }
        case OptRitz::IS: {
            status.energy_tgt = static_cast<double>(tools::finite::measure::energy(tensors)); // Should take the energy from the initial state
            break;
        }
        case OptRitz::TE: {
            if(not energy_density_target) energy_density_target = settings::xdmrg::energy_density_target;
            if(energy_density_target.value() < 0.0 or energy_density_target.value() > 1.0)
                throw except::runtime_error(fmt::format(
                    "xdmrg::init_energy_target: with OptRitz::TE: invalid energy_density_target: Expected value in range [0.0 - 1.0], got: [{:.8f}]",
                    energy_density_target.value()));
            // Set energy boundaries. This function is supposed to run after find_energy_range!
            if(status.energy_max == status.energy_min)
                throw except::runtime_error("xdmrg::init_energy_target: with OptRitz::TE Failed because energy_max == {} and energy_min == {}\n"
                                            "Try running find_energy_range() first",
                                            status.energy_max, status.energy_min);

            status.energy_dens_target = energy_density_target.value();
            status.energy_tgt         = status.energy_min + status.energy_dens_target * (status.energy_max - status.energy_min);
            tools::log->info("Energy minimum     = {:.8f}", status.energy_min);
            tools::log->info("Energy maximum     = {:.8f}", status.energy_max);
            tools::log->info("Energy target      = {:.8f}", status.energy_tgt);
            break;
        }
    }
}

template<typename Scalar>
void xdmrg<Scalar>::run_preprocessing() {
    tools::log->info("Running {} preprocessing", status.algo_type_sv());
    auto t_pre = tid::tic_scope("pre");
    status.clear();
    init_bond_dimension_limits();
    init_truncation_error_limits();
    initialize_model(); // First use of random!

    initialize_state(ResetReason::INIT, settings::state::init::initial_state); // Second use of random!
    tensors.get_state().assert_validity();
    find_energy_range();
    tensors.get_state().assert_validity();
    init_energy_target();
    tensors.get_state().assert_validity();
    set_parity_shift_mpo();
    set_parity_shift_mpo_squared();
    set_energy_shift_mpo();
    rebuild_tensors(); // Rebuilds and compresses mpos, then rebuilds the environments
    update_precision_limit();
    write_to_file(StorageEvent::MODEL);

    // auto imodel = tools::finite::mpo::get_inverted_mpos(tensors.model->get_all_mpo_tensors(MposWithEdges::ON));

    if(tensors.template get_length<long>() <= 4) {
        // Print the spectrum if small
        // tensors.clear_cache();
        auto svd_solver = svd::solver();
        auto L          = tensors.template get_length<long>();
        auto sites      = num::range<size_t>(0, L);
        auto ham1       = tensors.model->template get_multisite_ham<Scalar>(sites);
        auto ham2       = tensors.model->template get_multisite_ham_squared<Scalar>(sites);
        auto norm_est   = tensors.model->get_energy_upper_bound();
        // auto        ham1i      = svd_solver.pseudo_inverse(ham1_);
        eig::solver solver1, solver2;
        solver1.eig<eig::Form::SYMM>(ham1.data(), ham1.dimension(0), eig::Vecs::OFF);
        solver2.eig<eig::Form::SYMM>(ham2.data(), ham2.dimension(0), eig::Vecs::OFF);
        // solver1i.eig<eig::Form::SYMM>(ham1i.data(), ham1i.dimension(0));

        auto    evals1     = eig::view::get_eigvals<RealScalar>(solver1.result);
        auto    evals2     = eig::view::get_eigvals<RealScalar>(solver2.result);
        VecReal diffs1     = VecReal::Zero(evals1.size());
        VecReal diffs2     = VecReal::Zero(evals1.size());
        auto    N1         = evals1.size() - 1;
        auto    N2         = evals2.size() - 1;
        diffs1.topRows(N1) = (evals1.tail(N1) - evals1.head(N1));
        diffs2.topRows(N2) = (evals2.tail(N2) - evals2.head(N2));

        // auto evals1i = eig::view::get_eigvals<fp64>(solver1i.result);
        fmt::print("{:^8} {:<20}\n", " ", "H¹");
        for(long idx = 0; idx < evals1.size(); ++idx) {
            // if(std::abs(evals1[idx]) > 1.1) continue;
            fmt::print("idx {:2}: {:20.16f} {:>10.3e}\n", idx, fp(evals1[idx]), fp(diffs1[idx]));
        }
        fmt::print("{:^8} {:<20} {:<20} {:<20}\n", " ", "H²", "diff", "sqrt(H²)");
        for(long idx = 0; idx < evals2.size(); ++idx) {
            // if(std::abs(evals2[idx]) > 1.1) continue;
            fmt::print("idx {:2}: {:20.16f} {:>10.3e} {:20.16f}\n", idx, fp(evals2[idx]), fp(diffs2[idx]), fp(std::sqrt(evals2[idx])));
        }
        auto h5file = h5pp::File("../../output/spectrum.h5", h5pp::FileAccess::RENAME);
        h5file.writeDataset(evals1, "H_evals");
        h5file.writeDataset(diffs1, "H_diffs");
        h5file.writeDataset(evals2, "H2_evals");
        h5file.writeDataset(diffs2, "H2_diffs");
        // fmt::print("{:^8} {:<20} {:<20}\n", " ", "H¹", "H⁻¹");
        // for(long idx = 0; idx < std::min(evals1_.size(), evals1i.size()); ++idx) {
        //     fmt::print("idx {:2}: {:20.16f} {:20.16f}\n", idx, evals1_[idx], evals1i[idx]);
        // }
        fmt::print("\n");
        fmt::print("Hamiltonian norm estimate: {:.16f}\n", fp(norm_est));
        // Try the iterative scheme
        // auto impos = tools::finite::mpo::get_inverted_mpos(tensors.model->get_compressed_mpos_squared(MposWithEdges::ON));
        // auto impos = tools::finite::mpo::get_inverted_mpos(tensors.model->get_all_mpo_tensors(MposWithEdges::ON));

        // auto imodel = *tensors.model;
        // for(auto &&[pos, impo] : iter::enumerate(imodel.MPO)) impo->set_mpo(impos[pos]);
        // auto ham3i = imodel.get_multisite_ham(sites);
        // // auto ham3i = svd::solver::pseudo_inverse(ham3i);
        // eig::solver solver3i;
        // solver3i.eig<eig::Form::SYMM>(ham3i.data(), ham3i.dimension(0));
        // auto evals3i = eig::view::get_eigvals<fp64>(solver3i.result);

        // fmt::print("{:^8} {:<20} {:<20} {:<20} {:<20}\n", " ", "H¹", "H⁻¹", "H⁻¹(iter)", "diff");
        // for(long idx = 0; idx < std::min(evals1i.size(), evals3i.size()); ++idx) {
        //     fmt::print("idx {:2}: {:20.16f} {:20.16f} {:20.16f} {:.4e}\n", idx, evals1_[idx], evals1i[idx], evals3i[idx],
        //                std::abs(evals1i[idx] - evals3i[idx]));
        // }
        // fmt::print("\n");

        // tools::log->info("Iterative inverse  H⁻¹");
        // for(long idx = 0; idx < evals3i.size(); ++idx) { fmt::print("idx {:2}: {:20.16f}\n", idx, evals3i[idx]); }
        // exit(0);
    }
    tools::log->info("Finished {} preprocessing", status.algo_type_sv());
}

template<typename Scalar>
void xdmrg<Scalar>::run_algorithm() {
    if(tensors.state->get_name().empty()) tensors.state->set_name("state_emid");
    tools::log->info("Starting {} simulation of model [{}] for state [{}] with ritz [{}]", status.algo_type_sv(), enum2sv(settings::model::model_type),
                     tensors.state->get_name(), enum2sv(status.opt_ritz));
    auto t_run       = tid::tic_scope("run");
    status.algo_stop = AlgorithmStop::NONE;

    auto get_precision = [&]() {
        auto       position = tensors.template get_position<long>();
        RealScalar h2norm   = H_norm_estimate * H_norm_estimate;
        RealScalar quot     = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar vh2v     = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar eps      = std::numeric_limits<RealScalar>::epsilon();
        if(position >= 0 and h2norm > RealScalar{0}) {
            vh2v = std::abs(tools::finite::measure::expval_hamiltonian_squared(tensors));
            quot = vh2v / h2norm; // Assume the state is normalized (vv = 1)
        }
        if(quot < eps * 1000 or !std::isfinite(vh2v)) {
            tools::log->info("Switched to X2 precision: <H²>/|H²| = {:.16e} / {:.16e} = {:.4e} < 1000 * eps", vh2v, h2norm, quot);
            return ContractionPrecision::X2;
        } else {
            tools::log->debug("Selected TBLIS backend: <H²>/|H²| = {:.16e} / {:.16e} = {:.4e} > 1000 * eps", vh2v, h2norm, quot);
            return ContractionPrecision::SAME;
        }
    };

    while(true) {
        auto precision = get_precision();
        auto h1info    = SetH1MvInfo(precision);
        auto h2info    = SetH2MvInfo(precision);
        auto envinfo   = SetEnvInfo(precision);

        tools::log->trace("Starting step {}, iter {}, pos {}, dir {}, precision:{}", status.step, status.iter, status.position, status.direction,
                          enum2sv(precision));
        // Apply end-of-half-sweep actions
        // Updating bond dimension must go first since it decides based on truncation error, but a projection+normalize resets truncation.
        update_bond_dimension_limit();   // Updates the bond dimension if the state precision is being limited by bond dimension
        update_truncation_error_limit(); // Updates the truncation error limit if the state is being truncated
        update_mixing_factor();          // Updates the mixing factor used in DMRG3S
        set_energy_shift_mpo();          // Shifts the MPOs by the target energy used in the folded/generalized objective.
        set_parity_shift_mpo();          // Shifts the energy spectrum of states with opposite parity away from the current energy.
        set_parity_shift_mpo_squared();  // Shifts the energy-squared spectrum of states with opposite parity up by 1 (makes sense with ritz == SM)
        rebuild_tensors();               // Rebuilds mpos (and compresses them) and edges, only if they were modified.
        try_projection();                // Tries to project the state to the nearest global spin parity sector along settings::state::sector::target_axis
        try_mps_compression();           // Tries to compress all the MPS bond dimensions without sacrificing too much precision

        // Perform the step
        update_state();
        print_status();

        check_convergence();
        write_to_file();

        tools::log->trace("Finished iter {}, step {}, pos {}, dir {}", status.iter, status.step, status.position, status.direction);

        // It's important not to perform the last move, so we break now: that last state would not get optimized
        if(status.algo_stop != AlgorithmStop::NONE) break;
        update_eigs_tolerance(); // Updates the tolerance on the iterative eigensolver
        update_dmrg_blocksize(); // Updates the number sites used in dmrg steps using the information typical scale
        // Prepare for the next step

        move_center_point(); // Moves the center point AC to the next site and increments status.iter and status.step
        status.wall_time = tid::get_unscoped("t_tot").get_time();
        status.algo_time = t_run->get_time();
    }
    tools::log->info("Finished {} simulation of state [{}] -- stop reason: {}", status.algo_type_sv(), tensors.state->get_name(), status.algo_stop_sv());
    status.algorithm_has_finished = true;
    //    tools::finite::measure::parity_components(*tensors.state, qm::spin::half::sz);
}

template<typename Scalar>
void xdmrg<Scalar>::update_state() {
    using namespace tools::finite;
    using namespace tools::finite::opt;
    auto t_step = tid::tic_scope("step");
    {
        auto h1info      = SetH1MvInfo(ContractionBackend::TBLIS);
        auto h2info      = SetH2MvInfo(ContractionBackend::TBLIS);
        auto envinfo     = SetEnvInfo(ContractionPrecision::X2);
        auto bexp_result = tools::finite::bex::expand_bonds(tensors, get_bond_expansion_config(BondExpansionOrder::PREOPT));
    }

    auto opt_meta = get_opt_meta();

    tools::log->debug("Starting {} iter {} | step {} | pos {} | dir {} | ritz {} | type {}", status.algo_type_sv(), status.iter, status.step, status.position,
                      status.direction, enum2sv(settings::get_ritz(status.algo_type)), enum2sv(opt_meta.optType));
    // Try activating the sites asked for;
    tools::finite::pos::activate_sites(tensors, opt_meta.chosen_sites);
    if(tensors.active_sites.empty()) {
        tools::log->debug("No more sites to activate");
        return;
    }

    tensors.rebuild_edges();

    // auto h1norm_krylov = tools::finite::measure::local_hamiltonian_norm(tensors, 10, 1e-3f);
    // auto h2norm_krylov = tools::finite::measure::local_hamiltonian_squared_norm(tensors, 10, 1e-3f);
    // auto mpo1_dims     = tensors.model->get_mpo_active().front().get().MPO().dimensions();
    // auto mpo2_dims     = tensors.model->get_mpo_active().front().get().MPO2().dimensions();
    // auto h1info        = SetH1MvInfo(mpo1_dims, h1norm_krylov); // Use more accurate matvec
    // auto h2info        = SetH2MvInfo(mpo2_dims, h2norm_krylov); // Use more accurate matvec
    // auto enve          = tensors.get_edges().get_ene_active();
    // auto envv          = tensors.get_edges().get_var_active();
    // auto enve_max_norm = std::max(tenx::norm(enve.L.get_block()), tenx::norm(enve.R.get_block()));
    // auto envv_max_norm = std::max(tenx::norm(envv.L.get_block()), tenx::norm(envv.R.get_block()));
    // auto h1norm_env    = std::min(enve_max_norm, H_norm_estimate);
    // auto h2norm_env    = std::min(envv_max_norm, H_norm_estimate * H_norm_estimate);

    // tools::log->info("Estimated H1_local norm:  krylov={:.4e} envs={:.4e} ", fp(h1norm_krylov), fp(h1norm_env));
    // tools::log->info("Estimated H2_local norm:  krylov={:.4e} envs={:.4e} ", fp(h2norm_krylov), fp(h2norm_env));

    tools::log->debug("Updating state: {}", opt_meta.string()); // Announce the current configuration for optimization

    // Run the optimization
    auto initial_state = opt::get_opt_initial_mps(tensors, opt_meta);
    auto opt_state     = opt::get_updated_state(tensors, initial_state, opt_meta);
    // Determine the quality of the optimized state.
    opt_state.set_relchange(opt_state.get_variance() / var_latest);
    opt_state.set_bond_limit(opt_meta.svd_cfg->rank_max.value());
    opt_state.set_trnc_limit(opt_meta.svd_cfg->truncation_limit.value());

    /* clang-format off */
    opt_meta.optExit = OptExit::SUCCESS;
    if(opt_state.get_grad_max()       > static_cast<RealScalar>(1.000)                            ) opt_meta.optExit |= OptExit::FAIL_GRADIENT;
    if(opt_state.get_eigs_rnorm()     > static_cast<RealScalar>(settings::solvers::eig::abstol_max)) opt_meta.optExit |= OptExit::FAIL_RESIDUAL;
    if(opt_state.get_eigs_nev()       == 0 and
       opt_meta.optSolver             == OptSolver::EIGS                                          ) opt_meta.optExit |= OptExit::FAIL_RESIDUAL; // No convergence
    if(opt_state.get_overlap()        < static_cast<RealScalar>(0.010)                            ) opt_meta.optExit |= OptExit::FAIL_OVERLAP;
    if(opt_state.get_relchange()      > static_cast<RealScalar>(1.001)                            ) opt_meta.optExit |= OptExit::FAIL_WORSENED;
    else if(opt_state.get_relchange() > static_cast<RealScalar>(0.999)                            ) opt_meta.optExit |= OptExit::FAIL_NOCHANGE;
    /* clang-format on */
    opt_state.set_optexit(opt_meta.optExit);

    tools::log->trace("Optimization [{}|{}]: {}. Variance change {:8.2e} --> {:8.2e} ({:.3f} %)", enum2sv(opt_meta.optAlgo), enum2sv(opt_meta.optSolver),
                      flag2str(opt_meta.optExit), var_latest, opt_state.get_variance(), opt_state.get_relchange() * 100);
    if(opt_state.get_relchange() > 1000) {
        tools::log->warn("Variance increase by x {:.2e} | variance new {:.3e} | variance old {:.3e}", opt_state.get_relchange(), opt_state.get_variance(),
                         var_latest);
    }

    if(tools::log->level() <= spdlog::level::debug) {
        tools::log->debug("Optimization result: {:<24} | E {:<20.16f}| σ²H {:<8.2e} | rnorm {:8.2e} | overlap {:.16f} | "
                          "sites {} | {:20} | {} | time {:.2e} s",
                          opt_state.get_name(), opt_state.get_energy(), opt_state.get_variance(), opt_state.get_eigs_rnorm(), opt_state.get_overlap(),
                          opt_state.get_sites(), fmt::format("[{}][{}]", enum2sv(opt_state.get_optalgo()), enum2sv(opt_state.get_optsolver())),
                          flag2str(opt_state.get_optexit()), opt_state.get_time());
    }

    // Do the truncation with SVD
    auto logPolicy = LogPolicy::SILENT;
    if constexpr(settings::debug) logPolicy = LogPolicy::VERBOSE;
    tensors.merge_multisite_mps(opt_state.get_tensor(), MergeEvent::OPT, opt_meta.svd_cfg, logPolicy);
    tensors.rebuild_edges(); // This will only do work if edges were modified, which is the case in 1-site dmrg.

    // Update current energy density ε
    if(status.opt_ritz == OptRitz::TE)
        status.energy_dens = (static_cast<double>(tools::finite::measure::energy(tensors)) - status.energy_min) / (status.energy_max - status.energy_min);

    tools::log->trace("Updating variance record holder");
    auto ene_mrg = tools::finite::measure::energy(tensors);
    auto var_mrg = tools::finite::measure::energy_variance(tensors);

    if(var_mrg < 0) {
        tools::log->info("Variance is negative! {:.16e}", var_mrg);
        auto mpo_squared_compress_before_postmortem = settings::model::use_compressed_mpo_squared;
        // Disable mpo compression
        settings::model::use_compressed_mpo         = MpoCompress::NONE;
        settings::model::use_compressed_mpo_squared = MpoCompress::NONE;
        {
            using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
            auto       pos   = tensors.template get_position<Eigen::Index>();
            auto       NL    = tools::finite::measure::isometry_left(tensors.get_state(), pos - 1);
            auto       NR    = tools::finite::measure::isometry_right(tensors.get_state(), pos + 1);
            auto       NLm   = tenx::MatrixMap(NL);
            auto       NRm   = tenx::MatrixMap(NR);
            auto       IL    = MatrixType::Identity(NLm.rows(), NLm.cols());
            auto       IR    = MatrixType::Identity(NRm.rows(), NRm.cols());
            RealScalar NLerr = (NLm - IL).norm() / IL.norm();
            RealScalar NRerr = (NRm - IR).norm() / IR.norm();
            tools::log->info("NL err: {:.16e}", NLerr);
            tools::log->info("NR err: {:.16e}", NRerr);
        }

        {
            auto                t_res      = tid::tic_token("<ψ|H²_global ψ>");
            StateFinite<Scalar> tmp_state1 = tensors.get_state();
            StateFinite<Scalar> tmp_state2 = tensors.get_state();
            auto                mpos1      = tensors.get_model().get_mpo_tensors(Scalar{0}, MposWithEdges::ON, MpoCompress::NONE);
            auto                mpos2      = tensors.get_model().get_mpo2_tensors(Scalar{0}, MposWithEdges::ON, MpoCompress::NONE);
            auto                svdcfg     = svd::config(8192, 1e-20);
            svdcfg.svd_lib                 = svd::lib::lapack;
            svdcfg.svd_rtn                 = svd::rtn::gesdd;
            tools::finite::ops::apply_mpos_general(tmp_state1, mpos1, svdcfg);
            tools::finite::ops::apply_mpos_general(tmp_state2, mpos2, svdcfg);
            RealScalar E1_global = std::real(tools::finite::ops::overlap<Scalar>(tmp_state1, tensors.get_state()));
            RealScalar E2_global = std::real(tools::finite::ops::overlap<Scalar>(tmp_state2, tensors.get_state()));
            tools::log->info("H²              <ψ| H²_global ψ>                                = {:.16e} | t = {:.4e}", E2_global, t_res->get_last_interval());

            RealScalar VarH = E2_global - E1_global * E1_global;
            tools::log->info("energy variance <H²_global> - <H_global>²                       = {:.16e} | t = {:.4e}", VarH, t_res->get_last_interval());
        }
        {
            auto                t_res        = tid::tic_token("<ψ (H_global-E_local) | (H_global-E_local) ψ>");
            StateFinite<Scalar> tmp_state    = tensors.get_state();
            auto                L            = tensors.template get_length<RealScalar>();
            auto                mpos_shifted = tensors.get_model().get_mpo_tensors(ene_mrg / L, MposWithEdges::ON, MpoCompress::NONE);
            auto                svdcfg       = svd::config(8192, 1e-20);
            tools::finite::ops::apply_mpos_general(tmp_state, mpos_shifted, svdcfg);
            RealScalar VarH_global = std::real(tools::finite::ops::overlap<Scalar>(tmp_state, tmp_state));
            tools::log->info("energy variance <ψ (H_global-E_local) | (H_global-E_local) ψ>   = {:.16e} | t = {:.4e}", VarH_global, t_res->get_last_interval());
        }
        {
            using ScalarL       = Scalar;
            using RealScalarL   = Eigen::NumTraits<ScalarL>::Real;
            using RealScalar128 = fp128;
            using Scalar128     = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, std::complex<RealScalar128>, RealScalar128>;

            auto tensorsL = tensors.template cast<ScalarL>();
            auto envi     = SetEnvInfo(ContractionPrecision::X2);
            auto mv1i     = SetH1MvInfo(ContractionPrecision::X2);
            auto mv2i     = SetH2MvInfo(ContractionPrecision::X2);
            tensorsL.get_model().build_mpo();
            tensorsL.get_model().build_mpo_squared();
            tensorsL.get_edges().eject_edges_all();
            tensorsL.rebuild_edges();
            tensorsL.clear_measurements();
            tensorsL.clear_cache();
            auto mps  = tensorsL.get_state().template get_multisite_mps<ScalarL>();
            auto mpo1 = tensorsL.get_model().template get_multisite_mpo<ScalarL>();
            auto mpo2 = tensorsL.get_model().template get_multisite_mpo_squared<ScalarL>();

            auto enve = tensorsL.get_edges().get_multisite_env_ene();
            auto envv = tensorsL.get_edges().get_multisite_env_var();
            auto H1t  = Eigen::Tensor<ScalarL, 3>(mps.dimensions());
            auto H2t  = Eigen::Tensor<ScalarL, 3>(mps.dimensions());
            tools::common::contraction::matrix_vector_product(H1t, mps, mpo1, enve.L, enve.R);
            tools::common::contraction::matrix_vector_product(H2t, mps, mpo2, envv.L, envv.R);

            auto          v           = tenx::VectorMap(mps);
            auto          H1v         = tenx::VectorMap(H1t);
            auto          H2v         = tenx::VectorMap(H2t);
            RealScalarL   v_H1v       = std::real(v.dot(H1v));
            RealScalarL   v_H2v       = std::real(v.dot(H2v));
            RealScalar128 v_H2v_fp128 = std::real(v.template cast<Scalar128>().dot(H2v.template cast<Scalar128>()));

            RealScalarL enveLnorm = enve.L.get_blkx2().norm();
            RealScalarL enveRnorm = enve.R.get_blkx2().norm();
            RealScalarL envvLnorm = envv.L.get_blkx2().norm();
            RealScalarL envvRnorm = envv.R.get_blkx2().norm();
            auto        envvLdims = envv.L.get_block().dimensions();
            auto        envvRdims = envv.R.get_block().dimensions();

            tools::log->info("X2    var_opt            = {:.16e}", opt_state.get_variance());
            tools::log->info("X2    var_mrg            = {:.16e}", var_mrg);
            tools::log->info("X2    active sites       = {} | first {} | last {}", tensorsL.active_sites.size(),
                             tensorsL.active_sites.empty() ? -1 : static_cast<long>(tensorsL.active_sites.front()),
                             tensorsL.active_sites.empty() ? -1 : static_cast<long>(tensorsL.active_sites.back()));
            tools::log->info("X2    envv dims          = L[{},{},{}] R[{},{},{}]", envvLdims[0], envvLdims[1], envvLdims[2], envvRdims[0], envvRdims[1],
                             envvRdims[2]);
            tools::log->info("X2    |H1v|              = {:.16e}", H1v.norm());
            tools::log->info("X2    |H2v|              = {:.16e}", H2v.norm());
            tools::log->info("X2    v_H1v              = {:.16e}", v_H1v);
            tools::log->info("X2    v_H2v              = {:.16e}", v_H2v);
            tools::log->info("X2    v_H2v    (fp128)   = {:.16e}", v_H2v_fp128);
            tools::log->info("X2    energy variance    = {:.16e}", v_H2v - v_H1v * v_H1v);
            tools::log->info("X2    |enveL|            = {:.16e}", enveLnorm);
            tools::log->info("X2    |enveR|            = {:.16e}", enveRnorm);
            tools::log->info("X2    |envvL|            = {:.16e}", envvLnorm);
            tools::log->info("X2    |envvR|            = {:.16e}", envvRnorm);
        }
        {
            using RealScalarL = fp64;
            using ScalarL     = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, std::complex<RealScalarL>, RealScalarL>;
            auto tensorsL     = tensors.template cast<ScalarL>();
            auto envi         = SetEnvInfo(ContractionBackend::EIGEN);
            auto mv1i         = SetH1MvInfo(ContractionBackend::EIGEN);
            auto mv2i         = SetH2MvInfo(ContractionBackend::EIGEN);
            tensorsL.get_model().build_mpo();
            tensorsL.get_model().build_mpo_squared();
            tensorsL.get_edges().eject_edges_all();
            tensorsL.rebuild_edges();
            tensorsL.clear_measurements();
            tensorsL.clear_cache();
            auto mps  = tensorsL.get_state().template get_multisite_mps<ScalarL>();
            auto mpo1 = tensorsL.get_model().template get_multisite_mpo<ScalarL>();
            auto mpo2 = tensorsL.get_model().template get_multisite_mpo_squared<ScalarL>();
            auto enve = tensorsL.get_edges().get_multisite_env_ene();
            auto envv = tensorsL.get_edges().get_multisite_env_var();
            auto H1t  = Eigen::Tensor<ScalarL, 3>(mps.dimensions());
            auto H2t  = Eigen::Tensor<ScalarL, 3>(mps.dimensions());
            tools::common::contraction::matrix_vector_product(H1t, mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
            tools::common::contraction::matrix_vector_product(H2t, mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());

            auto        v     = tenx::VectorMap(mps);
            auto        H1v   = tenx::VectorMap(H1t);
            auto        H2v   = tenx::VectorMap(H2t);
            RealScalarL v_H1v = std::real(v.dot(H1v));
            RealScalarL v_H2v = std::real(v.dot(H2v));

            RealScalarL enveLnorm = enve.L.get_blkx2().norm();
            RealScalarL enveRnorm = enve.R.get_blkx2().norm();
            RealScalarL envvLnorm = envv.L.get_blkx2().norm();
            RealScalarL envvRnorm = envv.R.get_blkx2().norm();
            auto        envvLdims = envv.L.get_block().dimensions();
            auto        envvRdims = envv.R.get_block().dimensions();

            tools::log->info("FP64  active sites       = {} | first {} | last {}", tensorsL.active_sites.size(),
                             tensorsL.active_sites.empty() ? -1 : static_cast<long>(tensorsL.active_sites.front()),
                             tensorsL.active_sites.empty() ? -1 : static_cast<long>(tensorsL.active_sites.back()));
            tools::log->info("FP64  envv dims          = L[{},{},{}] R[{},{},{}]", envvLdims[0], envvLdims[1], envvLdims[2], envvRdims[0], envvRdims[1],
                             envvRdims[2]);
            tools::log->info("FP64  |H1v|              = {:.16e}", H1v.norm());
            tools::log->info("FP64  |H2v|              = {:.16e}", H2v.norm());
            tools::log->info("FP64  v_H1v              = {:.16e}", v_H1v);
            tools::log->info("FP64  v_H2v              = {:.16e}", v_H2v);
            tools::log->info("FP64  energy variance    = {:.16e}", v_H2v - v_H1v * v_H1v);
            tools::log->info("FP64  |enveL|            = {:.16e}", enveLnorm);
            tools::log->info("FP64  |enveR|            = {:.16e}", enveRnorm);
            tools::log->info("FP64  |envvL|            = {:.16e}", envvLnorm);
            tools::log->info("FP64  |envvR|            = {:.16e}", envvRnorm);
        }

        {
            using RealScalarL = fp128;
            using ScalarL     = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, std::complex<RealScalarL>, RealScalarL>;
            auto tensorsL     = tensors.template cast<ScalarL>();
            auto envi         = SetEnvInfo(ContractionBackend::EIGEN);
            auto mv1i         = SetH1MvInfo(ContractionBackend::EIGEN);
            auto mv2i         = SetH2MvInfo(ContractionBackend::EIGEN);
            tensorsL.get_model().build_mpo();
            tensorsL.get_model().build_mpo_squared();
            tensorsL.get_edges().eject_edges_all();
            tensorsL.rebuild_edges();
            tensorsL.clear_measurements();
            tensorsL.clear_cache();
            auto mps  = tensorsL.get_state().template get_multisite_mps<ScalarL>();
            auto mpo1 = tensorsL.get_model().template get_multisite_mpo<ScalarL>();
            auto mpo2 = tensorsL.get_model().template get_multisite_mpo_squared<ScalarL>();
            auto enve = tensorsL.get_edges().get_multisite_env_ene();
            auto envv = tensorsL.get_edges().get_multisite_env_var();
            auto H1t  = Eigen::Tensor<ScalarL, 3>(mps.dimensions());
            auto H2t  = Eigen::Tensor<ScalarL, 3>(mps.dimensions());
            tools::common::contraction::matrix_vector_product(H1t, mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
            tools::common::contraction::matrix_vector_product(H2t, mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());

            auto        v     = tenx::VectorMap(mps);
            auto        H1v   = tenx::VectorMap(H1t);
            auto        H2v   = tenx::VectorMap(H2t);
            RealScalarL v_H1v = std::real(v.dot(H1v));
            RealScalarL v_H2v = std::real(v.dot(H2v));

            RealScalarL enveLnorm = enve.L.get_blkx2().norm();
            RealScalarL enveRnorm = enve.R.get_blkx2().norm();
            RealScalarL envvLnorm = envv.L.get_blkx2().norm();
            RealScalarL envvRnorm = envv.R.get_blkx2().norm();
            auto        envvLdims = envv.L.get_block().dimensions();
            auto        envvRdims = envv.R.get_block().dimensions();

            tools::log->info("FP128 active sites       = {} | first {} | last {}", tensorsL.active_sites.size(),
                             tensorsL.active_sites.empty() ? -1 : static_cast<long>(tensorsL.active_sites.front()),
                             tensorsL.active_sites.empty() ? -1 : static_cast<long>(tensorsL.active_sites.back()));
            tools::log->info("FP128 envv dims          = L[{},{},{}] R[{},{},{}]", envvLdims[0], envvLdims[1], envvLdims[2], envvRdims[0], envvRdims[1],
                             envvRdims[2]);
            tools::log->info("FP128 |H1v|              = {:.16e}", H1v.norm());
            tools::log->info("FP128 |H2v|              = {:.16e}", H2v.norm());
            tools::log->info("FP128 v_H1v              = {:.16e}", v_H1v);
            tools::log->info("FP128 v_H2v              = {:.16e}", v_H2v);
            tools::log->info("FP128 energy variance    = {:.16e}", v_H2v - v_H1v * v_H1v);
            tools::log->info("FP128 |enveL|            = {:.16e}", enveLnorm);
            tools::log->info("FP128 |enveR|            = {:.16e}", enveRnorm);
            tools::log->info("FP128 |envvL|            = {:.16e}", envvLnorm);
            tools::log->info("FP128 |envvR|            = {:.16e}", envvRnorm);
        }

        std::vector<std::string> msg1;
        std::vector<std::string> msg2;
        {
            // tensors.get_state().clear_cache();
            // tensors.get_state().clear_measurements();
            // tensors.get_edges().eject_edges_all();
            // TODO: Temporary environment rebuild diagnostic. Remove after deciding whether stale or numerically degraded environments cause negative variance.
            using EnvAccReal   = fp128;
            using EnvAccScalar = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, std::complex<EnvAccReal>, EnvAccReal>;
            struct EnvSnapshot {
                std::string                    label;
                std::string                    side;
                size_t                         position  = 0;
                size_t                         unique_id = 0;
                bool                           has_block = false;
                std::array<Eigen::Index, 3>    dims      = {0, 0, 0};
                Eigen::Tensor<EnvAccScalar, 3> block;
                EnvAccReal                     norm = 0;
            };
            auto env_to_acc = [](const auto &z) -> EnvAccScalar {
                using Z = std::decay_t<decltype(z)>;
                if constexpr(Eigen::NumTraits<Z>::IsComplex == 1)
                    return EnvAccScalar{static_cast<EnvAccReal>(std::real(z)), static_cast<EnvAccReal>(std::imag(z))};
                else
                    return static_cast<EnvAccReal>(z);
            };
            auto env_conj_acc = [](const EnvAccScalar &z) -> EnvAccScalar {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return std::conj(z);
                else
                    return z;
            };
            auto env_real_acc = [](const EnvAccScalar &z) -> EnvAccReal {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return std::real(z);
                else
                    return z;
            };
            auto env_x2_to_acc_tensor = [&](const auto &t) {
                auto out = Eigen::Tensor<EnvAccScalar, 3>(t.dimensions());
                for(Eigen::Index i = 0; i < t.size(); ++i) out.data()[i] = env_to_acc(t.hi_data()[i]) + env_to_acc(t.lo_data()[i]);
                return out;
            };
            auto env_norm_acc = [&](const Eigen::Tensor<EnvAccScalar, 3> &t) -> EnvAccReal {
                EnvAccReal sum = EnvAccReal{0};
                for(Eigen::Index i = 0; i < t.size(); ++i) sum += env_real_acc(env_conj_acc(t.data()[i]) * t.data()[i]);
                return std::sqrt(sum);
            };
            auto env_diff_norm_acc = [&](const Eigen::Tensor<EnvAccScalar, 3> &a, const Eigen::Tensor<EnvAccScalar, 3> &b) -> EnvAccReal {
                EnvAccReal sum = EnvAccReal{0};
                for(Eigen::Index i = 0; i < a.size(); ++i) {
                    const auto d  = a.data()[i] - b.data()[i];
                    sum          += env_real_acc(env_conj_acc(d) * d);
                }
                return std::sqrt(sum);
            };
            auto env_relevant_to_active_sites = [&](std::string_view side, size_t pos) {
                if(tensors.active_sites.empty()) return true;
                if(side == "L") return pos <= tensors.active_sites.front();
                if(side == "R") return pos >= tensors.active_sites.back();
                return true;
            };
            auto env_x2_component_norms = [&](const auto &t) {
                EnvAccReal hi2   = EnvAccReal{0};
                EnvAccReal lo2   = EnvAccReal{0};
                EnvAccReal full2 = EnvAccReal{0};
                for(Eigen::Index i = 0; i < t.size(); ++i) {
                    const auto hi  = env_to_acc(t.hi_data()[i]);
                    const auto lo  = env_to_acc(t.lo_data()[i]);
                    hi2           += env_real_acc(env_conj_acc(hi) * hi);
                    lo2           += env_real_acc(env_conj_acc(lo) * lo);
                    full2         += env_real_acc(env_conj_acc(hi + lo) * (hi + lo));
                }
                return std::array<EnvAccReal, 3>{std::sqrt(hi2), std::sqrt(lo2), std::sqrt(full2)};
            };
            auto log_env_x2_components = [&](const auto &envs, std::string_view label, std::string_view side) {
                for(const auto &env : envs) {
                    const auto pos = env->get_position();
                    if(not env_relevant_to_active_sites(side, pos)) continue;
                    if(not env->has_block()) continue;
                    const auto norms = env_x2_component_norms(env->get_blkx2());
                    tools::log->info("env x2 components {}{}[{:2}] dims [{},{},{}] |hi| {:.16e} |lo| {:.16e} |hi+lo| {:.16e}", label, side, pos,
                                     env->get_blkx2().dimension(0), env->get_blkx2().dimension(1), env->get_blkx2().dimension(2), norms[0], norms[1], norms[2]);
                }
            };
            auto env_norm4_acc = [&](const auto &t) -> EnvAccReal {
                EnvAccReal sum = EnvAccReal{0};
                for(Eigen::Index i = 0; i < t.size(); ++i) {
                    const auto z  = env_to_acc(t.data()[i]);
                    sum          += env_real_acc(env_conj_acc(z) * z);
                }
                return std::sqrt(sum);
            };
            auto env_diff_norm4_acc = [&](const auto &a, const auto &b) -> EnvAccReal {
                EnvAccReal sum = EnvAccReal{0};
                for(Eigen::Index i = 0; i < a.size(); ++i) {
                    const auto d  = env_to_acc(a.data()[i]) - env_to_acc(b.data()[i]);
                    sum          += env_real_acc(env_conj_acc(d) * d);
                }
                return std::sqrt(sum);
            };
            auto log_mpo2_site_diffs = [&](std::string_view label, const auto &model_ref) {
                for(size_t pos = 0; pos < tensors.template get_length<size_t>(); ++pos) {
                    const auto &mpo2_live = tensors.get_model().get_mpo(pos).MPO2();
                    const auto &mpo2_ref  = model_ref.get_mpo(pos).MPO2();
                    const auto  live_norm = env_norm4_acc(mpo2_live);
                    const auto  ref_norm  = env_norm4_acc(mpo2_ref);
                    const auto  live_dims = mpo2_live.dimensions();
                    const auto  ref_dims  = mpo2_ref.dimensions();
                    if(live_dims != ref_dims) {
                        tools::log->info("mpo2 compressed diff {} site[{:2}] dims live [{},{},{},{}] ref [{},{},{},{}]", label, pos, live_dims[0], live_dims[1],
                                         live_dims[2], live_dims[3], ref_dims[0], ref_dims[1], ref_dims[2], ref_dims[3]);
                        continue;
                    }
                    const auto diff = env_diff_norm4_acc(mpo2_live, mpo2_ref);
                    const auto den  = std::max(live_norm, ref_norm);
                    const auto rel  = den > EnvAccReal{0} ? diff / den : diff;
                    tools::log->info("mpo2 compressed diff {} site[{:2}] dims [{},{},{},{}] | live {:.16e} ref {:.16e} | diff {:.16e} rel {:.16e}", label, pos,
                                     live_dims[0], live_dims[1], live_dims[2], live_dims[3], live_norm, ref_norm, diff, rel);
                }
            };
            std::vector<EnvSnapshot> env_snapshots;
            auto                     collect_env_snapshots = [&](const auto &envs, std::string_view label, std::string_view side) {
                for(const auto &env : envs) {
                    const auto pos = env->get_position();
                    if(not env_relevant_to_active_sites(side, pos)) continue;
                    EnvSnapshot snap;
                    snap.label     = std::string(label);
                    snap.side      = std::string(side);
                    snap.position  = pos;
                    snap.has_block = env->has_block();
                    if(snap.has_block) {
                        snap.unique_id = env->get_unique_id();
                        snap.dims      = env->get_blkx2().dimensions();
                        snap.block     = env_x2_to_acc_tensor(env->get_blkx2());
                        snap.norm      = env_norm_acc(snap.block);
                    }
                    env_snapshots.emplace_back(std::move(snap));
                }
            };
            auto log_env_rebuild_diffs = [&](const auto &envs, std::string_view label, std::string_view side) {
                for(const auto &snap : env_snapshots) {
                    if(snap.label != label or snap.side != side) continue;
                    auto env_it = std::find_if(envs.begin(), envs.end(), [&](const auto &env) { return env->get_position() == snap.position; });
                    if(env_it == envs.end()) {
                        tools::log->info("env rebuild diff {}{}[{:2}] missing after rebuild", label, side, snap.position);
                        continue;
                    }
                    const auto &env       = **env_it;
                    const auto  has_after = env.has_block();
                    if(not snap.has_block or not has_after) {
                        tools::log->info("env rebuild diff {}{}[{:2}] block before {} after {}", label, side, snap.position, snap.has_block, has_after);
                        continue;
                    }
                    const auto id_after   = env.get_unique_id();
                    const auto blk_after  = env_x2_to_acc_tensor(env.get_blkx2());
                    const auto norm_after = env_norm_acc(blk_after);
                    if(snap.dims != env.get_blkx2().dimensions()) {
                        tools::log->info("env rebuild diff {}{}[{:2}] id {} -> {} dims [{},{},{}] -> [{},{},{}]", label, side, snap.position, snap.unique_id,
                                         id_after, snap.dims[0], snap.dims[1], snap.dims[2], env.get_blkx2().dimension(0), env.get_blkx2().dimension(1),
                                         env.get_blkx2().dimension(2));
                        continue;
                    }
                    const auto diff = env_diff_norm_acc(snap.block, blk_after);
                    const auto den  = std::max(snap.norm, norm_after);
                    const auto rel  = den > EnvAccReal{0} ? diff / den : diff;
                    tools::log->info("env rebuild diff {}{}[{:2}] id {} -> {} | norm {:.16e} -> {:.16e} | diff {:.16e} rel {:.16e}", label, side, snap.position,
                                     snap.unique_id, id_after, snap.norm, norm_after, diff, rel);
                }
            };
            auto log_env_ref_diffs = [&](const auto &envs_ref, std::string_view ref_label, std::string_view label, std::string_view side) {
                for(const auto &snap : env_snapshots) {
                    if(snap.label != label or snap.side != side) continue;
                    auto env_it = std::find_if(envs_ref.begin(), envs_ref.end(), [&](const auto &env) { return env->get_position() == snap.position; });
                    if(env_it == envs_ref.end()) {
                        tools::log->info("env {} diff {}{}[{:2}] missing in ref rebuild", ref_label, label, side, snap.position);
                        continue;
                    }
                    const auto &env_ref       = **env_it;
                    const auto  has_ref_block = env_ref.has_block();
                    if(not snap.has_block or not has_ref_block) {
                        tools::log->info("env {} diff {}{}[{:2}] block current {} ref {}", ref_label, label, side, snap.position, snap.has_block,
                                         has_ref_block);
                        continue;
                    }
                    const auto blk_ref  = env_x2_to_acc_tensor(env_ref.get_blkx2());
                    const auto norm_ref = env_norm_acc(blk_ref);
                    if(snap.dims != env_ref.get_blkx2().dimensions()) {
                        tools::log->info("env {} diff {}{}[{:2}] dims current [{},{},{}] ref [{},{},{}]", ref_label, label, side, snap.position, snap.dims[0],
                                         snap.dims[1], snap.dims[2], env_ref.get_blkx2().dimension(0), env_ref.get_blkx2().dimension(1),
                                         env_ref.get_blkx2().dimension(2));
                        continue;
                    }
                    const auto diff = env_diff_norm_acc(snap.block, blk_ref);
                    const auto den  = std::max(snap.norm, norm_ref);
                    const auto rel  = den > EnvAccReal{0} ? diff / den : diff;
                    tools::log->info("env {} diff {}{}[{:2}] | norm current {:.16e} ref {:.16e} | diff {:.16e} rel {:.16e}", ref_label, label, side,
                                     snap.position, snap.norm, norm_ref, diff, rel);
                }
            };
            auto env_dot_acc = [&](const Eigen::Tensor<EnvAccScalar, 3> &x, const Eigen::Tensor<EnvAccScalar, 3> &y) -> EnvAccReal {
                EnvAccScalar sum = EnvAccScalar{0};
                for(Eigen::Index i = 0; i < x.size(); ++i) sum += env_conj_acc(x.data()[i]) * y.data()[i];
                return env_real_acc(sum);
            };

            for(const auto &env : tensors.get_edges().eneL) {
                msg1.emplace_back(fmt::format("eneL[{:2}]: {} norm {:.8e}", env->get_position(), env->get_unique_id(), fp(tenx::norm(env->get_block()))));
            }
            for(const auto &env : tensors.get_edges().eneR) {
                msg1.emplace_back(fmt::format("eneR[{:2}]: {} norm {:.8e}", env->get_position(), env->get_unique_id(), fp(tenx::norm(env->get_block()))));
            }

            tools::log->info("env rebuild diff active sites {}", tensors.active_sites);
            collect_env_snapshots(tensors.get_edges().eneL, "ene", "L");
            collect_env_snapshots(tensors.get_edges().eneR, "ene", "R");
            collect_env_snapshots(tensors.get_edges().varL, "var", "L");
            collect_env_snapshots(tensors.get_edges().varR, "var", "R");
            tensors.get_edges().eject_edges_all();
            {
                const auto envinfo_live = tools::common::contraction::internal::get_info_env();
                tools::log->info("live env rebuild policy backend {} precision {}", enum2sv(envinfo_live.backend), enum2sv(envinfo_live.precision));
            }
            tensors.rebuild_edges();
            log_env_rebuild_diffs(tensors.get_edges().eneL, "ene", "L");
            log_env_rebuild_diffs(tensors.get_edges().eneR, "ene", "R");
            log_env_rebuild_diffs(tensors.get_edges().varL, "var", "L");
            log_env_rebuild_diffs(tensors.get_edges().varR, "var", "R");
            log_env_x2_components(tensors.get_edges().varL, "live-var", "L");
            log_env_x2_components(tensors.get_edges().varR, "live-var", "R");
            {
                // TODO: Temporary FP128 environment comparison. Remove after locating whether MPO2 compression or environment construction causes negative
                // variance.
                auto tensors_fp128 = tensors.template cast<EnvAccScalar>();
                auto envinfo_fp128 = SetEnvInfo(ContractionBackend::EIGEN);
                tensors_fp128.get_model().build_mpo();
                tensors_fp128.get_model().build_mpo_squared();
                auto mpo_squared_compress_postmortem = settings::model::use_compressed_mpo_squared;
                if(mpo_squared_compress_before_postmortem == MpoCompress::AUTO) mpo_squared_compress_before_postmortem = MpoCompress::DPL;
                settings::model::use_compressed_mpo_squared = mpo_squared_compress_before_postmortem;
                // TODO: Temporary diagnostic override. The surrounding post-mortem disables MPO2 compression; force the original mode here so the FP128
                // comparison uses the same compressed operator family as the live variance path.
                tensors_fp128.compress_mpo_squared();
                settings::model::use_compressed_mpo_squared = mpo_squared_compress_postmortem;
                tools::log->info("FP128 forced MPO2 compression mode {} | has_compressed {}", enum2sv(mpo_squared_compress_before_postmortem),
                                 tensors_fp128.get_model().has_compressed_mpo_squared());
                log_mpo2_site_diffs("live/fp128", tensors_fp128.get_model());
                auto tensors_fp128_down = tensors;
                for(size_t pos = 0; pos < tensors.template get_length<size_t>(); ++pos) {
                    const auto mpo2_down = Eigen::Tensor<Scalar, 4>(tensors_fp128.get_model().get_mpo(pos).template MPO2_as<Scalar>());
                    tensors_fp128_down.get_model().get_mpo(pos).set_mpo_squared(mpo2_down);
                }
                tensors_fp128_down.clear_cache();
                tensors_fp128_down.clear_measurements();
                tools::log->info("FP128-compressed MPO2 downcast to live scalar | has_compressed {}",
                                 tensors_fp128_down.get_model().has_compressed_mpo_squared());
                log_mpo2_site_diffs("live/fp128down", tensors_fp128_down.get_model());
                tensors_fp128_down.get_edges().eject_edges_all();
                {
                    auto       envinfo_down         = SetEnvInfo(ContractionBackend::AUTO, ContractionPrecision::X2);
                    const auto envinfo_down_rebuild = tools::common::contraction::internal::get_info_env();
                    tools::log->info("FP128-downcast env rebuild policy backend {} precision {}", enum2sv(envinfo_down_rebuild.backend),
                                     enum2sv(envinfo_down_rebuild.precision));
                    tensors_fp128_down.rebuild_edges();
                }
                tensors_fp128_down.clear_cache();
                tensors_fp128_down.clear_measurements();
                tensors_fp128.get_edges().eject_edges_all();
                {
                    const auto envinfo_fp128_rebuild = tools::common::contraction::internal::get_info_env();
                    tools::log->info("FP128 env rebuild policy backend {} precision {}", enum2sv(envinfo_fp128_rebuild.backend),
                                     enum2sv(envinfo_fp128_rebuild.precision));
                }
                tensors_fp128.rebuild_edges();
                tensors_fp128.clear_cache();
                tensors_fp128.clear_measurements();
                const auto &mps_fp128               = tensors_fp128.get_state().template get_multisite_mps<EnvAccScalar>();
                const auto &mpo2_fp128              = tensors_fp128.get_model().template get_multisite_mpo_squared<EnvAccScalar>();
                const auto &envv_fp128              = tensors_fp128.get_edges().get_multisite_env_var();
                const auto &mps_fp128_down          = tensors_fp128_down.get_state().template get_multisite_mps<Scalar>();
                const auto &mpo2_fp128_down         = tensors_fp128_down.get_model().template get_multisite_mpo_squared<Scalar>();
                const auto &envv_fp128_down         = tensors_fp128_down.get_edges().get_multisite_env_var();
                const auto &mps_live                = tensors.get_state().template get_multisite_mps<Scalar>();
                const auto &mpo2_live               = tensors.get_model().template get_multisite_mpo_squared<Scalar>();
                const auto &envv_live               = tensors.get_edges().get_multisite_env_var();
                auto        mps_live_acc            = Eigen::Tensor<EnvAccScalar, 3>(mps_live.template cast<EnvAccScalar>());
                auto        mpo2_live_acc           = Eigen::Tensor<EnvAccScalar, 4>(mpo2_live.template cast<EnvAccScalar>());
                auto        envvL_live_acc          = env_x2_to_acc_tensor(envv_live.L.get_blkx2());
                auto        envvR_live_acc          = env_x2_to_acc_tensor(envv_live.R.get_blkx2());
                auto        mps_fp128_down_acc      = Eigen::Tensor<EnvAccScalar, 3>(mps_fp128_down.template cast<EnvAccScalar>());
                auto        mpo2_fp128_down_acc     = Eigen::Tensor<EnvAccScalar, 4>(mpo2_fp128_down.template cast<EnvAccScalar>());
                auto        envvL_fp128_down_acc    = env_x2_to_acc_tensor(envv_fp128_down.L.get_blkx2());
                auto        envvR_fp128_down_acc    = env_x2_to_acc_tensor(envv_fp128_down.R.get_blkx2());
                auto        envvL_fp128_scalar_down = Eigen::Tensor<Scalar, 3>(envv_fp128.L.template get_block_as<Scalar>());
                auto        envvR_fp128_scalar_down = Eigen::Tensor<Scalar, 3>(envv_fp128.R.template get_block_as<Scalar>());
                auto        envvL_fp128_sdown_acc   = Eigen::Tensor<EnvAccScalar, 3>(envvL_fp128_scalar_down.template cast<EnvAccScalar>());
                auto        envvR_fp128_sdown_acc   = Eigen::Tensor<EnvAccScalar, 3>(envvR_fp128_scalar_down.template cast<EnvAccScalar>());
                auto        split_acc_to_x2         = [](const Eigen::Tensor<EnvAccScalar, 3> &src) {
                    auto dst = x2::Tensor<Scalar, 3>(src.dimension(0), src.dimension(1), src.dimension(2));
                    for(Eigen::Index i = 0; i < src.size(); ++i) {
                        if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1) {
                            using LiveReal   = typename Eigen::NumTraits<Scalar>::Real;
                            const auto z     = src.data()[i];
                            const auto rhi   = static_cast<LiveReal>(std::real(z));
                            const auto ihi   = static_cast<LiveReal>(std::imag(z));
                            const auto rlo   = static_cast<LiveReal>(std::real(z) - static_cast<EnvAccReal>(rhi));
                            const auto ilo   = static_cast<LiveReal>(std::imag(z) - static_cast<EnvAccReal>(ihi));
                            dst.hi_data()[i] = Scalar{rhi, ihi};
                            dst.lo_data()[i] = Scalar{rlo, ilo};
                        } else {
                            const auto hi    = static_cast<Scalar>(src.data()[i]);
                            const auto lo    = static_cast<Scalar>(src.data()[i] - static_cast<EnvAccScalar>(hi));
                            dst.hi_data()[i] = hi;
                            dst.lo_data()[i] = lo;
                        }
                    }
                    return dst;
                };
                auto envvL_fp128_split_x2    = split_acc_to_x2(env_x2_to_acc_tensor(envv_fp128.L.get_blkx2()));
                auto envvR_fp128_split_x2    = split_acc_to_x2(env_x2_to_acc_tensor(envv_fp128.R.get_blkx2()));
                auto H2t_fp128               = Eigen::Tensor<EnvAccScalar, 3>(mps_fp128.dimensions());
                auto h2info_fp128_compressed = SetH2MvInfo(ContractionBackend::EIGEN, mpo2_fp128.dimensions());
                tools::common::contraction::matrix_vector_product(H2t_fp128, mps_fp128, mpo2_fp128, envv_fp128.L.get_blkx2(), envv_fp128.R.get_blkx2());
                tools::log->info("FP128 compressed |mpo2| dims [{},{},{},{}]", mpo2_fp128.dimension(0), mpo2_fp128.dimension(1), mpo2_fp128.dimension(2),
                                 mpo2_fp128.dimension(3));
                tools::log->info("FP128 compressed v_H2v = {:.16e}", env_dot_acc(mps_fp128, H2t_fp128));
                // TODO: Temporary mixed-input diagnostic. Remove after identifying whether live FP64-compressed MPO2 or live variance environments dominate
                // the H2 expectation error.
                auto log_mixed_h2 = [&](std::string_view label, const auto &mps_in, const auto &mpo2_in, const auto &envL_in, const auto &envR_in) {
                    auto H2t_mix = Eigen::Tensor<EnvAccScalar, 3>(mps_in.dimensions());
                    auto h2info  = SetH2MvInfo(ContractionBackend::EIGEN, mpo2_in.dimensions());
                    tools::common::contraction::matrix_vector_product(H2t_mix, mps_in, mpo2_in, envL_in, envR_in);
                    auto envLdim = envL_in.dimensions();
                    auto envRdim = envR_in.dimensions();
                    tools::log->info("mixed H2 {:>18} | mpo2 [{},{},{},{}] envL [{},{},{}] envR [{},{},{}] v_H2v {:.16e}", label, mpo2_in.dimension(0),
                                     mpo2_in.dimension(1), mpo2_in.dimension(2), mpo2_in.dimension(3), envLdim[0], envLdim[1], envLdim[2], envRdim[0],
                                     envRdim[1], envRdim[2], env_dot_acc(mps_in, H2t_mix));
                };
                log_mixed_h2("live_mpo live_env", mps_live_acc, mpo2_live_acc, envvL_live_acc, envvR_live_acc);
                log_mixed_h2("fp128_mpo fp128_env", mps_fp128, mpo2_fp128, envv_fp128.L.get_blkx2(), envv_fp128.R.get_blkx2());
                log_mixed_h2("fp128_mpo live_env", mps_fp128, mpo2_fp128, envvL_live_acc, envvR_live_acc);
                log_mixed_h2("live_mpo fp128_env", mps_live_acc, mpo2_live_acc, envv_fp128.L.get_blkx2(), envv_fp128.R.get_blkx2());
                log_mixed_h2("live_mpo fp128_sdown", mps_live_acc, mpo2_live_acc, envvL_fp128_sdown_acc, envvR_fp128_sdown_acc);
                {
                    auto h2info_split = SetH2MvInfo(ContractionPrecision::X2, mpo2_live.dimensions());
                    auto H2t_fp128_split =
                        tools::common::contraction::matrix_vector_product_x2(mps_live, mpo2_live, envvL_fp128_split_x2, envvR_fp128_split_x2);
                    auto H2t_split_acc = env_x2_to_acc_tensor(H2t_fp128_split);
                    tools::log->info("mixed H2 {:>18} | mpo2 [{},{},{},{}] envL [{},{},{}] envR [{},{},{}] v_H2v {:.16e}", "live_mpo fp128_x2split",
                                     mpo2_live.dimension(0), mpo2_live.dimension(1), mpo2_live.dimension(2), mpo2_live.dimension(3),
                                     envvL_fp128_split_x2.dimension(0), envvL_fp128_split_x2.dimension(1), envvL_fp128_split_x2.dimension(2),
                                     envvR_fp128_split_x2.dimension(0), envvR_fp128_split_x2.dimension(1), envvR_fp128_split_x2.dimension(2),
                                     env_dot_acc(mps_live_acc, H2t_split_acc));
                }
                log_mixed_h2("down_mpo down_env", mps_fp128_down_acc, mpo2_fp128_down_acc, envvL_fp128_down_acc, envvR_fp128_down_acc);
                log_mixed_h2("fp128_mpo down_env", mps_fp128, mpo2_fp128, envvL_fp128_down_acc, envvR_fp128_down_acc);
                log_mixed_h2("down_mpo fp128_env", mps_fp128_down_acc, mpo2_fp128_down_acc, envv_fp128.L.get_blkx2(), envv_fp128.R.get_blkx2());
                log_env_ref_diffs(tensors_fp128_down.get_edges().varL, "fp128down", "var", "L");
                log_env_ref_diffs(tensors_fp128_down.get_edges().varR, "fp128down", "var", "R");
                log_env_x2_components(tensors_fp128_down.get_edges().varL, "down-var", "L");
                log_env_x2_components(tensors_fp128_down.get_edges().varR, "down-var", "R");
                log_env_ref_diffs(tensors_fp128.get_edges().eneL, "fp128", "ene", "L");
                log_env_ref_diffs(tensors_fp128.get_edges().eneR, "fp128", "ene", "R");
                log_env_ref_diffs(tensors_fp128.get_edges().varL, "fp128", "var", "L");
                log_env_ref_diffs(tensors_fp128.get_edges().varR, "fp128", "var", "R");
                log_env_x2_components(tensors_fp128.get_edges().varL, "fp128-var", "L");
                log_env_x2_components(tensors_fp128.get_edges().varR, "fp128-var", "R");
            }
            tensors.clear_cache();
            tensors.clear_measurements();

            const auto &mps  = tensors.get_state().template get_multisite_mps<Scalar>();
            const auto &mpo1 = tensors.get_model().template get_multisite_mpo<Scalar>();
            const auto &mpo2 = tensors.get_model().template get_multisite_mpo_squared<Scalar>();
            const auto &enve = tensors.get_edges().get_multisite_env_ene();
            const auto &envv = tensors.get_edges().get_multisite_env_var();
            tools::log->info("live MPO2 compression mode {} | has_compressed {} | mpo2 dims [{},{},{},{}]", enum2sv(mpo_squared_compress_before_postmortem),
                             tensors.get_model().has_compressed_mpo_squared(), mpo2.dimension(0), mpo2.dimension(1), mpo2.dimension(2), mpo2.dimension(3));

            auto H1t         = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H1tx2       = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H1tQ        = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H2t         = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H2tx2       = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H2tQ        = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H1t_x2_full = x2::Tensor<Scalar, 3>();
            auto H2t_x2_full = x2::Tensor<Scalar, 3>();
            {
                auto h1info = SetH1MvInfo(ContractionBackend::TBLIS, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionBackend::TBLIS, mpo2.dimensions());
                tools::log->info("contracting with tblis");
                tools::common::contraction::matrix_vector_product(H1t, mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
                tools::common::contraction::matrix_vector_product(H2t, mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());
            }
            {
                auto h1info = SetH1MvInfo(ContractionPrecision::X2, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionPrecision::X2, mpo2.dimensions());
                tools::log->info("contracting with x2");
                H1t_x2_full = tools::common::contraction::matrix_vector_product_x2(mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
                H2t_x2_full = tools::common::contraction::matrix_vector_product_x2(mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());
                H1tx2       = H1t_x2_full.to_EigenTensor();
                H2tx2       = H2t_x2_full.to_EigenTensor();
            }
            {
                auto h2info = SetH2MvInfo(ContractionBackend::EIGEN, mpo2.dimensions());
                tools::log->info("contracting with fp128");
                using ScalarQ = std::conditional_t<std::is_floating_point_v<Scalar>, fp128, cx128>;
                Eigen::Tensor<ScalarQ, 3> H1t_q(mps.dimensions());
                Eigen::Tensor<ScalarQ, 3> H2t_q(mps.dimensions());
                Eigen::Tensor<ScalarQ, 3> mps_q   = mps.template cast<ScalarQ>();
                Eigen::Tensor<ScalarQ, 4> mpo1_q  = mpo1.template cast<ScalarQ>();
                Eigen::Tensor<ScalarQ, 4> mpo2_q  = mpo2.template cast<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> enveL_q = enve.L.template get_block_as<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> enveR_q = enve.R.template get_block_as<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> envvL_q = envv.L.template get_block_as<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> envvR_q = envv.R.template get_block_as<ScalarQ>();
                tools::common::contraction::matrix_vector_product(H1t_q, mps_q, mpo1_q, enveL_q, enveR_q);
                tools::common::contraction::matrix_vector_product(H2t_q, mps_q, mpo2_q, envvL_q, envvR_q);
                H1tQ = H1t_q.template cast<Scalar>();
                H2tQ = H2t_q.template cast<Scalar>();
            }
            using VectorType   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            auto       v       = tenx::VectorMap(mps);
            auto       H1v     = tenx::VectorMap(H1t);
            auto       H1vx2   = tenx::VectorMap(H1tx2);
            auto       H1vQ    = tenx::VectorMap(H1tQ);
            auto       H2v     = tenx::VectorMap(H2t);
            auto       H2vx2   = tenx::VectorMap(H2tx2);
            auto       H2vQ    = tenx::VectorMap(H2tQ);
            RealScalar vH1_H1v = std::real(H1v.dot(H1v));
            RealScalar v_H1v   = std::real(v.dot(H1v));
            RealScalar v_H1vx2 = std::real(v.dot(H1vx2));
            RealScalar v_H1vQ  = std::real(v.dot(H1vQ));
            RealScalar v_H2v   = std::real(v.dot(H2v));
            RealScalar v_H2vx2 = std::real(v.dot(H2vx2));
            RealScalar v_H2vQ  = std::real(v.dot(H2vQ));

            using AccReal   = fp128;
            using AccScalar = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, std::complex<AccReal>, AccReal>;
            auto to_acc     = [](const Scalar &z) -> AccScalar {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return AccScalar{static_cast<AccReal>(std::real(z)), static_cast<AccReal>(std::imag(z))};
                else
                    return static_cast<AccReal>(z);
            };
            auto conj_acc = [](const AccScalar &z) -> AccScalar {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return std::conj(z);
                else
                    return z;
            };
            auto real_acc = [](const AccScalar &z) -> AccReal {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return std::real(z);
                else
                    return z;
            };
            auto dot_x2_real = [&](const x2::Tensor<Scalar, 3> &y) -> AccReal {
                AccScalar sum = AccScalar{0};
                for(Eigen::Index i = 0; i < mps.size(); ++i) {
                    const auto xacc  = to_acc(mps.data()[i]);
                    const auto yacc  = to_acc(y.hi_data()[i]) + to_acc(y.lo_data()[i]);
                    sum             += conj_acc(xacc) * yacc;
                }
                return real_acc(sum);
            };
            auto norm2_x2 = [&](const x2::Tensor<Scalar, 3> &y) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < y.size(); ++i) {
                    const auto yacc  = to_acc(y.hi_data()[i]) + to_acc(y.lo_data()[i]);
                    sum             += real_acc(conj_acc(yacc) * yacc);
                }
                return sum;
            };
            auto residual_norm2_x2 = [&](const x2::Tensor<Scalar, 3> &y, AccReal E) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < mps.size(); ++i) {
                    const auto xacc  = to_acc(mps.data()[i]);
                    const auto yacc  = to_acc(y.hi_data()[i]) + to_acc(y.lo_data()[i]);
                    const auto racc  = yacc - E * xacc;
                    sum             += real_acc(conj_acc(racc) * racc);
                }
                return sum;
            };

            AccReal v_H1v_x2_full     = dot_x2_real(H1t_x2_full);
            AccReal v_H2v_x2_full     = dot_x2_real(H2t_x2_full);
            AccReal vH1_H1v_x2_full   = norm2_x2(H1t_x2_full);
            AccReal proj_res2_x2_full = residual_norm2_x2(H1t_x2_full, v_H1v_x2_full);
            AccReal leak2_raw_x2_full = v_H2v_x2_full - vH1_H1v_x2_full;
            AccReal res2_est_x2_full  = proj_res2_x2_full + leak2_raw_x2_full;
            AccReal res2_psd_x2_full  = std::max(AccReal{0}, proj_res2_x2_full) + std::max(AccReal{0}, leak2_raw_x2_full);
            AccReal variance_x2_full  = v_H2v_x2_full - v_H1v_x2_full * v_H1v_x2_full;

            // TODO: Temporary x2 isolation diagnostic. Remove after deciding whether H2 x2 error comes from gemm_x2 or from the x2 MPO2/environments.
            auto x2_to_acc_tensor = [&](const x2::Tensor<Scalar, 3> &t) {
                auto out = Eigen::Tensor<AccScalar, 3>(t.dimensions());
                for(Eigen::Index i = 0; i < t.size(); ++i) out.data()[i] = to_acc(t.hi_data()[i]) + to_acc(t.lo_data()[i]);
                return out;
            };
            auto diff_norm_x2 = [&](const Eigen::Tensor<AccScalar, 3> &ref, const x2::Tensor<Scalar, 3> &x2res) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < ref.size(); ++i) {
                    const auto x2val  = to_acc(x2res.hi_data()[i]) + to_acc(x2res.lo_data()[i]);
                    const auto d      = ref.data()[i] - x2val;
                    sum              += real_acc(conj_acc(d) * d);
                }
                return std::sqrt(sum);
            };
            auto residual_norm2_acc = [&](const Eigen::Tensor<AccScalar, 3> &y, const Eigen::Tensor<AccScalar, 3> &x, AccReal E) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < x.size(); ++i) {
                    const auto r  = y.data()[i] - E * x.data()[i];
                    sum          += real_acc(conj_acc(r) * r);
                }
                return sum;
            };
            Eigen::Tensor<AccScalar, 3> mps_acc       = mps.template cast<AccScalar>();
            Eigen::Tensor<AccScalar, 4> mpo1_acc      = mpo1.template cast<AccScalar>();
            Eigen::Tensor<AccScalar, 4> mpo2_acc      = mpo2.template cast<AccScalar>();
            auto                        enveL_x2_acc  = x2_to_acc_tensor(enve.L.get_blkx2());
            auto                        enveR_x2_acc  = x2_to_acc_tensor(enve.R.get_blkx2());
            auto                        envvL_x2_acc  = x2_to_acc_tensor(envv.L.get_blkx2());
            auto                        envvR_x2_acc  = x2_to_acc_tensor(envv.R.get_blkx2());
            auto                        H1t_x2env_acc = Eigen::Tensor<AccScalar, 3>(mps.dimensions());
            auto                        H2t_x2env_acc = Eigen::Tensor<AccScalar, 3>(mps.dimensions());
            auto                        h1info_x2env  = SetH1MvInfo(ContractionBackend::EIGEN, mpo1.dimensions());
            auto                        h2info_x2env  = SetH2MvInfo(ContractionBackend::EIGEN, mpo2.dimensions());
            tools::common::contraction::matrix_vector_product(H1t_x2env_acc, mps_acc, mpo1_acc, enveL_x2_acc, enveR_x2_acc);
            tools::common::contraction::matrix_vector_product(H2t_x2env_acc, mps_acc, mpo2_acc, envvL_x2_acc, envvR_x2_acc);
            auto    v_acc                   = tenx::VectorMap(mps_acc);
            auto    H1v_x2env_acc           = tenx::VectorMap(H1t_x2env_acc);
            auto    H2v_x2env_acc           = tenx::VectorMap(H2t_x2env_acc);
            AccReal v_H1v_x2env_acc         = real_acc(v_acc.dot(H1v_x2env_acc));
            AccReal v_H2v_x2env_acc         = real_acc(v_acc.dot(H2v_x2env_acc));
            AccReal vH1_H1v_x2env_acc       = real_acc(H1v_x2env_acc.dot(H1v_x2env_acc));
            AccReal proj_res2_x2env_acc     = residual_norm2_acc(H1t_x2env_acc, mps_acc, v_H1v_x2env_acc);
            AccReal leak2_raw_x2env_acc     = v_H2v_x2env_acc - vH1_H1v_x2env_acc;
            AccReal res2_est_x2env_acc      = proj_res2_x2env_acc + leak2_raw_x2env_acc;
            AccReal variance_x2env_acc      = v_H2v_x2env_acc - v_H1v_x2env_acc * v_H1v_x2env_acc;
            AccReal H1_x2_vs_x2env_acc_norm = diff_norm_x2(H1t_x2env_acc, H1t_x2_full);
            AccReal H2_x2_vs_x2env_acc_norm = diff_norm_x2(H2t_x2env_acc, H2t_x2_full);

            VectorType resid1    = H1v - v_H1v * v;
            RealScalar rnorm1    = resid1.norm();
            RealScalar proj_res2 = std::real(resid1.dot(resid1));
            RealScalar leak2_raw = v_H2v - vH1_H1v;
            RealScalar res2_est  = proj_res2 + leak2_raw;
            RealScalar res2_psd  = std::max(RealScalar{0}, proj_res2) + std::max(RealScalar{0}, leak2_raw);
            RealScalar delta     = v_H2v - vH1_H1v;

            tools::log->info("var_opt             = {:.16e}", opt_state.get_variance());
            tools::log->info("var_mrg             = {:.16e}", var_mrg);
            tools::log->info("vH1_H1v             = {:.16e}", vH1_H1v);
            tools::log->info("v_H1v               = {:.16e} diff {:.16e}", v_H1v, v_H1v - std::sqrt(vH1_H1v));
            tools::log->info("v_H1v X2            = {:.16e}", v_H1vx2);
            tools::log->info("v_H1v Q             = {:.16e}", v_H1vQ);
            tools::log->info("v_H1v²              = {:.16e} diff {:.16e}", v_H1v * v_H1v, v_H1v * v_H1v - vH1_H1v);
            tools::log->info("v_H2v               = {:.16e}", v_H2v);
            tools::log->info("v_H2v X2            = {:.16e}", v_H2vx2);
            tools::log->info("v_H2v X2 full       = {:.16e}", v_H2v_x2_full);
            tools::log->info("v_H2v X2env fp128   = {:.16e}", v_H2v_x2env_acc);
            tools::log->info("v_H2v Q             = {:.16e}", v_H2vQ);
            tools::log->info("vH1_H1v X2 full     = {:.16e}", vH1_H1v_x2_full);
            tools::log->info("vH1_H1v X2env fp128 = {:.16e}", vH1_H1v_x2env_acc);
            tools::log->info("|H1x2-X2env_fp128|  = {:.16e}", H1_x2_vs_x2env_acc_norm);
            tools::log->info("|H2x2-X2env_fp128|  = {:.16e}", H2_x2_vs_x2env_acc_norm);
            tools::log->info("|H1v-vH1v*v|        = {:.16e}", rnorm1);
            tools::log->info("proj_res2           = {:.16e}", proj_res2);
            tools::log->info("proj_res2 X2 full   = {:.16e}", proj_res2_x2_full);
            tools::log->info("proj_res2 X2env fp128 = {:.16e}", proj_res2_x2env_acc);
            tools::log->info("leak2_raw           = {:.16e}", leak2_raw);
            tools::log->info("leak2_raw X2 full   = {:.16e}", leak2_raw_x2_full);
            tools::log->info("leak2_raw X2env fp128 = {:.16e}", leak2_raw_x2env_acc);
            tools::log->info("delta               = {:.16e}", delta);
            tools::log->info("E_local est         = {:.16e}", std::sqrt(vH1_H1v + rnorm1));
            tools::log->info("sqrt(|H1v-v_H1v*v|) = {:.16e}", std::sqrt(rnorm1));
            tools::log->info("res2_est            = {:.16e}", res2_est);
            tools::log->info("res2_est X2 full    = {:.16e}", res2_est_x2_full);
            tools::log->info("res2_est X2env fp128 = {:.16e}", res2_est_x2env_acc);
            tools::log->info("res2_psd            = {:.16e}", res2_psd);
            tools::log->info("res2_psd X2 full    = {:.16e}", res2_psd_x2_full);
            tools::log->info("energy variance     = {:.16e}", v_H2v - v_H1v * v_H1v);
            tools::log->info("energy variance X2f = {:.16e}", variance_x2_full);
            tools::log->info("energy variance X2env fp128 = {:.16e}", variance_x2env_acc);
        }

        {
            // tensors.get_state().clear_cache();
            // tensors.get_state().clear_measurements();
            tensors.get_edges().eject_edges_all();
            tensors.rebuild_edges();
            tensors.clear_cache();
            tensors.clear_measurements();
            for(const auto &env : tensors.get_edges().eneL) {
                msg2.emplace_back(fmt::format("eneL[{:2}]: {} norm {:.8e}", env->get_position(), env->get_unique_id(), fp(tenx::norm(env->get_block()))));
            }
            for(const auto &env : tensors.get_edges().eneR) {
                msg2.emplace_back(fmt::format("eneR[{:2}]: {} norm {:.8e}", env->get_position(), env->get_unique_id(), fp(tenx::norm(env->get_block()))));
            }
            const auto &mps         = tensors.get_state().template get_multisite_mps<Scalar>();
            const auto &mpo1        = tensors.get_model().template get_multisite_mpo<Scalar>();
            const auto &mpo2        = tensors.get_model().template get_multisite_mpo_squared<Scalar>();
            const auto &enve        = tensors.get_edges().get_multisite_env_ene();
            const auto &envv        = tensors.get_edges().get_multisite_env_var();
            auto        H1t         = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H1tx2       = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H1tQ        = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H2t         = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H2tx2       = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H2tQ        = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H1t_x2_full = x2::Tensor<Scalar, 3>();
            auto        H2t_x2_full = x2::Tensor<Scalar, 3>();
            {
                auto h1info = SetH1MvInfo(ContractionBackend::TBLIS, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionBackend::TBLIS, mpo2.dimensions());
                tools::log->info("contracting with tblis");
                tools::common::contraction::matrix_vector_product(H1t, mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
                tools::common::contraction::matrix_vector_product(H2t, mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());
            }
            {
                auto h1info = SetH1MvInfo(ContractionPrecision::X2, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionPrecision::X2, mpo2.dimensions());
                tools::log->info("contracting with x2");
                H1t_x2_full = tools::common::contraction::matrix_vector_product_x2(mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
                H2t_x2_full = tools::common::contraction::matrix_vector_product_x2(mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());
                H1tx2       = H1t_x2_full.to_EigenTensor();
                H2tx2       = H2t_x2_full.to_EigenTensor();
            }
            {
                auto h2info = SetH2MvInfo(ContractionBackend::EIGEN, mpo2.dimensions());
                tools::log->info("contracting with fp128");
                using ScalarQ = std::conditional_t<std::is_floating_point_v<Scalar>, fp128, cx128>;
                Eigen::Tensor<ScalarQ, 3> H1t_q(mps.dimensions());
                Eigen::Tensor<ScalarQ, 3> H2t_q(mps.dimensions());
                Eigen::Tensor<ScalarQ, 3> mps_q   = mps.template cast<ScalarQ>();
                Eigen::Tensor<ScalarQ, 4> mpo1_q  = mpo1.template cast<ScalarQ>();
                Eigen::Tensor<ScalarQ, 4> mpo2_q  = mpo2.template cast<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> enveL_q = enve.L.template get_block_as<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> enveR_q = enve.R.template get_block_as<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> envvL_q = envv.L.template get_block_as<ScalarQ>();
                Eigen::Tensor<ScalarQ, 3> envvR_q = envv.R.template get_block_as<ScalarQ>();
                tools::common::contraction::matrix_vector_product(H1t_q, mps_q, mpo1_q, enveL_q, enveR_q);
                tools::common::contraction::matrix_vector_product(H2t_q, mps_q, mpo2_q, envvL_q, envvR_q);
                H1tQ = H1t_q.template cast<Scalar>();
                H2tQ = H2t_q.template cast<Scalar>();
            }
            using VectorType   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            auto       v       = tenx::VectorMap(mps);
            auto       H1v     = tenx::VectorMap(H1t);
            auto       H1vx2   = tenx::VectorMap(H1tx2);
            auto       H1vQ    = tenx::VectorMap(H1tQ);
            auto       H2v     = tenx::VectorMap(H2t);
            auto       H2vx2   = tenx::VectorMap(H2tx2);
            auto       H2vQ    = tenx::VectorMap(H2tQ);
            RealScalar vH1_H1v = std::real(H1v.dot(H1v));
            RealScalar v_H1v   = std::real(v.dot(H1v));
            RealScalar v_H1vx2 = std::real(v.dot(H1vx2));
            RealScalar v_H1vQ  = std::real(v.dot(H1vQ));
            RealScalar v_H2v   = std::real(v.dot(H2v));
            RealScalar v_H2vx2 = std::real(v.dot(H2vx2));
            RealScalar v_H2vQ  = std::real(v.dot(H2vQ));

            using AccReal   = fp128;
            using AccScalar = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, std::complex<AccReal>, AccReal>;
            auto to_acc     = [](const Scalar &z) -> AccScalar {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return AccScalar{static_cast<AccReal>(std::real(z)), static_cast<AccReal>(std::imag(z))};
                else
                    return static_cast<AccReal>(z);
            };
            auto conj_acc = [](const AccScalar &z) -> AccScalar {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return std::conj(z);
                else
                    return z;
            };
            auto real_acc = [](const AccScalar &z) -> AccReal {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex == 1)
                    return std::real(z);
                else
                    return z;
            };
            auto dot_x2_real = [&](const x2::Tensor<Scalar, 3> &y) -> AccReal {
                AccScalar sum = AccScalar{0};
                for(Eigen::Index i = 0; i < mps.size(); ++i) {
                    const auto xacc  = to_acc(mps.data()[i]);
                    const auto yacc  = to_acc(y.hi_data()[i]) + to_acc(y.lo_data()[i]);
                    sum             += conj_acc(xacc) * yacc;
                }
                return real_acc(sum);
            };
            auto norm2_x2 = [&](const x2::Tensor<Scalar, 3> &y) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < y.size(); ++i) {
                    const auto yacc  = to_acc(y.hi_data()[i]) + to_acc(y.lo_data()[i]);
                    sum             += real_acc(conj_acc(yacc) * yacc);
                }
                return sum;
            };
            auto residual_norm2_x2 = [&](const x2::Tensor<Scalar, 3> &y, AccReal E) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < mps.size(); ++i) {
                    const auto xacc  = to_acc(mps.data()[i]);
                    const auto yacc  = to_acc(y.hi_data()[i]) + to_acc(y.lo_data()[i]);
                    const auto racc  = yacc - E * xacc;
                    sum             += real_acc(conj_acc(racc) * racc);
                }
                return sum;
            };

            AccReal v_H1v_x2_full     = dot_x2_real(H1t_x2_full);
            AccReal v_H2v_x2_full     = dot_x2_real(H2t_x2_full);
            AccReal vH1_H1v_x2_full   = norm2_x2(H1t_x2_full);
            AccReal proj_res2_x2_full = residual_norm2_x2(H1t_x2_full, v_H1v_x2_full);
            AccReal leak2_raw_x2_full = v_H2v_x2_full - vH1_H1v_x2_full;
            AccReal res2_est_x2_full  = proj_res2_x2_full + leak2_raw_x2_full;
            AccReal res2_psd_x2_full  = std::max(AccReal{0}, proj_res2_x2_full) + std::max(AccReal{0}, leak2_raw_x2_full);
            AccReal variance_x2_full  = v_H2v_x2_full - v_H1v_x2_full * v_H1v_x2_full;

            // TODO: Temporary x2 isolation diagnostic. Remove after deciding whether H2 x2 error comes from gemm_x2 or from the x2 MPO2/environments.
            auto x2_to_acc_tensor = [&](const x2::Tensor<Scalar, 3> &t) {
                auto out = Eigen::Tensor<AccScalar, 3>(t.dimensions());
                for(Eigen::Index i = 0; i < t.size(); ++i) out.data()[i] = to_acc(t.hi_data()[i]) + to_acc(t.lo_data()[i]);
                return out;
            };
            auto diff_norm_x2 = [&](const Eigen::Tensor<AccScalar, 3> &ref, const x2::Tensor<Scalar, 3> &x2res) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < ref.size(); ++i) {
                    const auto x2val  = to_acc(x2res.hi_data()[i]) + to_acc(x2res.lo_data()[i]);
                    const auto d      = ref.data()[i] - x2val;
                    sum              += real_acc(conj_acc(d) * d);
                }
                return std::sqrt(sum);
            };
            auto residual_norm2_acc = [&](const Eigen::Tensor<AccScalar, 3> &y, const Eigen::Tensor<AccScalar, 3> &x, AccReal E) -> AccReal {
                AccReal sum = AccReal{0};
                for(Eigen::Index i = 0; i < x.size(); ++i) {
                    const auto r  = y.data()[i] - E * x.data()[i];
                    sum          += real_acc(conj_acc(r) * r);
                }
                return sum;
            };
            Eigen::Tensor<AccScalar, 3> mps_acc       = mps.template cast<AccScalar>();
            Eigen::Tensor<AccScalar, 4> mpo1_acc      = mpo1.template cast<AccScalar>();
            Eigen::Tensor<AccScalar, 4> mpo2_acc      = mpo2.template cast<AccScalar>();
            auto                        enveL_x2_acc  = x2_to_acc_tensor(enve.L.get_blkx2());
            auto                        enveR_x2_acc  = x2_to_acc_tensor(enve.R.get_blkx2());
            auto                        envvL_x2_acc  = x2_to_acc_tensor(envv.L.get_blkx2());
            auto                        envvR_x2_acc  = x2_to_acc_tensor(envv.R.get_blkx2());
            auto                        H1t_x2env_acc = Eigen::Tensor<AccScalar, 3>(mps.dimensions());
            auto                        H2t_x2env_acc = Eigen::Tensor<AccScalar, 3>(mps.dimensions());
            auto                        h1info_x2env  = SetH1MvInfo(ContractionBackend::EIGEN, mpo1.dimensions());
            auto                        h2info_x2env  = SetH2MvInfo(ContractionBackend::EIGEN, mpo2.dimensions());
            tools::common::contraction::matrix_vector_product(H1t_x2env_acc, mps_acc, mpo1_acc, enveL_x2_acc, enveR_x2_acc);
            tools::common::contraction::matrix_vector_product(H2t_x2env_acc, mps_acc, mpo2_acc, envvL_x2_acc, envvR_x2_acc);
            auto    v_acc                   = tenx::VectorMap(mps_acc);
            auto    H1v_x2env_acc           = tenx::VectorMap(H1t_x2env_acc);
            auto    H2v_x2env_acc           = tenx::VectorMap(H2t_x2env_acc);
            AccReal v_H1v_x2env_acc         = real_acc(v_acc.dot(H1v_x2env_acc));
            AccReal v_H2v_x2env_acc         = real_acc(v_acc.dot(H2v_x2env_acc));
            AccReal vH1_H1v_x2env_acc       = real_acc(H1v_x2env_acc.dot(H1v_x2env_acc));
            AccReal proj_res2_x2env_acc     = residual_norm2_acc(H1t_x2env_acc, mps_acc, v_H1v_x2env_acc);
            AccReal leak2_raw_x2env_acc     = v_H2v_x2env_acc - vH1_H1v_x2env_acc;
            AccReal res2_est_x2env_acc      = proj_res2_x2env_acc + leak2_raw_x2env_acc;
            AccReal variance_x2env_acc      = v_H2v_x2env_acc - v_H1v_x2env_acc * v_H1v_x2env_acc;
            AccReal H1_x2_vs_x2env_acc_norm = diff_norm_x2(H1t_x2env_acc, H1t_x2_full);
            AccReal H2_x2_vs_x2env_acc_norm = diff_norm_x2(H2t_x2env_acc, H2t_x2_full);

            VectorType resid1    = H1v - v_H1v * v;
            RealScalar rnorm1    = resid1.norm();
            RealScalar proj_res2 = std::real(resid1.dot(resid1));
            RealScalar leak2_raw = v_H2v - vH1_H1v;
            RealScalar res2_est  = proj_res2 + leak2_raw;
            RealScalar res2_psd  = std::max(RealScalar{0}, proj_res2) + std::max(RealScalar{0}, leak2_raw);
            RealScalar delta     = v_H2v - vH1_H1v;

            tools::log->info("var_opt             = {:.16e}", opt_state.get_variance());
            tools::log->info("var_mrg             = {:.16e}", var_mrg);
            tools::log->info("vH1_H1v             = {:.16e}", vH1_H1v);
            tools::log->info("v_H1v               = {:.16e} diff {:.16e}", v_H1v, v_H1v - std::sqrt(vH1_H1v));
            tools::log->info("v_H1v X2            = {:.16e}", v_H1vx2);
            tools::log->info("v_H1v Q             = {:.16e}", v_H1vQ);
            tools::log->info("v_H1v²              = {:.16e} diff {:.16e}", v_H1v * v_H1v, v_H1v * v_H1v - vH1_H1v);
            tools::log->info("v_H2v               = {:.16e}", v_H2v);
            tools::log->info("v_H2v X2            = {:.16e}", v_H2vx2);
            tools::log->info("v_H2v X2 full       = {:.16e}", v_H2v_x2_full);
            tools::log->info("v_H2v X2env fp128   = {:.16e}", v_H2v_x2env_acc);
            tools::log->info("v_H2v Q             = {:.16e}", v_H2vQ);
            tools::log->info("vH1_H1v X2 full     = {:.16e}", vH1_H1v_x2_full);
            tools::log->info("vH1_H1v X2env fp128 = {:.16e}", vH1_H1v_x2env_acc);
            tools::log->info("|H1x2-X2env_fp128|  = {:.16e}", H1_x2_vs_x2env_acc_norm);
            tools::log->info("|H2x2-X2env_fp128|  = {:.16e}", H2_x2_vs_x2env_acc_norm);
            tools::log->info("|H1v-vH1v*v|        = {:.16e}", rnorm1);
            tools::log->info("proj_res2           = {:.16e}", proj_res2);
            tools::log->info("proj_res2 X2 full   = {:.16e}", proj_res2_x2_full);
            tools::log->info("proj_res2 X2env fp128 = {:.16e}", proj_res2_x2env_acc);
            tools::log->info("leak2_raw           = {:.16e}", leak2_raw);
            tools::log->info("leak2_raw X2 full   = {:.16e}", leak2_raw_x2_full);
            tools::log->info("leak2_raw X2env fp128 = {:.16e}", leak2_raw_x2env_acc);
            tools::log->info("delta               = {:.16e}", delta);
            tools::log->info("E_local est         = {:.16e}", std::sqrt(vH1_H1v + rnorm1));
            tools::log->info("sqrt(|H1v-v_H1v*v|) = {:.16e}", std::sqrt(rnorm1));
            tools::log->info("res2_est            = {:.16e}", res2_est);
            tools::log->info("res2_est X2 full    = {:.16e}", res2_est_x2_full);
            tools::log->info("res2_est X2env fp128 = {:.16e}", res2_est_x2env_acc);
            tools::log->info("res2_psd            = {:.16e}", res2_psd);
            tools::log->info("res2_psd X2 full    = {:.16e}", res2_psd_x2_full);
            tools::log->info("energy variance     = {:.16e}", v_H2v - v_H1v * v_H1v);
            tools::log->info("energy variance X2f = {:.16e}", variance_x2_full);
            tools::log->info("energy variance X2env fp128 = {:.16e}", variance_x2env_acc);
        }
        for(size_t i = 0; i < msg1.size(); ++i) { tools::log->info("{} | {}", msg1.at(i), msg2.at(i)); }

        exit(1);
    }

    status.energy_variance_lowest = std::min(static_cast<double>(var_mrg), status.energy_variance_lowest);
    var_delta                     = var_mrg - var_latest;
    ene_delta                     = ene_mrg - ene_latest;
    var_latest                    = var_mrg;
    ene_latest                    = ene_mrg;
    auto bondexp_result           = tools::finite::bex::expand_bonds(tensors, get_bond_expansion_config(BondExpansionOrder::POSTOPT));

    auto ene_ini = initial_state.get_energy();
    auto ene_opt = opt_state.get_energy();
    auto ene_exp = bondexp_result.ene_new;
    auto var_ini = initial_state.get_variance();
    auto var_opt = opt_state.get_variance();
    auto var_exp = bondexp_result.var_new;

    ene_delta_opt = ene_opt - ene_ini;
    ene_delta_svd = ene_exp - ene_opt;
    var_delta_opt = std::abs(var_opt - var_ini);
    var_delta_svd = std::abs(var_exp - var_opt);
    tools::log->trace("Energy   change Δsvd/Δopt: {:.16f} | ini {:.16f} opt {:.16f} exp {:.16f}", ene_delta_svd / ene_delta_opt, ene_ini, ene_opt, ene_exp);
    tools::log->trace("Variance change Δsvd/Δopt: {:.16f} | ini {:.16f} opt {:.16f} exp {:.16f}", var_delta_svd / var_delta_opt, var_ini, var_opt, var_exp);

    last_optsolver = opt_state.get_optsolver();
    last_optalgo   = opt_state.get_optalgo();

    if constexpr(settings::debug) {
        if(tools::log->level() <= spdlog::level::trace) tools::log->trace("Truncation errors: {::8.3e}", tensors.state->get_truncation_errors_active());
        if(tools::log->level() <= spdlog::level::trace) tools::log->trace("Truncation errors: {::8.3e}", tensors.state->get_truncation_errors());
        tools::log->debug("Before update            : variance {:8.2e} | mps dims {}", initial_state.get_variance(), initial_state.get_tensor().dimensions());
        tools::log->debug("After  optimization      : variance {:8.2e} | mps dims {}", opt_state.get_variance(), opt_state.get_tensor().dimensions());
        tools::log->debug("After  merge             : variance {:8.2e} | mps dims {}", var_mrg, tensors.get_state().get_bond_dims_active());
        tools::log->debug("After  bond expansion    : variance {:8.2e} | mps dims {}", var_exp, bondexp_result.dimMP);
    }

    if constexpr(settings::debug) tensors.assert_validity();
}

template<typename Scalar>
void xdmrg<Scalar>::find_energy_range() {
    // We only need to find an energy range if we are targeting a particular energy density window or target
    if(status.opt_ritz != OptRitz::TE) return; // We only need the extremal for OptRitz::TED

    tools::log->trace("Finding energy range");
    auto t_init = tid::tic_scope("init");
    // Here we define a set of tasks for fdmrg in order to produce the lowest and highest energy eigenstates,
    // We don't want it to randomize its own model, so we implant our current model before running the tasks.

    std::deque<fdmrg_task> gs_tasks = {fdmrg_task::INIT_CLEAR_STATUS, fdmrg_task::INIT_BOND_LIMITS, fdmrg_task::INIT_TRNC_LIMITS,
                                       fdmrg_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE, fdmrg_task::FIND_GROUND_STATE};

    std::deque<fdmrg_task> hs_tasks = {fdmrg_task::INIT_CLEAR_STATUS, fdmrg_task::INIT_BOND_LIMITS, fdmrg_task::INIT_TRNC_LIMITS,
                                       fdmrg_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE, fdmrg_task::FIND_HIGHEST_STATE};
    // Find the lowest energy state
    {
        auto          t_gs = tid::tic_scope("fDMRG");
        fdmrg<Scalar> fdmrg_gs{};
        fdmrg_gs.tensors.get_model() = tensors.get_model(); // Copy the model
        fdmrg_gs.tensors.state->set_name("state_emin");
        tools::log = tools::Logger::setLogger(fmt::format("{}-gs", status.algo_type_sv()), settings::console::loglevel, settings::console::timestamp);
        fdmrg_gs.run_task_list(gs_tasks);
        status.energy_min = static_cast<double>(tools::finite::measure::energy(fdmrg_gs.tensors));
        fdmrg_gs.h5file   = h5file;
        write_to_file(fdmrg_gs.tensors.get_state(), fdmrg_gs.tensors.get_model(), fdmrg_gs.tensors.get_edges(), StorageEvent::EMIN, CopyPolicy::OFF);
    }

    // Find the highest energy state
    {
        auto          t_hs = tid::tic_scope("fDMRG");
        fdmrg<Scalar> fdmrg_hs{};
        fdmrg_hs.tensors.get_model() = tensors.get_model(); // Copy the model
        fdmrg_hs.tensors.state->set_name("state_emax");
        tools::log = tools::Logger::setLogger(fmt::format("{}-hs", status.algo_type_sv()), settings::console::loglevel, settings::console::timestamp);
        fdmrg_hs.run_task_list(hs_tasks);
        status.energy_max = static_cast<double>(tools::finite::measure::energy(fdmrg_hs.tensors));
        fdmrg_hs.h5file   = h5file;
        write_to_file(fdmrg_hs.tensors.get_state(), fdmrg_hs.tensors.get_model(), fdmrg_hs.tensors.get_edges(), StorageEvent::EMAX, CopyPolicy::OFF);
    }

    // Reset our logger
    tools::log = tools::Logger::getLogger(fmt::format("{}", status.algo_type_sv()));
}

template<typename Scalar>
void xdmrg<Scalar>::set_energy_shift_mpo(std::optional<RealScalar> eshift) {
    // In xdmrg, the MPO energy shift defines the target energy E_tgt used by the excited-state search.
    // If "eshift" is not given explicitly, we use status.energy_tgt.
    //
    // This shift is used by all excited-state DMRG modes:
    //   - DMRG_X / DMRG_X_HYBRID: local energies are measured relative to E_tgt, so eigenpairs near the target sit near zero and can be selected/refined around
    //   that reference.
    //   - DMRG_FOLDED           : optimize the folded operator (H-E_tgt)^2, whose smallest eigenvalue corresponds to the state nearest E_tgt.
    //   - DMRG_GSI              : apply the same target shift consistently to the generalized problem, H-E_tgt and (H-E_tgt)^2.
    //
    // The same shift also gives the natural "distance to target" form of the variance:
    //      Var(H) = <(H-E_tgt)^2> - <H-E_tgt>^2
    //             = <H^2> - 2<H>E_tgt + E_tgt^2 - (<H> - E_tgt)^2
    //             = <H^2> - <H>^2.
    // Therefore, the physical variance is unchanged by the shift, while the shifted expectation
    // <H-E_tgt> = E-E_tgt directly measures how far the current state energy lies from the target.
    if(not tools::finite::pos::position_is_inward_edge(tensors)) return;
    constexpr auto eps = std::numeric_limits<RealScalar>::epsilon();
    if(var_latest < eps * 10) return; // No need to improve precision further.

    if(not eshift) eshift = std::real(status.energy_tgt);
    Scalar energy_shift = narrow_cast<Scalar>(eshift.value());
    tensors.set_energy_shift_mpo(energy_shift);
}

template<typename Scalar>
void xdmrg<Scalar>::update_time_step() {
    tools::log->trace("Updating time step");
    status.delta_t = std::complex<double>(1e-6, 0);
}
