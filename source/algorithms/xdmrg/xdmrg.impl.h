#pragma once
#include "../fdmrg.h"
#include "../xdmrg.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "io/fmt_custom.h"
#include "math/eig.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/svd.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction/contraction_policy.h"
#include "tools/common/contraction/matrix_vector_product.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/common/prof.h"
#include "tools/finite/env/BondExpansionConfig.h"
#include "tools/finite/env/BondExpansionResult.h"
#include "tools/finite/h5.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/norm.impl.h"
#include "tools/finite/measure/residual.impl.h"
#include "tools/finite/mps.h"
#include "tools/finite/multisite.h"
#include "tools/finite/ops.h"
#include "tools/finite/opt.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
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

        // Set a possibly new energy target
        // status.energy_tgt = settings::xdmrg::energy_spectrum_shift;

        // Reload the bond and truncation error limits (could be different in the config compared to the status we just loaded)
        double long_max                   = static_cast<double>(std::numeric_limits<long>::max());
        double bond_max                   = std::min(long_max, std::pow(2.0, settings::model::model_size / 2));
        status.bond_max                   = std::min({status.bond_max, safe_cast<long>(bond_max), settings::get_bond_max(status.algo_type)});
        status.bond_min                   = std::max(status.bond_min, settings::get_bond_min(status.algo_type));
        status.bond_lim                   = std::clamp(status.bond_lim, 1l, status.bond_max);
        status.bond_limit_has_reached_max = status.bond_lim == status.bond_max;
        tools::log->info("Initialized bond dimension limits: min {} lim {} max {}", status.bond_min, status.bond_lim, status.bond_max);

        status.trnc_min                   = settings::precision::svd_truncation_min;
        status.trnc_max                   = settings::precision::svd_truncation_max;
        status.trnc_lim                   = std::clamp(status.trnc_lim, status.trnc_min, status.trnc_max);
        status.trnc_limit_has_reached_min = status.trnc_lim == status.trnc_min;
        tools::log->info("Initialized truncation error limits: max {:8.2e} lim {:8.2e} min {:8.2e}", status.trnc_max, status.trnc_lim, status.trnc_min);

        status.opt_ritz = settings::xdmrg::ritz;

        // Apply shifts and compress the model
        tensors.move_center_point_to_inward_edge();
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
                    energy_density_target));
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

    initialize_state(ResetReason::INIT, settings::strategy::initial_state); // Second use of random!
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
        diffs1.topRows(N1) = (evals1.bottomRows(N1) - evals1.topRows(N1));
        diffs2.topRows(N2) = (evals2.bottomRows(N2) - evals2.topRows(N2));

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

    auto get_backend = [&]() {
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
            tools::log->warn("Switched to X2 backend: <H²>/|H²| = {:.4e} < 1000 * eps", fp(quot));
            return ContractionBackend::X2;
        } else {
            tools::log->info("Selected TBLIS backend: <H²>/|H²| = {:.4e} > 1000 * eps", fp(quot));
            return ContractionBackend::TBLIS;
        }
    };

    while(true) {
        auto backend = get_backend();
        auto h1info  = SetH1MvInfo(ContractionBackend::X2);
        auto h2info  = SetH2MvInfo(ContractionBackend::X2);
        auto envinfo = SetEnvInfo(ContractionBackend::X2);

        tools::log->trace("Starting step {}, iter {}, pos {}, dir {}, backend:{}", status.step, status.iter, status.position, status.direction,
                          enum2sv(backend));
        // Apply end-of-half-sweep actions
        // Updating bond dimension must go first since it decides based on truncation error, but a projection+normalize resets truncation.
        update_bond_dimension_limit();   // Updates the bond dimension if the state precision is being limited by bond dimension
        update_truncation_error_limit(); // Updates the truncation error limit if the state is being truncated
        update_mixing_factor();          // Updates the mixing factor used in DMRG3S
        set_energy_shift_mpo();          // Shifts the energy H -> H-<E> by subtracting E/L on each MPO.
        set_parity_shift_mpo();          // Shifts the energy spectrum of states with opposite parity away from the current energy.
        set_parity_shift_mpo_squared();  // Shifts the energy-squared spectrum of states with opposite parity up by 1 (makes sense with ritz == SM)
        rebuild_tensors();               // Rebuilds mpos (and compresses them) and edges, only if they were modified.
        try_projection();                // Tries to project the state to the nearest global spin parity sector along settings::strategy::target_axis
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
        auto envinfo     = SetEnvInfo(ContractionBackend::X2);
        auto bexp_result = expand_bonds(BondExpansionOrder::PREOPT);
    }

    auto opt_meta = get_opt_meta();

    tools::log->debug("Starting {} iter {} | step {} | pos {} | dir {} | ritz {} | type {}", status.algo_type_sv(), status.iter, status.step, status.position,
                      status.direction, enum2sv(settings::get_ritz(status.algo_type)), enum2sv(opt_meta.optType));
    // Try activating the sites asked for;
    tensors.activate_sites(opt_meta.chosen_sites);
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
    if(opt_state.get_eigs_rnorm()     > static_cast<RealScalar>(settings::precision::eigs_abstol_max)) opt_meta.optExit |= OptExit::FAIL_RESIDUAL;
    if(opt_state.get_eigs_nev()       == 0 and
       opt_meta.optSolver             == OptSolver::EIGS                                          ) opt_meta.optExit |= OptExit::FAIL_RESIDUAL; // No convergence
    if(opt_state.get_overlap()        < static_cast<RealScalar>(0.010)                            ) opt_meta.optExit |= OptExit::FAIL_OVERLAP;
    if(opt_state.get_relchange()      > static_cast<RealScalar>(1.001)                            ) opt_meta.optExit |= OptExit::FAIL_WORSENED;
    else if(opt_state.get_relchange() > static_cast<RealScalar>(0.999)                            ) opt_meta.optExit |= OptExit::FAIL_NOCHANGE;
    /* clang-format on */
    opt_state.set_optexit(opt_meta.optExit);

    tools::log->trace("Optimization [{}|{}]: {}. Variance change {:8.2e} --> {:8.2e} ({:.3f} %)", enum2sv(opt_meta.optAlgo), enum2sv(opt_meta.optSolver),
                      flag2str(opt_meta.optExit), fp(var_latest), fp(opt_state.get_variance()), fp(opt_state.get_relchange() * 100));
    if(opt_state.get_relchange() > 1000) {
        tools::log->warn("Variance increase by x {:.2e} | variance new {:.3e} | variance old {:.3e}", fp(opt_state.get_relchange()),
                         fp(opt_state.get_variance()), fp(var_latest));
    }

    if(tools::log->level() <= spdlog::level::debug) {
        tools::log->debug("Optimization result: {:<24} | E {:<20.16f}| σ²H {:<8.2e} | rnorm {:8.2e} | overlap {:.16f} | "
                          "sites {} | {:20} | {} | time {:.2e} s",
                          opt_state.get_name(), fp(opt_state.get_energy()), fp(opt_state.get_variance()), fp(opt_state.get_eigs_rnorm()),
                          fp(opt_state.get_overlap()), opt_state.get_sites(),
                          fmt::format("[{}][{}]", enum2sv(opt_state.get_optalgo()), enum2sv(opt_state.get_optsolver())), flag2str(opt_state.get_optexit()),
                          opt_state.get_time());
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
        tools::log->info("Variance is negative! {:.16e}", fp(var_mrg));
        // Disable mpo compression
        settings::precision::use_compressed_mpo         = MpoCompress::NONE;
        settings::precision::use_compressed_mpo_squared = MpoCompress::NONE;
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
            tools::log->info("NL err: {:.16e}", fp(NLerr));
            tools::log->info("NR err: {:.16e}", fp(NRerr));
        }

        {
            auto                t_res      = tid::tic_token("<ψ|H²_global ψ>");
            StateFinite<Scalar> tmp_state1 = tensors.get_state();
            StateFinite<Scalar> tmp_state2 = tensors.get_state();
            auto                mpos1      = tensors.get_model().get_mpo_tensors(Scalar{0}, MposWithEdges::ON, MpoCompress::NONE);
            auto                mpos2      = tensors.get_model().get_mpo2_tensors(Scalar{0}, MposWithEdges::ON, MpoCompress::NONE);
            auto                svdcfg     = svd::config(8192, 1e-20);
            svdcfg.svd_lib                 = svd::lib::lapacke;
            svdcfg.svd_rtn                 = svd::rtn::gesdd;
            tools::finite::ops::apply_mpos_general(tmp_state1, mpos1, svdcfg);
            tools::finite::ops::apply_mpos_general(tmp_state2, mpos2, svdcfg);
            RealScalar E1_global = std::real(tools::finite::ops::overlap<Scalar>(tmp_state1, tensors.get_state()));
            RealScalar E2_global = std::real(tools::finite::ops::overlap<Scalar>(tmp_state2, tensors.get_state()));
            tools::log->info("H²              <ψ| H²_global ψ>                                = {:.16e} | t = {:.4e}", fp(E2_global),
                             t_res->get_last_interval());

            RealScalar VarH = E2_global - E1_global * E1_global;
            tools::log->info("energy variance <H²_global> - <H_global>²                       = {:.16e} | t = {:.4e}", fp(VarH), t_res->get_last_interval());
        }
        {
            auto                t_res        = tid::tic_token("<ψ (H_global-E_local) | (H_global-E_local) ψ>");
            StateFinite<Scalar> tmp_state    = tensors.get_state();
            auto                L            = tensors.template get_length<RealScalar>();
            auto                mpos_shifted = tensors.get_model().get_mpo_tensors(ene_mrg / L, MposWithEdges::ON, MpoCompress::NONE);
            auto                svdcfg       = svd::config(8192, 1e-20);
            tools::finite::ops::apply_mpos_general(tmp_state, mpos_shifted, svdcfg);
            RealScalar VarH_global = std::real(tools::finite::ops::overlap<Scalar>(tmp_state, tmp_state));
            tools::log->info("energy variance <ψ (H_global-E_local) | (H_global-E_local) ψ>   = {:.16e} | t = {:.4e}", fp(VarH_global),
                             t_res->get_last_interval());
        }
        {
            using ScalarL       = Scalar;
            using RealScalarL   = Eigen::NumTraits<ScalarL>::Real;
            using RealScalar128 = fp128;
            using Scalar128     = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, std::complex<RealScalar128>, RealScalar128>;

            auto tensorsL = tensors.template cast<ScalarL>();
            auto envi     = SetEnvInfo(ContractionBackend::X2);
            auto mv1i     = SetH1MvInfo(ContractionBackend::X2);
            auto mv2i     = SetH2MvInfo(ContractionBackend::X2);
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

            tools::log->info("X2    var_opt            = {:.16e}", fp(opt_state.get_variance()));
            tools::log->info("X2    var_mrg            = {:.16e}", fp(var_mrg));
            tools::log->info("X2    |H1v|              = {:.16e}", fp(H1v.norm()));
            tools::log->info("X2    |H2v|              = {:.16e}", fp(H2v.norm()));
            tools::log->info("X2    v_H1v              = {:.16e}", fp(v_H1v));
            tools::log->info("X2    v_H2v              = {:.16e}", fp(v_H2v));
            tools::log->info("X2    v_H2v    (fp128)   = {:.16e}", fp(v_H2v_fp128));
            tools::log->info("X2    energy variance    = {:.16e}", fp(v_H2v - v_H1v * v_H1v));
            tools::log->info("X2    |enveL|            = {:.16e}", fp(enveLnorm));
            tools::log->info("X2    |enveR|            = {:.16e}", fp(enveRnorm));
            tools::log->info("X2    |envvL|            = {:.16e}", fp(envvLnorm));
            tools::log->info("X2    |envvR|            = {:.16e}", fp(envvRnorm));
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

            tools::log->info("FP64  |H1v|              = {:.16e}", fp(H1v.norm()));
            tools::log->info("FP64  |H2v|              = {:.16e}", fp(H2v.norm()));
            tools::log->info("FP64  v_H1v              = {:.16e}", fp(v_H1v));
            tools::log->info("FP64  v_H2v              = {:.16e}", fp(v_H2v));
            tools::log->info("FP64  energy variance    = {:.16e}", fp(v_H2v - v_H1v * v_H1v));
            tools::log->info("FP64  |enveL|            = {:.16e}", fp(enveLnorm));
            tools::log->info("FP64  |enveR|            = {:.16e}", fp(enveRnorm));
            tools::log->info("FP64  |envvL|            = {:.16e}", fp(envvLnorm));
            tools::log->info("FP64  |envvR|            = {:.16e}", fp(envvRnorm));
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

            tools::log->info("FP128 |H1v|              = {:.16e}", fp(H1v.norm()));
            tools::log->info("FP128 |H2v|              = {:.16e}", fp(H2v.norm()));
            tools::log->info("FP128 v_H1v              = {:.16e}", fp(v_H1v));
            tools::log->info("FP128 v_H2v              = {:.16e}", fp(v_H2v));
            tools::log->info("FP128 energy variance    = {:.16e}", fp(v_H2v - v_H1v * v_H1v));
            tools::log->info("FP128 |enveL|            = {:.16e}", fp(enveLnorm));
            tools::log->info("FP128 |enveR|            = {:.16e}", fp(enveRnorm));
            tools::log->info("FP128 |envvL|            = {:.16e}", fp(envvLnorm));
            tools::log->info("FP128 |envvR|            = {:.16e}", fp(envvRnorm));
        }

        std::vector<std::string> msg1;
        std::vector<std::string> msg2;
        {
            // tensors.get_state().clear_cache();
            // tensors.get_state().clear_measurements();
            // tensors.get_edges().eject_edges_all();
            // TODO: COMPARE THE IDS/NORMS/DIFFS OF EVERY ENVIRONMENT BEFORE AND AFTER REBUILD TO SEE WHICH NEEDS UPDATING

            for(const auto &env : tensors.get_edges().eneL) {
                msg1.emplace_back(fmt::format("eneL[{:2}]: {} norm {:.8e}", env->get_position(), env->get_unique_id(), fp(tenx::norm(env->get_block()))));
            }
            for(const auto &env : tensors.get_edges().eneR) {
                msg1.emplace_back(fmt::format("eneR[{:2}]: {} norm {:.8e}", env->get_position(), env->get_unique_id(), fp(tenx::norm(env->get_block()))));
            }

            tensors.rebuild_edges();
            tensors.clear_cache();
            tensors.clear_measurements();

            const auto &mps  = tensors.get_state().template get_multisite_mps<Scalar>();
            const auto &mpo1 = tensors.get_model().template get_multisite_mpo<Scalar>();
            const auto &mpo2 = tensors.get_model().template get_multisite_mpo_squared<Scalar>();
            const auto &enve = tensors.get_edges().get_multisite_env_ene();
            const auto &envv = tensors.get_edges().get_multisite_env_var();

            auto H1t   = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H1tx2 = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H1tQ  = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H2t   = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H2tx2 = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto H2tQ  = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            {
                auto h1info = SetH1MvInfo(ContractionBackend::TBLIS, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionBackend::TBLIS, mpo2.dimensions());
                tools::log->info("contracting with tblis");
                tools::common::contraction::matrix_vector_product(H1t, mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
                tools::common::contraction::matrix_vector_product(H2t, mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());
            }
            {
                auto h1info = SetH1MvInfo(ContractionBackend::X2, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionBackend::X2, mpo2.dimensions());
                tools::log->info("contracting with x2");
                tools::common::contraction::matrix_vector_product(H1tx2, mps, mpo1, enve.L, enve.R);
                tools::common::contraction::matrix_vector_product(H2tx2, mps, mpo2, envv.L, envv.R);
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
            using VectorType     = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            auto       v         = tenx::VectorMap(mps);
            auto       H1v       = tenx::VectorMap(H1t);
            auto       H1vx2     = tenx::VectorMap(H1tx2);
            auto       H1vQ      = tenx::VectorMap(H1tQ);
            auto       H2v       = tenx::VectorMap(H2t);
            auto       H2vx2     = tenx::VectorMap(H2tx2);
            auto       H2vQ      = tenx::VectorMap(H2tQ);
            RealScalar vH1_H1v   = std::real(H1v.dot(H1v));
            RealScalar v_H1v     = std::real(v.dot(H1v));
            RealScalar v_H1vx2   = std::real(v.dot(H1vx2));
            RealScalar v_H1vQ    = std::real(v.dot(H1vQ));
            RealScalar v_H2v     = std::real(v.dot(H2v));
            RealScalar v_H2vx2   = std::real(v.dot(H2vx2));
            RealScalar v_H2vQ    = std::real(v.dot(H2vQ));
            VectorType resid1    = H1v - v_H1v * v;
            RealScalar rnorm1    = resid1.norm();
            RealScalar proj_res2 = std::real(resid1.dot(resid1));
            RealScalar leak2_raw = v_H2v - vH1_H1v;
            RealScalar res2_est  = proj_res2 + leak2_raw;
            RealScalar delta     = v_H2v - vH1_H1v;

            tools::log->info("var_opt             = {:.16e}", fp(opt_state.get_variance()));
            tools::log->info("var_mrg             = {:.16e}", fp(var_mrg));
            tools::log->info("vH1_H1v             = {:.16e}", fp(vH1_H1v));
            tools::log->info("v_H1v               = {:.16e} diff {:.16e}", fp(v_H1v), fp(v_H1v - std::sqrt(vH1_H1v)));
            tools::log->info("v_H1v X2            = {:.16e}", fp(v_H1vx2));
            tools::log->info("v_H1v Q             = {:.16e}", fp(v_H1vQ));
            tools::log->info("v_H1v²              = {:.16e} diff {:.16e}", fp(v_H1v * v_H1v), fp(v_H1v * v_H1v - vH1_H1v));
            tools::log->info("v_H2v               = {:.16e}", fp(v_H2v));
            tools::log->info("v_H2v X2            = {:.16e}", fp(v_H2vx2));
            tools::log->info("v_H2v Q             = {:.16e}", fp(v_H2vQ));
            tools::log->info("|H1v-vH1v*v|        = {:.16e}", fp(rnorm1));
            tools::log->info("delta               = {:.16e}", fp(delta));
            tools::log->info("E_local est         = {:.16e}", fp(std::sqrt(vH1_H1v + rnorm1)));
            tools::log->info("sqrt(|H1v-v_H1v*v|) = {:.16e}", fp(std::sqrt(rnorm1)));
            tools::log->info("res2_est            = {:.16e}", fp(res2_est));
            tools::log->info("energy variance     = {:.16e}", fp(v_H2v - v_H1v * v_H1v));
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
            const auto &mps   = tensors.get_state().template get_multisite_mps<Scalar>();
            const auto &mpo1  = tensors.get_model().template get_multisite_mpo<Scalar>();
            const auto &mpo2  = tensors.get_model().template get_multisite_mpo_squared<Scalar>();
            const auto &enve  = tensors.get_edges().get_multisite_env_ene();
            const auto &envv  = tensors.get_edges().get_multisite_env_var();
            auto        H1t   = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H1tx2 = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H1tQ  = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H2t   = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H2tx2 = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            auto        H2tQ  = Eigen::Tensor<Scalar, 3>(mps.dimensions());
            {
                auto h1info = SetH1MvInfo(ContractionBackend::TBLIS, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionBackend::TBLIS, mpo2.dimensions());
                tools::log->info("contracting with tblis");
                tools::common::contraction::matrix_vector_product(H1t, mps, mpo1, enve.L.get_blkx2(), enve.R.get_blkx2());
                tools::common::contraction::matrix_vector_product(H2t, mps, mpo2, envv.L.get_blkx2(), envv.R.get_blkx2());
            }
            {
                auto h1info = SetH1MvInfo(ContractionBackend::X2, mpo1.dimensions());
                auto h2info = SetH2MvInfo(ContractionBackend::X2, mpo2.dimensions());
                tools::log->info("contracting with x2");
                tools::common::contraction::matrix_vector_product(H1tx2, mps, mpo1, enve.L, enve.R);
                tools::common::contraction::matrix_vector_product(H2tx2, mps, mpo2, envv.L, envv.R);
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
            using VectorType     = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            auto       v         = tenx::VectorMap(mps);
            auto       H1v       = tenx::VectorMap(H1t);
            auto       H1vx2     = tenx::VectorMap(H1tx2);
            auto       H1vQ      = tenx::VectorMap(H1tQ);
            auto       H2v       = tenx::VectorMap(H2t);
            auto       H2vx2     = tenx::VectorMap(H2tx2);
            auto       H2vQ      = tenx::VectorMap(H2tQ);
            RealScalar vH1_H1v   = std::real(H1v.dot(H1v));
            RealScalar v_H1v     = std::real(v.dot(H1v));
            RealScalar v_H1vx2   = std::real(v.dot(H1vx2));
            RealScalar v_H1vQ    = std::real(v.dot(H1vQ));
            RealScalar v_H2v     = std::real(v.dot(H2v));
            RealScalar v_H2vx2   = std::real(v.dot(H2vx2));
            RealScalar v_H2vQ    = std::real(v.dot(H2vQ));
            VectorType resid1    = H1v - v_H1v * v;
            RealScalar rnorm1    = resid1.norm();
            RealScalar proj_res2 = std::real(resid1.dot(resid1));
            RealScalar leak2_raw = v_H2v - vH1_H1v;
            RealScalar res2_est  = proj_res2 + leak2_raw;
            RealScalar delta     = v_H2v - vH1_H1v;

            tools::log->info("var_opt             = {:.16e}", fp(opt_state.get_variance()));
            tools::log->info("var_mrg             = {:.16e}", fp(var_mrg));
            tools::log->info("vH1_H1v             = {:.16e}", fp(vH1_H1v));
            tools::log->info("v_H1v               = {:.16e} diff {:.16e}", fp(v_H1v), fp(v_H1v - std::sqrt(vH1_H1v)));
            tools::log->info("v_H1v X2            = {:.16e}", fp(v_H1vx2));
            tools::log->info("v_H1v Q             = {:.16e}", fp(v_H1vQ));
            tools::log->info("v_H1v²              = {:.16e} diff {:.16e}", fp(v_H1v * v_H1v), fp(v_H1v * v_H1v - vH1_H1v));
            tools::log->info("v_H2v               = {:.16e}", fp(v_H2v));
            tools::log->info("v_H2v X2            = {:.16e}", fp(v_H2vx2));
            tools::log->info("v_H2v Q             = {:.16e}", fp(v_H2vQ));
            tools::log->info("|H1v-vH1v*v|        = {:.16e}", fp(rnorm1));
            tools::log->info("delta               = {:.16e}", fp(delta));
            tools::log->info("E_local est         = {:.16e}", fp(std::sqrt(vH1_H1v + rnorm1)));
            tools::log->info("sqrt(|H1v-v_H1v*v|) = {:.16e}", fp(std::sqrt(rnorm1)));
            tools::log->info("res2_est            = {:.16e}", fp(res2_est));
            tools::log->info("energy variance     = {:.16e}", fp(v_H2v - v_H1v * v_H1v));
        }
        for(size_t i = 0; i < msg1.size(); ++i) { tools::log->info("{} | {}", msg1.at(i), msg2.at(i)); }

        exit(1);
    }

    status.energy_variance_lowest = std::min(static_cast<double>(var_mrg), status.energy_variance_lowest);
    var_delta                     = var_mrg - var_latest;
    ene_delta                     = ene_mrg - ene_latest;
    var_latest                    = var_mrg;
    ene_latest                    = ene_mrg;
    auto bondexp_result           = expand_bonds(BondExpansionOrder::POSTOPT);

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
    tools::log->trace("Energy   change Δsvd/Δopt: {:.16f} | ini {:.16f} opt {:.16f} exp {:.16f}", fp(ene_delta_svd / ene_delta_opt), fp(ene_ini), fp(ene_opt),
                      fp(ene_exp));
    tools::log->trace("Variance change Δsvd/Δopt: {:.16f} | ini {:.16f} opt {:.16f} exp {:.16f}", fp(var_delta_svd / var_delta_opt), fp(var_ini), fp(var_opt),
                      fp(var_exp));

    last_optsolver = opt_state.get_optsolver();
    last_optalgo   = opt_state.get_optalgo();

    if constexpr(settings::debug) {
        if(tools::log->level() <= spdlog::level::trace) tools::log->trace("Truncation errors: {::8.3e}", tensors.state->get_truncation_errors_active());
        if(tools::log->level() <= spdlog::level::trace) tools::log->trace("Truncation errors: {::8.3e}", tensors.state->get_truncation_errors());
        tools::log->debug("Before update            : variance {:8.2e} | mps dims {}", fp(initial_state.get_variance()),
                          initial_state.get_tensor().dimensions());
        tools::log->debug("After  optimization      : variance {:8.2e} | mps dims {}", fp(opt_state.get_variance()), opt_state.get_tensor().dimensions());
        tools::log->debug("After  merge             : variance {:8.2e} | mps dims {}", fp(var_mrg), tensors.get_state().get_bond_dims_active());
        tools::log->debug("After  bond expansion    : variance {:8.2e} | mps dims {}", fp(var_exp), bondexp_result.dimMP);
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
void xdmrg<Scalar>::set_energy_shift_mpo() {
    // In xdmrg we find an excited energy eigenstate by optimizing the energy variance of some state close to a target energy.
    // We can target a particular energy by setting an energy shift (equal to the target energy), which then becomes the energy minimum
    // once we fold the spectrum by squaring the Hamiltonian (i.e. we optimize (H-E_tgt)²):
    //      Var H = <(H-E_tgt)²> - <H-E_tgt>²     = <H²> - 2<H>E_tgt + E_tgt² - (<H> - E_tgt)²
    //                                            =  H²  - 2*E*E_tgt + E_tgt² - E² + 2*E*E_tgt - E_tgt²
    //                                            =  H²  - E²
    // The first term <(H-E_tgt)²> is computed using a double-layer of mpos with energy shifted by E_tgt.
    // If we didn't shift mpo's, the last line, H²-E² is subtraction of two large numbers --> catastrophic cancellation --> loss of precision.
    // However, by minimizing the variance of shifted mpos:
    //              Var H = <(H-E_tgt)²> - <H-E_tgt>² = <(H-E_tgt)²> - (E-E_tgt)²
    // we get the subtraction of two very small terms since E-E_shf should be small.

    if(not tensors.position_is_inward_edge()) return;
    constexpr auto eps = std::numeric_limits<RealScalar>::epsilon();
    if(var_latest < eps * 10) return; // No need to improve precision further.

    if(settings::precision::use_energy_shifted_mpo) {
        auto energy_shift = tools::finite::measure::energy(tensors);
        tensors.set_energy_shift_mpo(energy_shift);
    } else {
        Scalar energy_shift = narrow_cast<Scalar>(std::real(status.energy_tgt));
        tensors.set_energy_shift_mpo(energy_shift);
    }
}

template<typename Scalar>
void xdmrg<Scalar>::update_time_step() {
    tools::log->trace("Updating time step");
    status.delta_t = std::complex<double>(1e-6, 0);
}
