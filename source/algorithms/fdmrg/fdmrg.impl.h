#pragma once
#include "../fdmrg.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "io/fmt_custom.h"
#include "math/cast.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/common/prof.h"
#include "tools/finite/env/BondExpansionConfig.h"
#include "tools/finite/env/BondExpansionResult.h"
#include "tools/finite/h5.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/mps.h"
#include "tools/finite/multisite.h"
#include "tools/finite/opt.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
#include "tools/finite/print.h"

template<typename Scalar>
fdmrg<Scalar>::fdmrg() : AlgorithmFinite<Scalar>(settings::xdmrg::ritz, AlgorithmType::fDMRG) {
    tools::log->trace("Constructing class_fdmrg (without a file)");
}

template<typename Scalar>
fdmrg<Scalar>::fdmrg(std::shared_ptr<h5pp::File> h5file_) : AlgorithmFinite<Scalar>(std::move(h5file_), settings::fdmrg::ritz, AlgorithmType::fDMRG) {
    tools::log->trace("Constructing class_fdmrg");
}

template<typename Scalar>
std::string_view fdmrg<Scalar>::get_state_name() const {
    if(status.opt_ritz == OptRitz::SR)
        return "state_emin";
    else
        return "state_emax";
}

template<typename Scalar>
void fdmrg<Scalar>::resume() {
    // Resume can imply many things
    // 1) Resume a simulation which terminated prematurely
    // 2) Resume a previously successful simulation. This may be desireable if the config
    //    wants something that is not present in the file.
    //      a) A certain number of states
    //      b) A state inside a particular energy window
    //      c) The ground or "roof" states
    // To guide the behavior, we check the setting ResumePolicy.

    auto state_prefixes = tools::common::h5::resume::find_state_prefixes(*h5file, status.algo_type, "state_");
    if(state_prefixes.empty()) throw except::state_error("no resumable states were found");
    for(const auto &state_prefix : state_prefixes) {
        tools::log->info("Resuming state [{}]", state_prefix);
        try {
            tools::finite::h5::load::simulation(*h5file, state_prefix, tensors, status, status.algo_type);
        } catch(const except::load_error &le) { continue; }

        // Our first task is to decide on a state name for the newly loaded state
        // The simplest is to inferr it from the state prefix itself
        auto name = tools::common::h5::resume::extract_state_name(state_prefix);

        // Reload the bond and truncation error limits (could be different in the config compared to the status we just loaded)
        // Reload the bond and truncation error limits (could be different in the config compared to the status we just loaded)
        double long_max                   = static_cast<double>(std::numeric_limits<long>::max());
        double bond_max                   = std::min(long_max, std::pow(2.0, settings::model::model_size / 2));
        status.bond_max                   = std::min(status.bond_max, safe_cast<long>(bond_max));
        status.bond_min                   = std::max(status.bond_min, settings::get_bond_min(status.algo_type));
        status.bond_lim                   = std::min(status.bond_lim, status.bond_max);
        status.bond_limit_has_reached_max = false;

        status.trnc_min                   = settings::precision::svd_truncation_min;
        status.trnc_max                   = settings::precision::svd_truncation_max;
        status.trnc_limit_has_reached_min = false;

        // Apply shifts and compress the model
        tensors.move_center_point_to_inward_edge();
        set_parity_shift_mpo();
        set_parity_shift_mpo_squared();
        set_energy_shift_mpo();
        rebuild_tensors(); // Rebuilds and compresses mpos, then rebuilds the environments
        update_precision_limit();
        update_dmrg_blocksize();

        // Initialize a custom task list
        std::deque<fdmrg_task> task_list;

        if(status.algorithm_has_succeeded)
            task_list = {fdmrg_task::POST_PRINT_RESULT};
        else {
            task_list.emplace_back(fdmrg_task::INIT_CLEAR_CONVERGENCE);
            // This could be a savepoint state
            // Simply "continue" the algorithm until convergence
            if(name.find("emax") != std::string::npos)
                task_list.emplace_back(fdmrg_task::FIND_HIGHEST_STATE);
            else if(name.find("emin") != std::string::npos)
                task_list.emplace_back(fdmrg_task::FIND_GROUND_STATE);
            else
                throw except::runtime_error("Unrecognized state name for fdmrg: [{}]", name);
            task_list.emplace_back(fdmrg_task::POST_DEFAULT);
        }
        run_task_list(task_list);
    }
}

template<typename Scalar>
void fdmrg<Scalar>::run_task_list(std::deque<fdmrg_task> &task_list) {
    while(not task_list.empty()) {
        auto task = task_list.front();
        switch(task) {
            case fdmrg_task::INIT_RANDOMIZE_MODEL: initialize_model(); break;
            case fdmrg_task::INIT_RANDOMIZE_INTO_PRODUCT_STATE: initialize_state(ResetReason::INIT, StateInit::RANDOM_PRODUCT_STATE); break;
            case fdmrg_task::INIT_RANDOMIZE_INTO_ENTANGLED_STATE: initialize_state(ResetReason::INIT, StateInit::RANDOM_ENTANGLED_STATE); break;
            case fdmrg_task::INIT_BOND_LIMITS: init_bond_dimension_limits(); break;
            case fdmrg_task::INIT_TRNC_LIMITS: init_truncation_error_limits(); break;
            case fdmrg_task::INIT_WRITE_MODEL: write_to_file(StorageEvent::MODEL); break;
            case fdmrg_task::INIT_CLEAR_STATUS: status.clear(); break;
            case fdmrg_task::INIT_CLEAR_CONVERGENCE: clear_convergence_status(); break;
            case fdmrg_task::INIT_DEFAULT: run_preprocessing(); break;
            case fdmrg_task::FIND_GROUND_STATE:
                status.opt_ritz = OptRitz::SR;
                tensors.state->set_name(get_state_name());
                run_algorithm();
                break;
            case fdmrg_task::FIND_HIGHEST_STATE:
                status.opt_ritz = OptRitz::LR;
                tensors.state->set_name(get_state_name());
                run_algorithm();
                break;
            case fdmrg_task::POST_WRITE_RESULT: write_to_file(StorageEvent::FINISHED, CopyPolicy::FORCE); break;
            case fdmrg_task::POST_PRINT_RESULT: print_status_full(); break;
            case fdmrg_task::POST_PRINT_TIMERS: tools::common::timer::print_timers(); break;
            case fdmrg_task::POST_RBDS_ANALYSIS: run_rbds_analysis(); break;
            case fdmrg_task::POST_RTES_ANALYSIS: run_rtes_analysis(); break;
            case fdmrg_task::POST_DEFAULT: run_postprocessing(); break;
            case fdmrg_task::TIMER_RESET: tid::reset("fDMRG"); break;
        }
        task_list.pop_front();
    }
}

template<typename Scalar>
void fdmrg<Scalar>::run_default_task_list() {
    fdmrg_task fdmrg_task_find_state_ritz;
    switch(settings::fdmrg::ritz) {
        case OptRitz::SR: fdmrg_task_find_state_ritz = fdmrg_task::FIND_GROUND_STATE; break;
        case OptRitz::LR: fdmrg_task_find_state_ritz = fdmrg_task::FIND_HIGHEST_STATE; break;
        default: throw except::logic_error("fdmrg expects ritz SR or LR. Got: {}", enum2sv(settings::fdmrg::ritz));
    }

    std::deque<fdmrg_task> default_task_list = {
        fdmrg_task::INIT_DEFAULT,
        fdmrg_task_find_state_ritz,
        fdmrg_task::POST_DEFAULT,
    };

    run_task_list(default_task_list);
    if(not default_task_list.empty()) {
        for(auto &task : default_task_list) tools::log->critical("Unfinished task: {}", enum2sv(task));
        throw except::runtime_error("Simulation ended with unfinished tasks");
    }
}

template<typename Scalar>
void fdmrg<Scalar>::run_preprocessing() {
    tools::log->info("Running {} preprocessing", status.algo_type_sv());
    auto t_pre = tid::tic_scope("pre");
    status.clear();
    if(tensors.state->get_name().empty()) tensors.state->set_name(get_state_name());
    initialize_model(); // First use of random!
    init_bond_dimension_limits();
    init_truncation_error_limits();
    initialize_state(ResetReason::INIT, settings::strategy::initial_state);
    set_parity_shift_mpo();
    set_parity_shift_mpo_squared();
    set_energy_shift_mpo();
    rebuild_tensors(); // Rebuilds and compresses mpos, then rebuilds the environments
    update_precision_limit();
    tools::log->info("Finished {} preprocessing", status.algo_type_sv());
}

template<typename Scalar>
void fdmrg<Scalar>::run_algorithm() {
    if(tensors.state->get_name().empty()) tensors.state->set_name(get_state_name());
    tools::log->info("Starting {} algorithm with model [{}] for state [{}]", status.algo_type_sv(), enum2sv(settings::model::model_type),
                     tensors.state->get_name());
    auto t_run       = tid::tic_scope("run");
    status.algo_stop = AlgorithmStop::NONE;
    while(true) {
        update_state();
        print_status();
        check_convergence();
        write_to_file();

        tools::log->trace("Finished step {}, iter {}, pos {}, dir {}", status.step, status.iter, status.position, status.direction);

        // It's important not to perform the last move, so we break now: that last state would not get optimized
        if(status.algo_stop != AlgorithmStop::NONE) break;
        update_bond_dimension_limit();   // Will update bond dimension if the state precision is being limited by bond dimension
        update_truncation_error_limit(); // Will update truncation error limit if the state is being truncated
        update_dmrg_blocksize();
        try_projection();
        set_energy_shift_mpo(); // Shift the energy in the mpos to get rid of critical cancellation (shifts by the current energy)
        rebuild_tensors();
        move_center_point();
        status.wall_time = tid::get_unscoped("t_tot").get_time();
        status.algo_time = t_run->get_time();
    }
    tools::log->info("Finished {} simulation of state [{}] -- stop reason: {}", status.algo_type_sv(), tensors.state->get_name(), status.algo_stop_sv());
    status.algorithm_has_finished = true;
    if(settings::fdmrg::store_wavefn and tensors.template get_length<long>() <= 16) {
#pragma message "Save fdmrg wavevector properly"
        Eigen::Tensor<RealScalar, 1> psi = tools::finite::mps::mps2tensor<Scalar>(tensors.get_state()).real();
        this->write_tensor_to_file(psi, "psi", StorageEvent::FINISHED);
    }
}

template<typename Scalar>
void fdmrg<Scalar>::update_state() {
    auto t_step                = tid::tic_scope("step");
    auto bondexp_preopt_result = expand_bonds(BondExpansionOrder::PREOPT);
    auto opt_meta              = get_opt_meta();
    variance_before_step       = std::nullopt;

    tools::log->debug("Starting {} iter {} | step {} | pos {} | dir {} | ritz {} | type {}", status.algo_type_sv(), status.iter, status.step, status.position,
                      status.direction, enum2sv(opt_meta.optRitz), enum2sv(opt_meta.optType));
    // Try activating the sites asked for;
    tensors.activate_sites(opt_meta.chosen_sites);
    if(tensors.active_sites.empty()) {
        tools::log->warn("Failed to activate sites");
        return;
    }
    tensors.rebuild_edges();

    // Hold the variance before the optimization step for comparison
    if(not variance_before_step) variance_before_step = tools::finite::measure::energy_variance(tensors); // Should just take value from cache

    auto initial_state = tools::finite::opt::get_opt_initial_mps(tensors, opt_meta);
    auto opt_state     = tools::finite::opt::find_ground_state(tensors, initial_state, status, opt_meta);

    // Determine the quality of the optimized state.
    opt_state.set_relchange(opt_state.get_variance() / variance_before_step.value());
    opt_state.set_bond_limit(opt_meta.svd_cfg->rank_max.value());
    opt_state.set_trnc_limit(opt_meta.svd_cfg->truncation_limit.value());
    /* clang-format off */
    opt_meta.optExit = OptExit::SUCCESS;
    if(opt_state.get_grad_max()       > static_cast<RealScalar>(1.000)                            ) opt_meta.optExit |= OptExit::FAIL_GRADIENT;
    if(opt_state.get_eigs_rnorm()     > static_cast<RealScalar>(settings::precision::eigs_tol_max)) opt_meta.optExit |= OptExit::FAIL_RESIDUAL;
    if(opt_state.get_eigs_nev()       == 0 and
       opt_meta.optSolver             == OptSolver::EIGS                                          ) opt_meta.optExit |= OptExit::FAIL_RESIDUAL; // No convergence
    if(opt_state.get_overlap()        < static_cast<RealScalar>(0.010)                            ) opt_meta.optExit |= OptExit::FAIL_OVERLAP;
    if(opt_state.get_relchange()      > static_cast<RealScalar>(1.001)                            ) opt_meta.optExit |= OptExit::FAIL_WORSENED;
    else if(opt_state.get_relchange() > static_cast<RealScalar>(0.999)                            ) opt_meta.optExit |= OptExit::FAIL_NOCHANGE;
    /* clang-format on */
    opt_state.set_optexit(opt_meta.optExit);

    tools::log->trace("Optimization [{}]: {}. Variance change {:8.2e} --> {:8.2e} ({:.3f} %)", enum2sv(opt_meta.optSolver), flag2str(opt_meta.optExit),
                      fp(variance_before_step.value()), fp(opt_state.get_variance()), fp(opt_state.get_relchange() * 100));
    if(opt_state.get_relchange() > static_cast<RealScalar>(1000)) tools::log->warn("Variance increase by x {:.2e}", fp(opt_state.get_relchange()));

    if(tools::log->level() <= spdlog::level::debug) {
        tools::log->debug("Optimization result: {:<24} | E {:<20.16f}| σ²H {:<8.2e} | rnorm {:8.2e} | overlap {:.16f} | "
                          "sites {} |"
                          "{} | {} | time {:.2e} s",
                          opt_state.get_name(), fp(opt_state.get_energy()), fp(opt_state.get_variance()), fp(opt_state.get_eigs_rnorm()),
                          fp(opt_state.get_overlap()), opt_state.get_sites(), enum2sv(opt_state.get_optsolver()), flag2str(opt_state.get_optexit()),
                          opt_state.get_time());
    }
    last_optsolver = opt_state.get_optsolver();
    tensors.state->tag_active_sites_normalized(false);

    // Do the truncation with SVD
    // TODO: We may need to detect here whether the truncation error limit needs lowering due to a variance increase in the svd merge
    auto logPolicy = LogPolicy::SILENT;
    if constexpr(settings::debug) logPolicy = LogPolicy::VERBOSE;
    tensors.merge_multisite_mps(opt_state.get_tensor(), MergeEvent::OPT, opt_meta.svd_cfg, logPolicy);
    tensors.rebuild_edges(); // This will only do work if edges were modified, which is the case in 1-site dmrg.
    if constexpr(settings::debug) {
        if(tools::log->level() <= spdlog::level::trace) tools::log->trace("Truncation errors: {::8.3e}", tensors.state->get_truncation_errors_active());
    }

    if constexpr(settings::debug) {
        auto variance_before_svd = opt_state.get_variance();
        auto variance_after_svd  = tools::finite::measure::energy_variance(tensors);
        tools::log->debug("Variance check before SVD: {:8.2e}", fp(variance_before_svd));
        tools::log->debug("Variance check after  SVD: {:8.2e}", fp(variance_after_svd));
        tools::log->debug("Variance change from  SVD: {:.16f}%", fp(100 * variance_after_svd / variance_before_svd));
    }

    tools::log->trace("Updating variance record holder");
    auto ene_mrg                  = tools::finite::measure::energy(tensors);
    auto var_mrg                  = tools::finite::measure::energy_variance(tensors);
    status.energy_variance_lowest = std::min(static_cast<double>(var_mrg), status.energy_variance_lowest);
    var_delta                     = var_mrg - var_latest;
    ene_delta                     = ene_mrg - ene_latest;
    var_latest                    = var_mrg;
    ene_latest                    = ene_mrg;

    auto bondexp_postopt_result = expand_bonds(BondExpansionOrder::POSTOPT);

    auto ene_ini = initial_state.get_energy();
    auto ene_opt = opt_state.get_energy();
    auto ene_exp = bondexp_postopt_result.ene_new;
    auto var_ini = std::abs(initial_state.get_variance());
    auto var_opt = std::abs(opt_state.get_variance());
    auto var_exp = bondexp_postopt_result.var_new;
    ;
    var_delta_opt = ene_opt - ene_ini;
    var_delta_svd = ene_exp - ene_opt;
    var_delta_opt = var_opt - var_ini;
    var_delta_svd = var_exp - var_opt;
    std_delta_opt = std::sqrt(var_opt) - std::sqrt(var_ini);
    std_delta_svd = std::sqrt(var_exp) - std::sqrt(var_opt);
    tools::log->trace("Energy   change Δsvd/Δopt: {:.16f}", fp(ene_delta_svd / ene_delta_opt));
    tools::log->trace("Variance change Δsvd/Δopt: {:.16f}", fp(var_delta_svd / var_delta_opt));
    tools::log->trace("Std.dev. change Δsvd/Δopt: {:.16f}", fp(std_delta_svd / std_delta_opt));

    last_optsolver = opt_state.get_optsolver();
    last_optalgo   = opt_state.get_optalgo();

    if constexpr(settings::debug) {
        if(tools::log->level() <= spdlog::level::trace) tools::log->trace("Truncation errors: {::8.3e}", tensors.state->get_truncation_errors_active());
        tools::log->debug("Before update            : variance {:8.2e} | mps dims {}", fp(initial_state.get_variance()),
                          initial_state.get_tensor().dimensions());
        tools::log->debug("After  optimization      : variance {:8.2e} | mps dims {}", fp(opt_state.get_variance()), opt_state.get_tensor().dimensions());
        tools::log->debug("After  merge             : variance {:8.2e} | mps dims {}", fp(var_mrg), tensors.get_state().get_bond_dims_active());
        tools::log->debug("After  bond expansion    : variance {:8.2e} | mps dims {}", fp(var_exp), bondexp_postopt_result.dimMP);
    }

    if constexpr(settings::debug) tensors.assert_validity();
}
