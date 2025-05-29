#pragma once
#include "../AlgorithmInfinite.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "math/cast.h"
#include "math/num.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/state/StateInfinite.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/infinite/h5.h"
#include "tools/infinite/measure.h"


template<typename Scalar>
AlgorithmInfinite<Scalar>::AlgorithmInfinite(std::shared_ptr<h5pp::File> h5ppFile_, OptRitz opt_ritz_, AlgorithmType algo_type)
    : AlgorithmBase(std::move(h5ppFile_), opt_ritz_, algo_type) {
    tools::log->trace("Constructing algorithm infinite");
    tensors.initialize(settings::model::model_type);
    tensors.state->set_algorithm(status.algo_type);
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::run() {
    if(not settings::algorithm_is_on(status.algo_type)) return;
    auto t_tot = tid::get_unscoped("t_tot", tid::level::normal).tic_token();
    run_preprocessing();
    run_algorithm();
    run_postprocessing();
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::run_preprocessing() {
    status.clear();
    initialize_model(); // First use of random!
    init_bond_dimension_limits();
    write_to_file(StorageEvent::MODEL);
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::run_postprocessing() {
    auto t_pos = tid::tic_scope("post");
    write_to_file(StorageEvent::FINISHED);
    copy_from_tmp(StorageEvent::FINISHED);
    print_status_full();
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::initialize_model() {
    tools::log->info("Initializing model");
    tensors.initialize_model();
    clear_convergence_status();
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::update_precision_limit(std::optional<double> energy_upper_bound) {
    if(not energy_upper_bound)
        energy_upper_bound = static_cast<double>(tools::infinite::measure::energy_per_site_mpo(tensors)) * static_cast<double>(status.iter);
    // The variance precision limit depends on the Hamiltonian operator norm ~ the largest eigenvalue.
    // We can get a rough order of magnitude estimate the largest eigenvalue by adding the absolute value of all the
    // Hamiltonian couplings and fields.
    double energy_abs                 = std::abs(energy_upper_bound.value());
    double digits10                   = std::numeric_limits<double>::digits10;
    double energy_exp                 = std::ceil(std::max(0.0, std::log10(energy_abs)));
    double max_digits                 = std::floor(std::max(0.0, digits10 - energy_exp));
    status.energy_variance_max_digits = safe_cast<size_t>(max_digits);
    status.energy_variance_prec_limit = std::pow(10.0, -max_digits);
    tools::log->info("Estimated limit on energy variance precision: {:.3e}", status.energy_variance_prec_limit);
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::update_bond_dimension_limit() {
    status.bond_limit_has_reached_max = status.bond_lim >= status.bond_max;
    if(settings::strategy::bond_increase_when == UpdatePolicy::NEVER) {
        status.bond_lim = status.bond_max;
        return;
    }
    if(status.bond_limit_has_reached_max) return;
    auto tic = tid::tic_scope("bond_grow");

    // If we got here we want to increase the bond dimension limit progressively during the simulation
    bool is_saturated      = status.algorithm_saturated_for > 1; // Allow one round while saturated so that extra efforts get a chance.
    bool is_has_stuck      = status.algorithm_has_stuck_for > 1;
    bool is_truncated      = tensors.state->is_limited_by_bond(status.bond_lim) or tensors.state->is_truncated(status.trnc_lim);
    bool grow_if_truncated = settings::strategy::bond_increase_when == UpdatePolicy::TRUNCATED;
    bool grow_if_saturated = settings::strategy::bond_increase_when == UpdatePolicy::SAT_ALGO;
    bool grow_if_has_stuck = settings::strategy::bond_increase_when == UpdatePolicy::STK_ALGO;

    if(grow_if_truncated and not is_truncated) {
        tools::log->info("State is not limited by its bond dimension. Kept current bond limit {}", status.bond_lim);
        return;
    }
    if(grow_if_saturated and not is_saturated) {
        tools::log->info("Algorithm is not saturated. Kept current bond limit {}", status.bond_lim);
        return;
    }
    if(grow_if_has_stuck and not is_has_stuck) {
        tools::log->info("Algorithm is not stuck. Kept current bond limit {}", status.bond_lim);
        return;
    }

    // If we got to this point we will update the bond dimension by a factor
    auto grow_rate = settings::strategy::bond_increase_rate;
    if(grow_rate <= 1.0) throw except::runtime_error("Error: bond_increase_rate == {:.3f} | must be larger than one", grow_rate);

    // Write current results before updating bond dimension
    write_to_file(StorageEvent::BOND_UPDATE);

    // If we got to this point we will update the bond dimension by a factor
    auto factor = settings::strategy::bond_increase_rate;
    if(factor <= 1.0) throw except::logic_error("Error: bond_increase_rate == {:.3f} | must be larger than one", factor);

    auto bond_new = static_cast<double>(status.bond_lim);
    if(grow_rate <= 2.0 and grow_rate > 1.0) {
        bond_new = std::ceil(bond_new * grow_rate);
        bond_new = num::round_up_to_multiple_of<double>(bond_new, 4);
    } else if(grow_rate > 2.0) {
        bond_new = bond_new + grow_rate;
    } else
        throw except::logic_error("Expected grow_rate > 1.0. Got {}", grow_rate);
    bond_new = std::min(bond_new, static_cast<double>(status.bond_max));

    tools::log->info("Updating bond dimension limit {} -> {} | truncated {} | saturated {}", status.bond_lim, bond_new, is_truncated, is_saturated);
    status.bond_lim                   = safe_cast<long>(bond_new);
    status.bond_limit_has_reached_max = status.bond_lim == status.bond_max;
    status.algorithm_has_stuck_for    = 0;
    status.algorithm_saturated_for    = 0;
    // Last sanity check before leaving here
    if(status.bond_lim > status.bond_max) throw except::runtime_error("bond_lim is larger than get_bond_max! {} > {}", status.bond_lim, status.bond_max);
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::update_truncation_error_limit() {
    if(status.trnc_lim == 0.0) throw std::runtime_error("trnc_lim is zero!");
    status.trnc_min                   = settings::precision::svd_truncation_min;
    status.trnc_limit_has_reached_min = status.trnc_lim <= status.trnc_min;
    if(settings::strategy::trnc_decrease_when == UpdatePolicy::NEVER or settings::strategy::trnc_decrease_rate == 0.0) {
        status.trnc_lim                   = status.trnc_min;
        status.trnc_limit_has_reached_min = true;
        return;
    }
    if(status.trnc_limit_has_reached_min) return;
    auto tic = tid::tic_scope("trnc_down", tid::level::higher);

    // If we got here we want to decrease the truncation error limit progressively during the simulation
    bool is_saturated      = status.algorithm_saturated_for > 1; // Allow one round while saturated so that extra efforts get a chance.
    bool is_has_stuck      = status.algorithm_has_stuck_for > 1; // Allow one round while saturated so that extra efforts get a chance.
    bool is_truncated      = tensors.state->is_limited_by_bond(status.bond_lim) or tensors.state->is_truncated(status.trnc_lim);
    bool drop_if_truncated = settings::strategy::trnc_decrease_when == UpdatePolicy::TRUNCATED;
    bool drop_if_saturated = settings::strategy::trnc_decrease_when == UpdatePolicy::SAT_ALGO;
    bool drop_if_has_stuck = settings::strategy::trnc_decrease_when == UpdatePolicy::STK_ALGO;

    if(drop_if_truncated and not is_truncated) {
        tools::log->info("State is not truncated. Kept current truncation error limit {:8.2e}", status.trnc_lim);
        return;
    }
    if(drop_if_saturated and not is_saturated) {
        tools::log->info("Algorithm is not saturated. Kept current truncation error limit {:8.2e}", status.trnc_lim);
        return;
    }
    if(drop_if_has_stuck and not is_has_stuck) {
        tools::log->info("Algorithm is not stuck. Kept current truncation error limit {:8.2e}", status.trnc_lim);
        return;
    }

    // Write current results before updating the truncation error limit
    write_to_file(StorageEvent::TRNC_UPDATE);

    // If we got to this point we will update the truncation error limit by a factor
    auto rate = settings::strategy::trnc_decrease_rate;
    if(rate > 1.0 or rate < 0) throw except::runtime_error("Error: trnc_decrease_rate == {:8.2e} | must be in [0, 1]");

    auto trnc_new = std::max(status.trnc_min, status.trnc_lim * rate);

    tools::log->info("Updating truncation error limit {:8.2e} -> {:8.2e} | truncated {} | saturated {}", status.trnc_lim, trnc_new, is_truncated, is_saturated);
    status.trnc_lim                   = trnc_new;
    status.trnc_limit_has_reached_min = status.trnc_lim == status.trnc_min;
    status.algorithm_has_stuck_for    = 0;
    status.algorithm_saturated_for    = 0;

    // Last sanity check before leaving here
    if(status.trnc_lim < status.trnc_min) throw except::logic_error("trnc_lim is smaller than trnc_min ! {:8.2e} > {:8.2e}", status.trnc_lim, status.trnc_min);
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::initialize_state([[maybe_unused]] ResetReason reason, std::optional<std::string> sector, std::optional<bool> use_eigenspinors,
                                                 std::optional<std::string> pattern) {
    tools::log->trace("Initializing state");
    if(not sector) sector = settings::strategy::initial_axis;
    if(not pattern) pattern = settings::strategy::initial_pattern;
    if(not use_eigenspinors) use_eigenspinors = settings::strategy::use_eigenspinors;

    status.iter = 0;
    tensors.reset_to_random_product_state(sector.value(), use_eigenspinors.value(), pattern.value());
    // Randomize state
    clear_convergence_status();
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::clear_convergence_status() {
    tools::log->trace("Clearing saturation status");
    var_mpo_iter.clear();
    var_ham_iter.clear();
    var_mom_iter.clear();
    entropy_iter.clear();
    status.algo_stop                  = AlgorithmStop::NONE;
    status.algorithm_has_finished     = false;
    status.algorithm_has_succeeded    = false;
    status.algorithm_has_to_stop      = false;
    status.algorithm_has_stuck_for    = 0;
    status.algorithm_saturated_for    = 0;
    status.algorithm_converged_for    = 0;
    status.entanglement_converged_for = 0;
    status.entanglement_saturated_for = 0;
    status.variance_mpo_converged_for = 0;
    status.variance_mpo_saturated_for = 0;
    status.variance_ham_converged_for = 0;
    status.variance_ham_saturated_for = 0;
    status.variance_mom_converged_for = 0;
    status.variance_mom_saturated_for = 0;
    status.bond_limit_has_reached_max = false;
    status.trnc_limit_has_reached_min = false;
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::check_convergence_variance_mpo(std::optional<RealScalar> threshold, std::optional<RealScalar> sensitivity) {
    tools::log->debug("Checking convergence of variance mpo");
    if(not threshold) threshold = settings::precision::variance_convergence_threshold;
    if(not sensitivity) sensitivity = settings::precision::variance_saturation_sensitivity;
    var_mpo_iter.emplace_back(tools::infinite::measure::energy_variance_per_site_mpo(tensors));
    status.variance_mpo_converged_for = count_convergence(var_mpo_iter, threshold.value());
    auto report =
        check_saturation(var_mpo_iter, sensitivity.value(), SaturationPolicy::val | SaturationPolicy::mov | SaturationPolicy::min | SaturationPolicy::log);
    if(report.has_computed) { status.variance_mpo_saturated_for = report.saturated_count; }
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::check_convergence_variance_ham(std::optional<RealScalar> threshold, std::optional<RealScalar> sensitivity) {
    tools::log->trace("Checking convergence of variance ham");
    if(not threshold) threshold = settings::precision::variance_convergence_threshold;
    if(not sensitivity) sensitivity = settings::precision::variance_saturation_sensitivity;
    var_ham_iter.emplace_back(tools::infinite::measure::energy_variance_per_site_ham(tensors));
    status.variance_ham_converged_for = count_convergence(var_ham_iter, threshold.value());
    auto report =
        check_saturation(var_ham_iter, sensitivity.value(), SaturationPolicy::val | SaturationPolicy::mov | SaturationPolicy::min | SaturationPolicy::log);
    if(report.has_computed) { status.variance_ham_saturated_for = report.saturated_count; }
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::check_convergence_variance_mom(std::optional<RealScalar> threshold, std::optional<RealScalar> sensitivity) {
    tools::log->trace("Checking convergence of variance mom");
    if(not threshold) threshold = settings::precision::variance_convergence_threshold;
    if(not sensitivity) sensitivity = settings::precision::variance_saturation_sensitivity;
    var_mom_iter.emplace_back(tools::infinite::measure::energy_variance_per_site_mom(tensors));
    status.variance_mom_converged_for = count_convergence(var_mom_iter, threshold.value());
    auto report                       = check_saturation(var_mom_iter, sensitivity.value(),
                                                         SaturationPolicy::val | SaturationPolicy::min | SaturationPolicy::max | SaturationPolicy::mov | SaturationPolicy::log);
    if(report.has_computed) { status.variance_mom_saturated_for = report.saturated_count; }
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::check_convergence_entg_entropy(std::optional<RealScalar> sensitivity) {
    tools::log->debug("Checking convergence of entanglement");
    if(not sensitivity) sensitivity = settings::precision::entropy_saturation_sensitivity;
    entropy_iter.emplace_back(tools::infinite::measure::entanglement_entropy(tensors.get_state()));
    auto report = check_saturation(entropy_iter, sensitivity.value(), SaturationPolicy::val | SaturationPolicy::mov);
    if(report.has_computed) {
        status.entanglement_saturated_for = report.saturated_count;
        status.entanglement_converged_for = report.saturated_count;
    }
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::write_to_file(StorageEvent storage_event, CopyPolicy copy_policy) {
    status.event = storage_event;
    auto sinfo   = StorageInfo(status, tensors.state->get_name());
    tools::log->debug("Writing to file: Reason [{}] | state prefix [{}]", enum2sv(sinfo.storage_event), sinfo.get_state_prefix());
    // Start saving tensors and metadata
    tools::infinite::h5::save::bonds(*h5file, sinfo, tensors.get_state());
    tools::infinite::h5::save::state(*h5file, sinfo, tensors.get_state());
    tools::infinite::h5::save::edges(*h5file, sinfo, tensors.get_edges());
    if(storage_event == StorageEvent::PROJECTION) {
        status.event = StorageEvent::NONE;
        return; // Some storage reasons should not go further. Like projection.
    }

    // The main results have now been written. Next we append data to tables
    tools::infinite::h5::save::measurements(*h5file, sinfo, tensors, status);
    tools::common::h5::save::status(*h5file, sinfo, status);
    tools::common::h5::save::timer(*h5file, sinfo);
    tools::common::h5::save::memory(*h5file, sinfo);
    // Copy from temporary location to destination depending on a given policy
    copy_from_tmp(storage_event, copy_policy);
    status.event = StorageEvent::NONE;
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::print_status() {
    if(num::mod(status.iter, settings::print_freq(status.algo_type)) != 0) { return; }
    if(settings::print_freq(status.algo_type) == 0) { return; }
    //    compute_observables();
    std::string report;
    report += fmt::format("{:<} ", tensors.state->get_name());
    report += fmt::format("iter: {:<4} ", status.iter);
    report += fmt::format("step: {:<5} ", status.step);

    switch(status.algo_type) {
        case AlgorithmType::iDMRG:
            report += fmt::format("E/L: mpo {:<20.16f} ham {:<20.16f} mom {:<20.16f}",
                                  fp(tools::infinite::measure::energy_per_site_mpo(tensors)), //
                                  fp(tools::infinite::measure::energy_per_site_ham(tensors)), //
                                  fp(tools::infinite::measure::energy_per_site_mom(tensors)));
            break;
        case AlgorithmType::iTEBD:
            report += fmt::format("E/L: ham {:<20.16f} mom {:<20.16f}", fp(tools::infinite::measure::energy_per_site_ham(tensors)),
                                  fp(tools::infinite::measure::energy_per_site_mom(tensors)));
            break;
        default: throw except::runtime_error("Wrong simulation type");
    }

    switch(status.algo_type) {
        case AlgorithmType::iDMRG:
            report += fmt::format("lg σ²H: mpo {:<10.6f} ham {:<10.6f} mom {:<10.6f}",
                                  fp(tools::infinite::measure::energy_variance_per_site_mpo(tensors)), //
                                  fp(tools::infinite::measure::energy_variance_per_site_ham(tensors)), //
                                  fp(tools::infinite::measure::energy_variance_per_site_mom(tensors)));
            break;
        case AlgorithmType::iTEBD:
            report += fmt::format("lg σ²H: ham {:<10.6f} mom {:<10.6f}",
                                  fp(tools::infinite::measure::energy_variance_per_site_ham(tensors)), //
                                  fp(tools::infinite::measure::energy_variance_per_site_mom(tensors)));
            break;
        default: throw except::runtime_error("Wrong simulation type");
    }
    report += fmt::format("ε: {:<8.2e} ", tools::infinite::measure::truncation_error(tensors.get_state()));
    report += fmt::format("Sₑ(l): {:<10.8f} ", fp(tools::infinite::measure::entanglement_entropy(tensors.get_state())));
    report += fmt::format("χmax: {:<3} χlim: {:<3} χ: {:<3} ", status.bond_max, status.bond_lim, tools::infinite::measure::bond_dimension(tensors.get_state()));
    report += fmt::format("Sites: {:6}", tensors.get_length());

    report += fmt::format("stk: {:<1} ", status.algorithm_has_stuck_for);
    report += fmt::format("sat: [σ² {:<1} Sₑ {:<1}] ", status.variance_mpo_saturated_for, status.entanglement_saturated_for);
    switch(status.algo_type) {
        case AlgorithmType::iDMRG:
            report += fmt::format("sat: [σ² {:<1} Sₑ {:<1}] ", status.variance_mpo_saturated_for, status.entanglement_saturated_for);
            break;
        case AlgorithmType::iTEBD: report += fmt::format("sat: [Sₑ {:<1}] ", status.entanglement_saturated_for); break;
        default: throw except::runtime_error("Wrong simulation type");
    }
    report += fmt::format("con: {:<4} ", status.algorithm_converged_for);
    report += fmt::format("time:{:>8.2f}s ", tid::get_unscoped("t_tot").get_time());
    report += fmt::format("mem: [rss {:<.1f} peak {:<.1f} vm {:<.1f}] MB ", debug::mem_rss_in_mb(), debug::mem_hwm_in_mb(), debug::mem_vm_in_mb());
    tools::log->info(report);
}

template<typename Scalar>
void AlgorithmInfinite<Scalar>::print_status_full() {
    using namespace std;
    using namespace tools::infinite::measure;
    tools::log->info("{:=^60}", "");
    tools::log->info("= {: ^56} =", fmt::format("Completed [{}][{}]", status.algo_type_sv(), tensors.state->get_name()));
    tools::log->info("Iterations            = {:<16d}", status.iter);
    switch(status.algo_type) {
        case AlgorithmType::iDMRG:
            tools::log->info("Energy MPO            = {:<16.16f}", fp(tools::infinite::measure::energy_per_site_mpo(tensors)));
            tools::log->info("Energy HAM            = {:<16.16f}", fp(tools::infinite::measure::energy_per_site_ham(tensors)));
            tools::log->info("Energy MOM            = {:<16.16f}", fp(tools::infinite::measure::energy_per_site_mom(tensors)));
            break;
        case AlgorithmType::iTEBD:
            tools::log->info("Energy HAM            = {:<16.16f}", fp(tools::infinite::measure::energy_per_site_ham(tensors)));
            tools::log->info("Energy MOM            = {:<16.16f}", fp(tools::infinite::measure::energy_per_site_mom(tensors)));
            break;
        default: throw except::runtime_error("Wrong simulation type");
    }
    switch(status.algo_type) {
        case AlgorithmType::iDMRG:
            tools::log->info("lg σ²H MPO         = {:<8.2e}", fp(tools::infinite::measure::energy_variance_per_site_mpo(tensors)));
            tools::log->info("lg σ²H HAM         = {:<8.2e}", fp(tools::infinite::measure::energy_variance_per_site_ham(tensors)));
            tools::log->info("lg σ²H MOM         = {:<8.2e}", fp(tools::infinite::measure::energy_variance_per_site_mom(tensors)));
            break;
        case AlgorithmType::iTEBD:
            tools::log->info("lg σ²H HAM         = {:<8.2e}", fp(tools::infinite::measure::energy_variance_per_site_ham(tensors)));
            tools::log->info("lg σ²H MOM         = {:<8.2e}", fp(tools::infinite::measure::energy_variance_per_site_mom(tensors)));
            break;
        default: throw except::runtime_error("Wrong simulation type");
    }
    tools::log->info("Truncation error      = {:<8.2e}", tools::infinite::measure::truncation_error(tensors.get_state()));
    tools::log->info("Entanglement Entropy  = {:<16.16f}", fp(tools::infinite::measure::entanglement_entropy(tensors.get_state())));
    tools::log->info("χmax                  = {:<16d}", status.bond_max);
    tools::log->info("χ                     = {:<16d}", tools::infinite::measure::bond_dimension(tensors.get_state()));

    switch(status.algo_type) {
        case AlgorithmType::iDMRG: break;
        case AlgorithmType::iTEBD: tools::log->info("δt                    = {:<16.16f}", status.delta_t.to_floating_point<double>()); break;

        default: throw except::runtime_error("Wrong simulation type");
    }

    tools::log->info("Algorithm succeeded      = {:<}", status.algorithm_has_succeeded);
    tools::log->info("Algorithm saturated for  = {:<}", status.algorithm_saturated_for);
    tools::log->info("Algorithm converged for  = {:<}", status.algorithm_converged_for);
    tools::log->info("Algorithm has stuck for  = {:<}", status.algorithm_has_stuck_for);
    tools::log->info("σ² MPO                   = Converged : {:<4}  Saturated: {:<4}", status.variance_mpo_converged_for, status.variance_mpo_saturated_for);
    tools::log->info("σ² HAM                   = Converged : {:<4}  Saturated: {:<4}", status.variance_ham_converged_for, status.variance_ham_saturated_for);
    tools::log->info("σ² MOM                   = Converged : {:<4}  Saturated: {:<4}", status.variance_mom_converged_for, status.variance_mom_saturated_for);
    tools::log->info("Sₑ                       = Converged : {:<4}  Saturated: {:<4}", status.entanglement_converged_for, status.entanglement_saturated_for);
    tools::log->info("Time                     = {:<16.16f}", tid::get_unscoped("t_tot").get_time());
    tools::log->info("Mem RSS                  = {:<.1f} MB", debug::mem_rss_in_mb());
    tools::log->info("Mem Peak                 = {:<.1f} MB", debug::mem_hwm_in_mb());
    tools::log->info("Mem VM                   = {:<.1f} MB", debug::mem_vm_in_mb());
    tools::log->info("{:=^60}", "");
}
