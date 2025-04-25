#include "AlgorithmBase.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/cast.h"
#include "math/num.h"
#include "math/stat.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include <complex>
#include <h5pp/h5pp.h>
AlgorithmBase::AlgorithmBase(OptRitz opt_ritz_, AlgorithmType algo_type_) {
    status.opt_ritz  = opt_ritz_;
    status.algo_type = algo_type_;
    tools::log->set_error_handler([](const std::string &msg) { throw except::runtime_error(msg); });
    tools::log = tools::Logger::setLogger(fmt::format("{}", status.algo_type_sv()), settings::console::loglevel, settings::console::timestamp);
    tools::log->trace("Constructing class_algorithm_base");
    if(settings::test_unwind) throw std::runtime_error("Testing stack unwinding");
}
AlgorithmBase::AlgorithmBase(std::shared_ptr<h5pp::File> h5file_, OptRitz opt_ritz_, AlgorithmType algo_type_) : AlgorithmBase(opt_ritz_, algo_type_) {
    h5file = h5file_;
}

void AlgorithmBase::copy_from_tmp(StorageEvent storage_event, CopyPolicy copy_policy) {
    tools::common::h5::tmp::copy_from_tmp(*h5file, status.iter, status.step, storage_event, copy_policy);
}

void AlgorithmBase::init_bond_dimension_limits() {
    status.bond_max = settings::get_bond_max(status.algo_type); // Bond max from the loaded config file
    if(has_any_flags(status.algo_type, AlgorithmType::fDMRG, AlgorithmType::xDMRG, AlgorithmType::fLBIT)) {
        // Finite systems have limited bond dimension!
        // Be careful not to overflow long when the systems are very large (e.g. > 128 sites)
        double long_max = static_cast<double>(std::numeric_limits<long>::max());
        double bond_max = std::min(long_max, std::pow(2.0, settings::model::model_size / 2));
        status.bond_max = std::min(status.bond_max, safe_cast<long>(bond_max));
    }
    status.bond_lim = std::min(status.bond_max, settings::get_bond_min(status.algo_type));
    status.bond_min = std::max(status.bond_min, settings::get_bond_min(status.algo_type));
    if(settings::strategy::bond_increase_when == UpdatePolicy::NEVER) {
        status.bond_lim                   = status.bond_max;
        status.bond_limit_has_reached_max = true;
    }
    // Sanity check
    if(status.bond_lim == 0) throw except::runtime_error("Bond dimension limit invalid: {}", status.bond_lim);
    tools::log->info("Initialized bond dimension limits: min {} lim {} max {}", status.bond_min, status.bond_lim, status.bond_max);
}

void AlgorithmBase::init_truncation_error_limits() {
    if(settings::strategy::trnc_decrease_when == UpdatePolicy::NEVER) {
        status.trnc_min = settings::precision::svd_truncation_min;
        status.trnc_lim = settings::precision::svd_truncation_min;
        status.trnc_max = settings::precision::svd_truncation_min;
    } else {
        status.trnc_min = settings::precision::svd_truncation_min;
        status.trnc_lim = settings::precision::svd_truncation_max;
        status.trnc_max = settings::precision::svd_truncation_max;
    }
    if(settings::strategy::trnc_decrease_when == UpdatePolicy::NEVER) {
        status.trnc_lim                   = status.trnc_min;
        status.trnc_limit_has_reached_min = true;
    }
    // Sanity check
    if(status.trnc_lim == 0.0) throw except::runtime_error("Truncation error limit invalid: {}", status.trnc_lim);
    tools::log->info("Initialized truncation error limits: max {:8.2e} lim {:8.2e} min {:8.2e}", status.trnc_max, status.trnc_lim, status.trnc_min);
}

template<typename T>
size_t AlgorithmBase::count_convergence(const std::vector<T> &Y_vec, T threshold, size_t start_idx) {
    size_t scount = 0; // Counts how many converged points there have been since saturation (start_idx)
    for(const auto &[i, y] : iter::enumerate(Y_vec)) {
        if(i < start_idx) continue;
        if(y <= threshold) scount++;
    }
    size_t rcount = 0; // Counts in reverse how many converged points there have been in a row. Useful with noisy signals that can't saturate.
    for(const auto &y : iter::reverse(Y_vec)) {
        if(y <= threshold)
            rcount++;
        else
            break;
    }
    return std::max(scount, rcount);
}
template size_t AlgorithmBase::count_convergence(const std::vector<fp32> &Y_vec, fp32 threshold, size_t start_idx);
template size_t AlgorithmBase::count_convergence(const std::vector<fp64> &Y_vec, fp64 threshold, size_t start_idx);
template size_t AlgorithmBase::count_convergence(const std::vector<fp128> &Y_vec, fp128 threshold, size_t start_idx);


template<typename T>
AlgorithmBase::SaturationReport<T> AlgorithmBase::check_saturation(const std::vector<T> &Y_vec, T sensitivity, SaturationPolicy policy) {
    SaturationReport<T> report;
    constexpr size_t    min_data_points = 2;
    if(Y_vec.size() < min_data_points) { return report; }

    report.Y_vec = Y_vec;

    // Rescale
    if(has_flag(policy, SaturationPolicy::log))
        for(auto &y : report.Y_vec) y = std::log10(y);

    report.Y_sat.resize(report.Y_vec.size());

    // Get the cumulative max/min of the rescaled signal
    report.Y_avg.resize(report.Y_vec.size());
    report.Y_med.resize(report.Y_vec.size());
    report.Y_min.resize(report.Y_vec.size());
    report.Y_max.resize(report.Y_vec.size());
    report.Y_nim.resize(report.Y_vec.size());
    report.Y_xam.resize(report.Y_vec.size());
    report.Y_mid.resize(report.Y_vec.size());
    report.Y_dif.resize(report.Y_vec.size());
    for(size_t i = 0; i < report.Y_avg.size(); ++i) report.Y_avg[i] = stat::mean(report.Y_vec, i);
    for(size_t i = 0; i < report.Y_med.size(); ++i) report.Y_med[i] = stat::median(report.Y_vec, i);
    for(size_t i = 1; i <= report.Y_min.size(); ++i) report.Y_min[i - 1] = stat::min(report.Y_vec, 0, i);
    for(size_t i = 1; i <= report.Y_max.size(); ++i) report.Y_max[i - 1] = stat::max(report.Y_vec, 0, i);
    for(size_t i = 0; i < report.Y_nim.size(); ++i) report.Y_nim[i] = stat::min(report.Y_vec, i);
    for(size_t i = 0; i < report.Y_xam.size(); ++i) report.Y_xam[i] = stat::max(report.Y_vec, i);

    // Update min and max so they follow the hull/contour of Y_vec
    for(size_t i = 0; i < report.Y_min.size(); ++i) report.Y_min[i] = std::max(report.Y_min[i], report.Y_nim[i]);
    for(size_t i = 0; i < report.Y_max.size(); ++i) report.Y_max[i] = std::min(report.Y_max[i], report.Y_xam[i]);
    // In the last point, the min/max gap shrinks to zero. Fix that by using the last two values
    report.Y_min.back() = std::min(report.Y_vec.rbegin()[0], report.Y_vec.rbegin()[1]);
    report.Y_max.back() = std::max(report.Y_vec.rbegin()[0], report.Y_vec.rbegin()[1]);
    for(size_t i = 0; i < report.Y_vec.size(); ++i) report.Y_mid[i] = static_cast<T>(0.5) * (report.Y_min[i] + report.Y_max[i]);
    // The fix to min/max causes the last to entries in Y_mid to be identical. Here we fix that.
    report.Y_mid.back() = static_cast<T>(0.5) * (report.Y_mid.back() + report.Y_vec.back());

    for(size_t i = 1; i < report.Y_vec.size(); ++i) report.Y_dif[i] = report.Y_mid[i] - report.Y_mid[i - 1];
    report.Y_dif[0] = report.Y_dif[1]; // Fill with the second value

    if(not num::all_equal(report.Y_vec.size(), report.Y_min.size(), report.Y_max.size(), report.Y_sat.size()))
        throw except::logic_error("Report vectors are not equal size:\n Y_vec {}\n Y_min {}\n Y_max {}\n Y_sat {}", report.Y_vec.size(), report.Y_min.size(),
                                  report.Y_max.size(), report.Y_sat.size());

    // Calculate the running standard deviation from i to end
    report.Y_vec_std.resize(report.Y_vec.size());
    report.Y_avg_std.resize(report.Y_avg.size());
    report.Y_med_std.resize(report.Y_med.size());
    report.Y_min_std.resize(report.Y_min.size());
    report.Y_max_std.resize(report.Y_max.size());
    report.Y_mid_std.resize(report.Y_mid.size());
    report.Y_dif_avg.resize(report.Y_dif.size());
    // size_t dif_offset = report.Y_dif.size() >= 5 ?  report.Y_dif.size()-5 : report.Y_dif.size()-1;
    for(size_t i = 0; i < report.Y_vec.size() - 1; ++i) report.Y_vec_std[i] = stat::stdev(report.Y_vec, i);
    for(size_t i = 0; i < report.Y_avg.size() - 1; ++i) report.Y_avg_std[i] = stat::stdev(report.Y_avg, i);
    for(size_t i = 0; i < report.Y_med.size() - 1; ++i) report.Y_med_std[i] = stat::stdev(report.Y_med, i);
    for(size_t i = 0; i < report.Y_min.size() - 1; ++i) report.Y_min_std[i] = stat::stdev(report.Y_min, i);
    for(size_t i = 0; i < report.Y_max.size() - 1; ++i) report.Y_max_std[i] = stat::stdev(report.Y_max, i);
    for(size_t i = 0; i < report.Y_mid.size() - 1; ++i) report.Y_mid_std[i] = stat::stdev(report.Y_mid, i);
    for(size_t i = 0; i < report.Y_dif.size(); ++i) {
        auto ilim           = report.Y_dif.size() > 4 ? std::min(i, report.Y_dif.size() - 4) : i;
        report.Y_dif_avg[i] = std::abs(stat::mean(report.Y_dif, ilim));
    }
    report.Y_vec_std.back() = report.Y_vec_std.rbegin()[1]; // The last entry is always zero, so just reuse the penultimate entry
    report.Y_avg_std.back() = report.Y_avg_std.rbegin()[1]; // The last entry is always zero, so just reuse the penultimate entry
    report.Y_med_std.back() = report.Y_med_std.rbegin()[1]; // The last entry is always zero, so just reuse the penultimate entry
    report.Y_min_std.back() = report.Y_min_std.rbegin()[1]; // The last entry is always zero, so just reuse the penultimate entry
    report.Y_max_std.back() = report.Y_max_std.rbegin()[1]; // The last entry is always zero, so just reuse the penultimate entry
    report.Y_mid_std.back() = report.Y_mid_std.rbegin()[1]; // The last entry is always zero, so just reuse the penultimate entry
    report.Y_mov_std        = stat::stdev_moving(report.Y_vec, static_cast<T>(0.20));

    // auto idx90 = report.Y_vec.size() * 8ul/10ul;
    // auto fluctuation =  std::max(report.Y_vec_std.at(idx90), sensitivity);
    for(size_t i = 0; i < report.Y_sat.size(); ++i) {
        // Saturated if the std is below 1 sigma compared to the surrounding fluctuations.
        auto fluctuation = std::max<T>(sensitivity, report.Y_mov_std[i] * static_cast<T>(0.341));
        bool vec_sat     = report.Y_vec_std[i] < fluctuation and has_flag(policy, SaturationPolicy::val);
        bool avg_sat     = report.Y_avg_std[i] < sensitivity and has_flag(policy, SaturationPolicy::avg);
        bool med_sat     = report.Y_med_std[i] < sensitivity and has_flag(policy, SaturationPolicy::med);
        bool min_sat     = report.Y_min_std[i] < sensitivity and has_flag(policy, SaturationPolicy::min);
        bool max_sat     = report.Y_max_std[i] < sensitivity and has_flag(policy, SaturationPolicy::max);
        bool mid_sat     = report.Y_mid_std[i] < fluctuation and has_flag(policy, SaturationPolicy::mid);
        bool mov_sat     = report.Y_mov_std[i] < sensitivity and has_flag(policy, SaturationPolicy::mov);
        bool dif_sat     = report.Y_dif_avg[i] < sensitivity and has_flag(policy, SaturationPolicy::dif);
        report.Y_sat[i]  = vec_sat or avg_sat or med_sat or min_sat or max_sat or mid_sat or mov_sat or dif_sat;
    }

    // Since the last element in Y_vec is always zero, Y_sat is always zero. we just copy the saturation state of the second to last element.
    // if(report.Y_sat.size() > 1) { report.Y_sat.back() = report.Y_sat.rbegin()[1]; }

    // From the end, count how many Y_sat[i] are 1,  before finding a 0.
    report.saturated_point = report.Y_sat.size() - 1;
    for(const auto &[i, y] : iter::enumerate_reverse(report.Y_sat)) {
        if(y == 0) break;
        report.saturated_point = i;
        report.saturated_count++;
    }
    report.has_computed      = true;
    report.saturation_policy = policy;
    report.has_saturated     = report.saturated_count > 0;
    return report;
}

template AlgorithmBase::SaturationReport<fp32>  AlgorithmBase::check_saturation(const std::vector<fp32> &Y_vec, fp32 sensitivity, SaturationPolicy policy);
template AlgorithmBase::SaturationReport<fp64>  AlgorithmBase::check_saturation(const std::vector<fp64> &Y_vec, fp64 sensitivity, SaturationPolicy policy);
template AlgorithmBase::SaturationReport<fp128> AlgorithmBase::check_saturation(const std::vector<fp128> &Y_vec, fp128 sensitivity, SaturationPolicy policy);