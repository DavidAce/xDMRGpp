#pragma once

#include "algorithms/AlgorithmStatus.h"
#include "config/enums.h"
#include <complex>
#include <memory>
#include <optional>
#include <vector>

namespace h5pp {
    class File;
}
namespace spdlog {
    class logger;
}

class AlgorithmBase {
    public:
    AlgorithmBase() = default;
    AlgorithmBase(OptRitz opt_ritz_, AlgorithmType algo_type_);
    AlgorithmBase(std::shared_ptr<h5pp::File> h5ppFile_, OptRitz opt_ritz_, AlgorithmType algo_type_);

    std::shared_ptr<h5pp::File> h5file;
    AlgorithmStatus             status;
    static constexpr double     quietNaN = std::numeric_limits<double>::quiet_NaN();

    // Virtual Functions
    virtual void run()                                                                                                         = 0;
    virtual void run_algorithm()                                                                                               = 0;
    virtual void run_preprocessing()                                                                                           = 0;
    virtual void run_postprocessing()                                                                                          = 0;
    virtual void update_state()                                                                                                = 0;
    virtual void check_convergence()                                                                                           = 0;
    virtual void write_to_file(StorageEvent storage_event = StorageEvent::ITERATION, CopyPolicy copy_policy = CopyPolicy::TRY) = 0;
    virtual void print_status()                                                                                                = 0;
    virtual void print_status_full()                                                                                           = 0;
    virtual void clear_convergence_status()                                                                                    = 0;
    virtual void update_precision_limit(std::optional<double> energy_upper_bound = std::nullopt)                               = 0;
    virtual void update_bond_dimension_limit()                                                                                 = 0;
    virtual void update_truncation_error_limit()                                                                               = 0;

    // common functions
    void copy_from_tmp(StorageEvent storage_event = StorageEvent::ITERATION, CopyPolicy copy_policy = CopyPolicy::TRY);
    void init_bond_dimension_limits();
    void init_truncation_error_limits();

    protected:
    // enum class SaturationScale { lin, log };
    template<typename T>
    struct SaturationReport {
        bool             has_computed      = false;
        bool             has_saturated     = false;
        size_t           saturated_count   = 0;
        size_t           saturated_point   = 0;
        SaturationPolicy saturation_policy = SaturationPolicy::val;
        std::vector<T>   Y_vec;     // The values used to measure saturation. This is the log of given data when log is on
        std::vector<T>   Y_avg;     // The values used to measure saturation. This is the log of given data when log is on
        std::vector<T>   Y_med;     // The values used to measure saturation. This is the log of given data when log is on
        std::vector<T>   Y_min;     // The cumulative minimum of Y_vec
        std::vector<T>   Y_nim;     // The cumulative minimum of Y_vec
        std::vector<T>   Y_max;     // The cumulative maximum of Y_vec
        std::vector<T>   Y_xam;     // The cumulative maximum of Y_vec
        std::vector<T>   Y_mid;     // The midpoint between Y_min and Y_max
        std::vector<T>   Y_dif;     // The cumulative average difference between adjacent points
        std::vector<T>   Y_vec_std; // The "moving start" standard deviation of Y_vec from [i:end]
        std::vector<T>   Y_avg_std; // The "moving start" standard deviation of Y_vec from [i:end]
        std::vector<T>   Y_med_std; // The "moving start" standard deviation of Y_vec from [i:end]
        std::vector<T>   Y_min_std; // The "moving start" standard deviation of Y_min from [i:end]
        std::vector<T>   Y_max_std; // The "moving start" standard deviation of Y_max from [i:end]
        std::vector<T>   Y_mov_std; // The "moving window" standard error of 25% width.
        std::vector<T>   Y_mid_std; // The standard deviation of the midpoint between Y_min and Y_max
        std::vector<T>   Y_dif_avg; // The standard deviation of the midpoint between Y_min and Y_max
        std::vector<int> Y_sat;     // Flags that tell if Y_vec has saturated at that index
    };
    template<typename T>
    size_t count_convergence(const std::vector<T> &Y_vec, T threshold, size_t start_idx = 0);
    template<typename T>
    SaturationReport<T> check_saturation(const std::vector<T> &Y_vec, T threshold, SaturationPolicy policy);
};
