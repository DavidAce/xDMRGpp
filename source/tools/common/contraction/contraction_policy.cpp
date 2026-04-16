
#include "contraction_policy.h"
#include "config/cuda.h"
#include "config/settings.h"
#include <fmt/core.h>
#include <atomic>
#include <memory>

namespace tools::common::contraction::internal {

    InfoH1Mv &get_info_h1mv() {
        thread_local InfoH1Mv info{};
        return info;
    }

    InfoH2Mv &get_info_h2mv() {
        thread_local InfoH2Mv info{};
        return info;
    }

    InfoEnv &get_info_env() {
        thread_local InfoEnv info{};
        return info;
    }

    // Atomic doubles. NaN means "unset".
    static std::atomic<double> g_H1_global_norm{std::numeric_limits<double>::quiet_NaN()};
    static std::atomic<double> g_H2_global_norm{std::numeric_limits<double>::quiet_NaN()};

    void set_H1_global_norm(double v) { g_H1_global_norm.store(v, std::memory_order_release); }
    void set_H2_global_norm(double v) { g_H2_global_norm.store(v, std::memory_order_release); }

    double get_H1_global_norm() { return g_H1_global_norm.load(std::memory_order_acquire); }
    double get_H2_global_norm() { return g_H2_global_norm.load(std::memory_order_acquire); }

}

namespace tools::common::contraction {
    [[nodiscard]] static ContractionBackend
        pick_normal_precision_backend(ContractionBackend requested, long problem_size, bool tblis_supported, bool cutensor_supported) {
        const bool gpu_planned =
            cutensor_supported and config::cuda::available() and
            (requested == ContractionBackend::CUTENSOR ||
             (requested == ContractionBackend::AUTO && problem_size >= static_cast<long>(settings::cuda::gpu_switchsize)));

        switch(requested) {
            case ContractionBackend::EIGEN: return ContractionBackend::EIGEN;
            case ContractionBackend::TBLIS: return tblis_supported ? ContractionBackend::TBLIS : ContractionBackend::EIGEN;
            case ContractionBackend::CUTENSOR:
                if(gpu_planned) return ContractionBackend::CUTENSOR;
                if(tblis_supported) return ContractionBackend::TBLIS;
                return ContractionBackend::EIGEN;
            case ContractionBackend::AUTO:
                if(gpu_planned) return ContractionBackend::CUTENSOR;
                if(tblis_supported) return ContractionBackend::TBLIS;
                return ContractionBackend::EIGEN;
        }
        return ContractionBackend::EIGEN;
    }

    std::string get_h1_contraction_summary(long problem_size, bool tblis_supported, bool cutensor_supported) {
        const auto info = internal::get_info_h1mv();
        auto       res  = fmt::format("{}/{}", enum2sv(info.backend), enum2sv(info.precision));
        if(info.precision == ContractionPrecision::SAME or info.precision == ContractionPrecision::X2_AS_NEEDED) {
            const auto backend = pick_normal_precision_backend(info.backend, problem_size, tblis_supported, cutensor_supported);
            res += fmt::format("->{}", enum2sv(backend));
        }
        return res;
    }
}
