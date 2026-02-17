
#include "contraction_policy.h"
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

    void set_global_norm_h1mv(double v) { g_H1_global_norm.store(v, std::memory_order_release); }
    void set_global_norm_h2mv(double v) { g_H2_global_norm.store(v, std::memory_order_release); }

    double get_global_norm_h1mv() { return g_H1_global_norm.load(std::memory_order_acquire); }
    double get_global_norm_h2mv() { return g_H2_global_norm.load(std::memory_order_acquire); }

}