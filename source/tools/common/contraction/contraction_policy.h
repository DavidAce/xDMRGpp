#pragma once

#include <array>
#include <concepts>
#include <limits>
#include <string_view>
#include <utility>
enum class ContractionBackend { EIGEN, TBLIS, CUTENSOR, AUTO };
enum class ContractionPrecision { SAME, X2, FP80, X2_AS_NEEDED };

inline std::string_view enum2sv(ContractionBackend backend) {
    switch(backend) {
        case ContractionBackend::EIGEN: return "EIGEN";
        case ContractionBackend::TBLIS: return "TBLIS";
        case ContractionBackend::CUTENSOR: return "CUTENSOR";
        case ContractionBackend::AUTO: return "AUTO";
    }
    return "UNKNOWN";
}

inline std::string_view enum2sv(ContractionPrecision precision) {
    switch(precision) {
        case ContractionPrecision::SAME: return "SAME";
        case ContractionPrecision::X2: return "X2";
        case ContractionPrecision::FP80: return "FP80";
        case ContractionPrecision::X2_AS_NEEDED: return "X2_AS_NEEDED";
    }
    return "UNKNOWN";
}

namespace tools::common::contraction::internal {
    struct InfoH1Mv {
        ContractionBackend  backend       = ContractionBackend::AUTO;
        ContractionPrecision precision    = ContractionPrecision::X2_AS_NEEDED;
        std::array<long, 4> H1_local_dims = {0};
        double              H1_local_norm = std::numeric_limits<double>::quiet_NaN();
    };
    struct InfoH2Mv {
        ContractionBackend  backend       = ContractionBackend::AUTO;
        ContractionPrecision precision    = ContractionPrecision::X2_AS_NEEDED;
        std::array<long, 4> H2_local_dims = {0};
        double              H2_local_norm = std::numeric_limits<double>::quiet_NaN();
    };

    struct InfoEnv {
        ContractionBackend  backend   = ContractionBackend::AUTO;
        ContractionPrecision precision = ContractionPrecision::SAME;
    };

    // Accessor returns a reference to a thread-local instance.
    InfoH1Mv &get_info_h1mv();
    InfoH2Mv &get_info_h2mv();
    InfoEnv  &get_info_env();

    // Set/get global norms .
    void   set_H1_global_norm(double H1_global_norm);
    void   set_H2_global_norm(double H2_global_norm);
    double get_H1_global_norm();
    double get_H2_global_norm();
}

struct SetH1MvInfo {
    using InfoH1Mv = tools::common::contraction::internal::InfoH1Mv;
    tools::common::contraction::internal::InfoH1Mv saved;

    explicit SetH1MvInfo(InfoH1Mv next)
        : saved(tools::common::contraction::internal::get_info_h1mv()) // snapshot old
    {
        tools::common::contraction::internal::get_info_h1mv() = next; // install new
    }

    SetH1MvInfo(ContractionBackend p) : SetH1MvInfo(InfoH1Mv{.backend = p, .precision = ContractionPrecision::SAME}) {}
    SetH1MvInfo(ContractionPrecision p) : SetH1MvInfo(InfoH1Mv{.precision = p}) {}
    SetH1MvInfo(ContractionBackend backend, ContractionPrecision precision) : SetH1MvInfo(InfoH1Mv{.backend = backend, .precision = precision}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH1MvInfo(std::array<long, 4> h1dims, T h1norm)
        : SetH1MvInfo(InfoH1Mv{
              .backend       = ContractionBackend::AUTO,
              .precision     = ContractionPrecision::X2_AS_NEEDED,
              .H1_local_dims = h1dims,
              .H1_local_norm = static_cast<double>(h1norm),
          }) {}

    SetH1MvInfo(ContractionBackend backend, std::array<long, 4> h1dims)
        : SetH1MvInfo(InfoH1Mv{.backend = backend, .precision = ContractionPrecision::SAME, .H1_local_dims = h1dims}) {}
    SetH1MvInfo(ContractionPrecision precision, std::array<long, 4> h1dims) : SetH1MvInfo(InfoH1Mv{.precision = precision, .H1_local_dims = h1dims}) {}
    SetH1MvInfo(ContractionBackend backend, ContractionPrecision precision, std::array<long, 4> h1dims)
        : SetH1MvInfo(InfoH1Mv{.backend = backend, .precision = precision, .H1_local_dims = h1dims}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH1MvInfo(ContractionBackend backend, std::array<long, 4> h1dims, T h1norm)
        : SetH1MvInfo(InfoH1Mv{.backend = backend, .precision = ContractionPrecision::SAME, .H1_local_dims = h1dims, .H1_local_norm = static_cast<double>(h1norm)}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH1MvInfo(ContractionPrecision precision, std::array<long, 4> h1dims, T h1norm)
        : SetH1MvInfo(InfoH1Mv{.precision = precision, .H1_local_dims = h1dims, .H1_local_norm = static_cast<double>(h1norm)}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH1MvInfo(ContractionBackend backend, ContractionPrecision precision, std::array<long, 4> h1dims, T h1norm)
        : SetH1MvInfo(InfoH1Mv{.backend = backend, .precision = precision, .H1_local_dims = h1dims, .H1_local_norm = static_cast<double>(h1norm)}) {}

    ~SetH1MvInfo() {
        tools::common::contraction::internal::get_info_h1mv() = saved; // restore old
    }

    SetH1MvInfo(const SetH1MvInfo &)            = delete;
    SetH1MvInfo &operator=(const SetH1MvInfo &) = delete;
};

struct SetH2MvInfo {
    using InfoH2Mv = tools::common::contraction::internal::InfoH2Mv;
    tools::common::contraction::internal::InfoH2Mv saved;

    explicit SetH2MvInfo(InfoH2Mv next)
        : saved(tools::common::contraction::internal::get_info_h2mv()) // snapshot old
    {
        tools::common::contraction::internal::get_info_h2mv() = next; // install new
    }

    SetH2MvInfo(ContractionBackend p) : SetH2MvInfo(InfoH2Mv{.backend = p, .precision = ContractionPrecision::SAME}) {}
    SetH2MvInfo(ContractionPrecision p) : SetH2MvInfo(InfoH2Mv{.precision = p}) {}
    SetH2MvInfo(ContractionBackend backend, ContractionPrecision precision) : SetH2MvInfo(InfoH2Mv{.backend = backend, .precision = precision}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH2MvInfo(std::array<long, 4> h2dims, T h2norm)
        : SetH2MvInfo(InfoH2Mv{
              .backend       = ContractionBackend::AUTO,
              .precision     = ContractionPrecision::X2_AS_NEEDED,
              .H2_local_dims = h2dims,
              .H2_local_norm = static_cast<double>(h2norm),
          }) {}

    SetH2MvInfo(ContractionBackend backend, std::array<long, 4> h2dims)
        : SetH2MvInfo(InfoH2Mv{.backend = backend, .precision = ContractionPrecision::SAME, .H2_local_dims = h2dims}) {}
    SetH2MvInfo(ContractionPrecision precision, std::array<long, 4> h2dims) : SetH2MvInfo(InfoH2Mv{.precision = precision, .H2_local_dims = h2dims}) {}
    SetH2MvInfo(ContractionBackend backend, ContractionPrecision precision, std::array<long, 4> h2dims)
        : SetH2MvInfo(InfoH2Mv{.backend = backend, .precision = precision, .H2_local_dims = h2dims}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH2MvInfo(ContractionBackend backend, std::array<long, 4> h2dims, T h2norm)
        : SetH2MvInfo(InfoH2Mv{.backend = backend, .precision = ContractionPrecision::SAME, .H2_local_dims = h2dims, .H2_local_norm = static_cast<double>(h2norm)}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH2MvInfo(ContractionPrecision precision, std::array<long, 4> h2dims, T h2norm)
        : SetH2MvInfo(InfoH2Mv{.precision = precision, .H2_local_dims = h2dims, .H2_local_norm = static_cast<double>(h2norm)}) {}

    template<typename T>
    requires(std::floating_point<T>)
    SetH2MvInfo(ContractionBackend backend, ContractionPrecision precision, std::array<long, 4> h2dims, T h2norm)
        : SetH2MvInfo(InfoH2Mv{.backend = backend, .precision = precision, .H2_local_dims = h2dims, .H2_local_norm = static_cast<double>(h2norm)}) {}

    ~SetH2MvInfo() {
        tools::common::contraction::internal::get_info_h2mv() = saved; // restore old
    }

    SetH2MvInfo(const SetH2MvInfo &)            = delete;
    SetH2MvInfo &operator=(const SetH2MvInfo &) = delete;
};

struct SetEnvInfo {
    using InfoEnv = tools::common::contraction::internal::InfoEnv;
    tools::common::contraction::internal::InfoEnv saved;

    explicit SetEnvInfo(InfoEnv next)
        : saved(tools::common::contraction::internal::get_info_env()) // snapshot old
    {
        tools::common::contraction::internal::get_info_env() = next; // install new
    }

    SetEnvInfo(ContractionBackend p) : SetEnvInfo(InfoEnv{.backend = p, .precision = ContractionPrecision::SAME}) {}
    SetEnvInfo(ContractionPrecision p) : SetEnvInfo(InfoEnv{.precision = p}) {}
    SetEnvInfo(ContractionBackend backend, ContractionPrecision precision) : SetEnvInfo(InfoEnv{.backend = backend, .precision = precision}) {}

    ~SetEnvInfo() {
        tools::common::contraction::internal::get_info_env() = saved; // restore old
    }

    SetEnvInfo(const SetEnvInfo &)            = delete;
    SetEnvInfo &operator=(const SetEnvInfo &) = delete;
};
