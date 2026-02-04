#pragma once

#include <array>
#include <limits>
#include <utility>
enum class MatVecBackend { EIGEN, TBLIS, X2, AUTO };

namespace tools::common::contraction::internal {
    struct MatVecOptions {
        MatVecBackend backend = MatVecBackend::AUTO;
        // add more knobs later: thresholds, allow_redo, etc.
        double              H1_norm = std::numeric_limits<double>::quiet_NaN();
        double              H2_norm = std::numeric_limits<double>::quiet_NaN();
        std::array<long, 4> H1_dims = {0};
        std::array<long, 4> H2_dims = {0};
    };

    // Accessor returns a reference to a thread-local instance.
    MatVecOptions &matvec_options_active();
}

struct MatVecRaiiOptions {
    using MatVecOptions = tools::common::contraction::internal::MatVecOptions;
    tools::common::contraction::internal::MatVecOptions saved;

    explicit MatVecRaiiOptions(MatVecOptions next)
        : saved(tools::common::contraction::internal::matvec_options_active()) // snapshot old
    {
        tools::common::contraction::internal::matvec_options_active() = next; // install new
    }

    MatVecRaiiOptions(MatVecBackend p) : MatVecRaiiOptions(MatVecOptions{.backend = p}) {}

    template<typename T>
    requires(std::floating_point<T>)
    MatVecRaiiOptions(T h1norm, T h2norm, std::array<long, 4> h1dims, std::array<long, 4> h2dims)
        : MatVecRaiiOptions(MatVecOptions{.backend = MatVecBackend::AUTO,
                                          .H1_norm = static_cast<double>(h1norm),
                                          .H2_norm = static_cast<double>(h2norm),
                                          .H1_dims = h1dims,
                                          .H2_dims = h2dims}) {}

    ~MatVecRaiiOptions() {
        tools::common::contraction::internal::matvec_options_active() = saved; // restore old
    }

    MatVecRaiiOptions(const MatVecRaiiOptions &)            = delete;
    MatVecRaiiOptions &operator=(const MatVecRaiiOptions &) = delete;
};