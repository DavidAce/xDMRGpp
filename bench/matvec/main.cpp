#define ANKERL_NANOBENCH_IMPLEMENT
#include "config/settings.h"
#include "config/threading.h"
#include "env/environment.h"
#include "math/tenx.h"
#include "math/x2/Tensor.h"
#include "nanobench.h"
#include "tools/common/contraction/contraction_policy.h"
#include "tools/common/contraction/matrix_vector_product.h"
#include "tools/common/log.h"
#include <fmt/core.h>

namespace {
    template<typename Scalar>
    void run_benchmarks(Eigen::Tensor<Scalar, 3>       &res,
                        const Eigen::Tensor<Scalar, 3> &mps,
                        const Eigen::Tensor<Scalar, 4> &mpo,
                        const Eigen::Tensor<Scalar, 3> &envL,
                        const Eigen::Tensor<Scalar, 3> &envR) {
        using tools::common::contraction::matrix_vector_product;

        auto run = [&](const char *name, auto &&fn) {
            ankerl::nanobench::Bench().warmup(4).epochs(10).minEpochIterations(100).run(name, [&] {
                fn();
                ankerl::nanobench::doNotOptimizeAway(res);
            });
        };

        run("matvec auto", [&] {
            matrix_vector_product(res, mps, mpo, envL, envR);
        });

        run("matvec eigen", [&] {
            auto h1info = SetH1MvInfo(ContractionBackend::EIGEN, mpo.dimensions());
            matrix_vector_product(res, mps, mpo, envL, envR);
        });

#if defined(DMRG_ENABLE_TBLIS)
        run("matvec tblis", [&] {
            auto h1info = SetH1MvInfo(ContractionBackend::TBLIS, mpo.dimensions());
            matrix_vector_product(res, mps, mpo, envL, envR);
        });
#endif

        run("matvec x2", [&] {
            auto h1info = SetH1MvInfo(ContractionBackend::X2, mpo.dimensions());
            matrix_vector_product(res, mps, mpo, envL, envR);
        });

        auto envL_x2 = x2::Tensor<Scalar, 3>(envL);
        auto envR_x2 = x2::Tensor<Scalar, 3>(envR);
        run("matvec x2 env-x2", [&] {
            auto h1info = SetH1MvInfo(ContractionBackend::X2, mpo.dimensions());
            matrix_vector_product(res, mps, mpo, envL_x2, envR_x2);
        });
    }
}

int main() {
    tools::log = tools::Logger::setLogger("matvec", 2, true);
    fmt::print("Compiler flags {}\n", env::build::compiler_flags);

    settings::threading::num_threads = 1;
    settings::configure_threads();

    using real = double;
    constexpr long d     = 2;
    constexpr long m     = 12;
    constexpr long chiL  = 128;
    constexpr long chiR  = 256;

    auto res  = Eigen::Tensor<real, 3>(d, chiL, chiR);
    auto mps  = Eigen::Tensor<real, 3>(d, chiL, chiR);
    auto mpo  = Eigen::Tensor<real, 4>(m, m, d, d);
    auto envL = Eigen::Tensor<real, 3>(chiL, chiL, m);
    auto envR = Eigen::Tensor<real, 3>(chiR, chiR, m);

    res.setZero();
    mps.setRandom();
    mpo.setRandom();
    envL.setRandom();
    envR.setRandom();

    tools::log->info("mps dims: {} | size: {}", mps.dimensions(), mps.size());
    tools::log->info("mpo dims: {} | size: {}", mpo.dimensions(), mpo.size());
    tools::log->info("envL dims: {} | size: {}", envL.dimensions(), envL.size());
    tools::log->info("envR dims: {} | size: {}", envR.dimensions(), envR.size());

    run_benchmarks(res, mps, mpo, envL, envR);
}
