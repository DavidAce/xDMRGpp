#define ANKERL_NANOBENCH_IMPLEMENT
#include "config/cuda.h"
#include "config/settings.h"
#include "config/threading.h"
#include "debug/affinity.h"
#include "math/float.h"
#include "nanobench.h"
#include "tools/common/contraction/contraction_cutensor.h"
#include "tools/common/contraction/contraction_policy.h"
#include "tools/common/contraction/contraction_tblis.h"
#include "tools/common/contraction/matrix_vector_product.h"
#include "tools/common/log.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <limits>
#include <map>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <fmt/core.h>
#include <unsupported/Eigen/CXX11/Tensor>
#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis.h>
#endif
#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace {
    int parse_gpu_id(std::string value) {
        std::ranges::transform(value, value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if(value == "auto") return -1;
        std::size_t pos = 0;
        int         dev = std::stoi(value, &pos);
        if(pos != value.size()) throw CLI::ValidationError("--gpu-id", "expected 'auto', -1, or a non-negative integer");
        if(dev < -1) throw CLI::ValidationError("--gpu-id", "expected 'auto', -1, or a non-negative integer");
        return dev;
    }

    template<typename Scalar>
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    bool has_backend(const std::vector<std::string> &backends, std::string_view needle) {
        return std::ranges::find(backends, needle) != backends.end();
    }

    std::string_view gpu_policy_name(GpuPolicy policy) {
        switch(policy) {
            case GpuPolicy::ON: return "ON";
            case GpuPolicy::OFF: return "OFF";
            case GpuPolicy::TRY: return "TRY";
        }
        return "UNKNOWN";
    }

    double to_mib(std::size_t bytes) { return static_cast<double>(bytes) / 1024.0 / 1024.0; }

    template<typename Scalar, int Rank>
    void fill_random(Eigen::Tensor<Scalar, Rank> &tensor) {
        Eigen::Map<VectorType<Scalar>>(tensor.data(), tensor.size()).setRandom();
    }

    template<typename Scalar>
    double max_abs_diff(const Eigen::Tensor<Scalar, 3> &lhs, const Eigen::Tensor<Scalar, 3> &rhs) {
        double diff = 0.0;
        for(Eigen::Index idx = 0; idx < lhs.size(); ++idx) diff = std::max(diff, static_cast<double>(std::abs(lhs.data()[idx] - rhs.data()[idx])));
        return diff;
    }

    template<typename Scalar>
    double max_abs_value(const Eigen::Tensor<Scalar, 3> &tensor) {
        double value = 0.0;
        for(Eigen::Index idx = 0; idx < tensor.size(); ++idx) value = std::max(value, static_cast<double>(std::abs(tensor.data()[idx])));
        return value;
    }

    template<typename Scalar>
    double diff_tolerance(const Eigen::Tensor<Scalar, 3> &reference) {
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        const auto scale = std::max(1.0, max_abs_value(reference));
        return 2.0 * std::sqrt(static_cast<double>(std::numeric_limits<RealScalar>::epsilon())) * scale;
    }

    template<typename Scalar>
    void profile_tblis_stages([[maybe_unused]] const Eigen::Tensor<Scalar, 3> &mps,
                              [[maybe_unused]] const Eigen::Tensor<Scalar, 4> &mpo,
                              [[maybe_unused]] const Eigen::Tensor<Scalar, 3> &envL,
                              [[maybe_unused]] const Eigen::Tensor<Scalar, 3> &envR) {
#if defined(DMRG_ENABLE_TBLIS)
        if constexpr(settings::tblis_use_openmp) { tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads())); }
        using tools::common::contraction::contract_tblis;
        using clock = std::chrono::steady_clock;
        const auto to_ms = [](clock::duration dt) { return std::chrono::duration<double, std::milli>(dt).count(); };

        if(mps.dimension(1) >= mps.dimension(2)) {
            auto T1  = Eigen::Tensor<Scalar, 4>(mps.dimension(0), mps.dimension(2), envL.dimension(1), envL.dimension(2));
            auto T2  = Eigen::Tensor<Scalar, 4>(mpo.dimension(1), mpo.dimension(3), mps.dimension(2), envL.dimension(1));
            auto res = Eigen::Tensor<Scalar, 3>(mpo.dimension(3), mps.dimension(1), mps.dimension(2));

            const auto t0 = clock::now();
            contract_tblis(mps.data(), mps.dimensions(), envL.data(), envL.dimensions(), T1.data(), T1.dimensions(), "afb", "fcd", "abcd", nullptr);
            const auto t1 = clock::now();
            contract_tblis(mpo.data(), mpo.dimensions(), T1.data(), T1.dimensions(), T2.data(), T2.dimensions(), "qhri", "rgjq", "higj", nullptr);
            const auto t2 = clock::now();
            contract_tblis(T2.data(), T2.dimensions(), envR.data(), envR.dimensions(), res.data(), res.dimensions(), "higj", "gkh", "ijk", nullptr);
            const auto t3 = clock::now();
            fmt::print("tblis stages ms | T1 {:.3f} | T2 {:.3f} | T3 {:.3f}\n", to_ms(t1 - t0), to_ms(t2 - t1), to_ms(t3 - t2));
        } else {
            auto T1  = Eigen::Tensor<Scalar, 4>(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
            auto T2  = Eigen::Tensor<Scalar, 4>(mps.dimension(1), envR.dimension(1), mpo.dimension(0), mpo.dimension(3));
            auto res = Eigen::Tensor<Scalar, 3>(mpo.dimension(3), mps.dimension(1), mps.dimension(2));

            const auto t0 = clock::now();
            contract_tblis(mps.data(), mps.dimensions(), envR.data(), envR.dimensions(), T1.data(), T1.dimensions(), "abf", "fcd", "abcd", nullptr);
            const auto t1 = clock::now();
            contract_tblis(T1.data(), T1.dimensions(), mpo.data(), mpo.dimensions(), T2.data(), T2.dimensions(), "qijk", "rkql", "ijrl", nullptr);
            const auto t2 = clock::now();
            contract_tblis(T2.data(), T2.dimensions(), envL.data(), envL.dimensions(), res.data(), res.dimensions(), "qkri", "qjr", "ijk", nullptr);
            const auto t3 = clock::now();
            fmt::print("tblis stages ms | T1 {:.3f} | T2 {:.3f} | T3 {:.3f}\n", to_ms(t1 - t0), to_ms(t2 - t1), to_ms(t3 - t2));
        }
#endif
    }

    template<typename Scalar>
    void run_case(long d,
                  long chiL,
                  long chiR,
                  long wL,
                  long wR,
                  std::size_t epochs,
                  std::size_t iterations,
                  const std::vector<std::string> &backends,
                  bool profile_tblis,
                  bool show_openmp_placement) {
        using tools::common::contraction::matrix_vector_product;
        using tools::common::contraction::internal::get_cutensor_operation_bytes;
        using tools::common::contraction::internal::InfoH1Mv;

        auto res_eigen = Eigen::Tensor<Scalar, 3>(d, chiL, chiR);
        auto res       = Eigen::Tensor<Scalar, 3>(d, chiL, chiR);
        auto mps       = Eigen::Tensor<Scalar, 3>(d, chiL, chiR);
        auto mpo       = Eigen::Tensor<Scalar, 4>(wL, wR, d, d);
        auto envL      = Eigen::Tensor<Scalar, 3>(chiL, chiL, wL);
        auto envR      = Eigen::Tensor<Scalar, 3>(chiR, chiR, wR);

        res.setZero();
        res_eigen.setZero();
        fill_random(mps);
        fill_random(mpo);
        fill_random(envL);
        fill_random(envR);

        const bool cutensor_selected = has_backend(backends, "cutensor");
        const bool auto_selected     = has_backend(backends, "auto");
        if(cutensor_selected or auto_selected) {
            std::size_t required_bytes = 0;
            if constexpr(tools::common::contraction::internal::cutensor_supported_v<Scalar>) required_bytes = get_cutensor_operation_bytes<Scalar>(mps, mpo, envL, envR);
            const auto memory_status = config::cuda::query_memory(required_bytes);
            fmt::print("problem size {} | gpu_policy {} | gpu_id {} | gpu_switchsize {} | {}\n", mps.size(), gpu_policy_name(settings::cuda::gpu_policy),
                       settings::cuda::gpu_id, settings::cuda::gpu_switchsize, config::cuda::description());
            if(required_bytes > 0) {
                fmt::print("cutensor MiB {:.2f} | free {:.2f} | usable {:.2f} | fits {}\n", to_mib(required_bytes), to_mib(memory_status.free_bytes),
                           to_mib(memory_status.usable_bytes), memory_status.fits ? "yes" : "no");
            }
        } else {
            fmt::print("problem size {}\n", mps.size());
        }

        if(show_openmp_placement)
            if(auto placement = debug::affinity::format_openmp_placement()) fmt::print("{}\n", *placement);

        {
            auto h1info = SetH1MvInfo(ContractionBackend::EIGEN, mpo.dimensions());
            matrix_vector_product(res_eigen, mps, mpo, envL, envR);
        }

        auto bench_case = [&](auto &&make_info, const char *name) {
            res.setZero();
            try {
                auto h1info = SetH1MvInfo(make_info());
                matrix_vector_product(res, mps, mpo, envL, envR);
            } catch(const std::exception &ex) {
                fmt::print("{:>16} unavailable: {}\n", name, ex.what());
                return;
            }
            const auto diff      = max_abs_diff(res, res_eigen);
            const auto tolerance = diff_tolerance(res_eigen);
            if(diff > tolerance) fmt::print("{:>16} diff to eigen: {:.3e} (tol {:.3e})\n", name, diff, tolerance);
            ankerl::nanobench::Bench().epochs(epochs).minEpochIterations(iterations).run(name, [&] {
                auto h1info = SetH1MvInfo(make_info());
                matrix_vector_product(res, mps, mpo, envL, envR);
                ankerl::nanobench::doNotOptimizeAway(res);
            });
        };

        if(has_backend(backends, "auto"))
            bench_case([&] { return InfoH1Mv{.backend = ContractionBackend::AUTO, .precision = ContractionPrecision::SAME, .H1_local_dims = mpo.dimensions()}; },
                       "matvec auto");
        if(has_backend(backends, "eigen"))
            bench_case([&] { return InfoH1Mv{.backend = ContractionBackend::EIGEN, .precision = ContractionPrecision::SAME, .H1_local_dims = mpo.dimensions()}; },
                       "matvec eigen");
        if(has_backend(backends, "tblis"))
            bench_case([&] { return InfoH1Mv{.backend = ContractionBackend::TBLIS, .precision = ContractionPrecision::SAME, .H1_local_dims = mpo.dimensions()}; },
                       "matvec tblis");
        if(profile_tblis and has_backend(backends, "tblis")) profile_tblis_stages(mps, mpo, envL, envR);
        if(has_backend(backends, "x2"))
            bench_case([&] { return InfoH1Mv{.backend = ContractionBackend::AUTO, .precision = ContractionPrecision::X2, .H1_local_dims = mpo.dimensions()}; },
                       "matvec x2");
        if(has_backend(backends, "cutensor") and config::cuda::compiled())
            bench_case([&] { return InfoH1Mv{.backend = ContractionBackend::CUTENSOR, .precision = ContractionPrecision::SAME, .H1_local_dims = mpo.dimensions()}; },
                       "matvec cutensor");
    }
}

int main(int argc, char **argv) {
    tools::log = tools::Logger::setLogger("matvec-dispatch", 2, true);

    long        d          = 2;
    long        chiL       = 128;
    long        chiR       = 128;
    long        wL         = 12;
    long        wR         = 12;
    std::size_t epochs     = 5;
    std::size_t iterations = 50;
    std::string           dtype      = "fp64";
    std::string           gpu_id     = "auto";
    std::map<std::string, GpuPolicy, std::less<>> gpu_policy_map{{"ON", GpuPolicy::ON}, {"OFF", GpuPolicy::OFF}, {"TRY", GpuPolicy::TRY}};
    std::vector<std::string> backends {"auto", "eigen", "tblis", "x2", "cutensor"};
    bool                     profile_tblis         = false;
    bool                     show_openmp_placement = false;
    bool                     check_affinity_only   = false;

    CLI::App app{"Benchmark matrix_vector_product backend dispatch"};
    app.add_option("--dtype", dtype, "Scalar type: fp32, fp64, cx32, cx64");
    app.add_option("-d,--physdim", d, "Physical dimension");
    app.add_option("--chiL", chiL, "Left bond dimension");
    app.add_option("--chiR", chiR, "Right bond dimension");
    app.add_option("--wL", wL, "Left MPO bond dimension");
    app.add_option("--wR", wR, "Right MPO bond dimension");
    app.add_option("-t,--threads", settings::threading::num_threads, "Eigen worker threads");
    app.add_option("--backends", backends, "Backends to benchmark: auto, eigen, tblis, x2, cutensor")->delimiter(',');
    app.add_flag("--profile-tblis", profile_tblis, "Print the three internal TBLIS stage timings once");
    app.add_flag("--show-openmp-placement", show_openmp_placement, "Print the CPU placement of one OpenMP parallel region");
    app.add_flag("--check-affinity-only", check_affinity_only, "Print affinity diagnostics, warn on restricted or oversubscribed CPU masks, and exit");
    app.add_option("--gpu-policy", settings::cuda::gpu_policy, "GPU contraction policy [ON | OFF | TRY]")
        ->transform(CLI::CheckedTransformer(gpu_policy_map, CLI::ignore_case))
        ->type_name("ENUM");
    app.add_option("--gpu-id", gpu_id, "CUDA device id, or auto");
    app.add_option("--gpu-switchsize", settings::cuda::gpu_switchsize, "Minimum problem size before AUTO can use the GPU");
    app.add_option("--gpu-max-alloc-fraction", settings::cuda::gpu_max_alloc_fraction, "Refuse GPU matvec when the estimated allocation exceeds this fraction of free memory");
    app.add_option("--epochs", epochs, "Nanobench epochs");
    app.add_option("--iterations", iterations, "Minimum nanobench iterations");
    app.parse(argc, argv);

    for(auto &backend : backends) std::ranges::transform(backend, backend.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    settings::cuda::gpu_id = parse_gpu_id(gpu_id);
    settings::configure_threads();

    const auto affinity_status = debug::affinity::query_status();
    if(affinity_status)
        for(const auto &message : debug::affinity::describe_pathologies(*affinity_status)) tools::log->warn("{}", message);
    if(check_affinity_only) {
        if(affinity_status) fmt::print("{}\n", debug::affinity::format_status(*affinity_status));
        if(show_openmp_placement)
            if(auto placement = debug::affinity::format_openmp_placement()) fmt::print("{}\n", *placement);
        return 0;
    }

    if(dtype == "fp32") return run_case<fp32>(d, chiL, chiR, wL, wR, epochs, iterations, backends, profile_tblis, show_openmp_placement), 0;
    if(dtype == "fp64") return run_case<fp64>(d, chiL, chiR, wL, wR, epochs, iterations, backends, profile_tblis, show_openmp_placement), 0;
    if(dtype == "cx32") return run_case<cx32>(d, chiL, chiR, wL, wR, epochs, iterations, backends, profile_tblis, show_openmp_placement), 0;
    if(dtype == "cx64") return run_case<cx64>(d, chiL, chiR, wL, wR, epochs, iterations, backends, profile_tblis, show_openmp_placement), 0;

    throw std::runtime_error(fmt::format("Unsupported dtype {}", dtype));
}
