#include <general/iter.h>
#include <config/blas_backend.h>
#include <h5pp/h5pp.h>
#include <math/eig.h>
#include <math/eig/matvec/matvec_mpo.h>
#include <math/tenx.h>
#include <primme/primme.h>
#include <tid/tid.h>
#include <tools/common/contraction.h>
#include <tools/common/log.h>
#include <tools/common/prof.h>
#include <tools/finite/opt_mps.h>

template<typename MatrixProductType>
void simps_preconditioner(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, int *ierr) {
    if(x == nullptr) return;
    if(y == nullptr) return;
    if(primme == nullptr) return;
    using T = typename MatrixProductType::Scalar;
    // When optimizing variance, the objects below refer to the ones to construct (H-E)², as in the second moment
    const auto  H_ptr     = static_cast<MatrixProductType *>(primme->matrix);
    const auto  shape_mps = H_ptr->get_shape_mps();
    const auto &mpo       = H_ptr->get_mpo();
    const auto &envL      = H_ptr->get_envL();
    const auto &envR      = H_ptr->get_envR();
    for(int i = 0; i < *blockSize; i++) {
        auto mps_in  = Eigen::TensorMap<const Eigen::Tensor<T, 3>>(static_cast<T *>(x) + *ldx * i, shape_mps);
        auto mps_out = Eigen::TensorMap<Eigen::Tensor<T, 3>>(static_cast<T *>(y) + *ldy * i, shape_mps);
        tools::common::contraction::matrix_inverse_vector_product(mps_out, mps_in, mpo, envL, envR);
    }
    *ierr = 0;
}

std::string solve(const eig::settings &config, const h5pp::File &h5file, std::string_view group) {
    std::string tgt;
    {
        auto dims  = h5file.readAttribute<std::array<long, 3>>(group, "dims");
        auto time  = h5file.readAttribute<double>(group, "time");
        auto iter  = h5file.readAttribute<long>(group, "iter");
        auto eval  = h5file.readAttribute<double>(group, "eval");
        auto rnorm = h5file.readAttribute<double>(group, "rnorm");
        auto mv    = h5file.readAttribute<long>(group, "mv");
        auto op    = h5file.readAttribute<long>(group, "op");
        tgt = fmt::format("Target | iter {:>5} | op {:>5} | mv {:>7} | rnorm {:8.2e} | var {:>12.6e} | time {:>10.3f} s | dims {}", iter, op, mv, rnorm, eval,
                          time, dims);
    }

    Eigen::Tensor<fp64, 3> mps   = h5file.readDataset<Eigen::Tensor<cx64, 3>>(fmt::format("{}/mps", group)).real();
    auto                   mpo2  = h5file.readDataset<Eigen::Tensor<fp64, 4>>(fmt::format("{}/mpo2", group));
    auto                   envL2 = h5file.readDataset<Eigen::Tensor<fp64, 3>>(fmt::format("{}/envL2", group));
    auto                   envR2 = h5file.readDataset<Eigen::Tensor<fp64, 3>>(fmt::format("{}/envR2", group));

    auto ham2 = MatVecMPO<double>(envL2, envR2, mpo2);
    auto dims = ham2.get_shape_mps();

    eig::solver s;
    s.config = config;
    s.config.initial_guess.push_back({mps.data(), 0});

    s.setLogLevel(1);
    s.eigs(ham2);
    auto rnorm       = *std::max_element(s.result.meta.residual_norms.begin(), s.result.meta.residual_norms.end());
    auto msg         = fmt::format(
        FMT_STRING("Result: method {:<32} | {:^10} | nev {:>3} | ncv {:>3} | tol {:8.2e} | iter {:>5} | op {:>5} | mv {:>7} | Found {:<5} | "
                                   "rnorm {:8.2e} | var {:>12.6e} | time {:>10.3f} s | t/op {:>10.3f} ms | t/mv {:>10.3f} ms | dims {}"),
        eig::MethodToString(config.primme_method.value()), group, s.config.maxNev.value(), s.config.maxNcv.value(), s.config.tol.value(),
        s.result.meta.iter, s.result.meta.num_op, s.result.meta.num_mv, s.result.meta.eigvecsR_found, rnorm, s.result.get_eigvals<fp64>()[0],
        s.result.meta.time_total, s.result.meta.time_total * 1000.0 / static_cast<double>(s.result.meta.num_op),
        s.result.meta.time_total * 1000.0 / static_cast<double>(s.result.meta.num_mv), dims);

    tools::log->info("{}", tgt);
    tools::log->info("{}", msg);

    return msg;
}

namespace threading {
    int stl_threads = 1;
    int omp_threads = 1;
}

int main() {
    tools::Logger::setLogger(tools::log, "bench", 0, true);
    auto filename = fmt::format("{}/eigs.h5", BENCH_DATA_DIR);
    if(not h5pp::fs::exists(filename)) {
        tools::log->error("File does not exist: {}", filename);
        exit(0);
    }
// Take care of threading
// Set the number of threads to be used
#if defined(EIGEN_USE_THREADS)
    if(threading::stl_threads <= 0) { threading::stl_threads = (int) std::thread::hardware_concurrency(); }
    tenx::threads::setNumThreads(threading::stl_threads);
    tools::log->info("Using Eigen Tensor with {} C++11 threads", tenx::threads::getNumThreads());
#else
    if(threading::stl_threads > 1)
        tools::log->warn("EIGEN_USE_THREADS is not defined: "
                         "Failed to enable threading in Eigen::Tensor with stl_threads = {}",
                         threading::stl_threads);
#endif

#if defined(_OPENMP)
    if(threading::omp_threads <= 0) { threading::omp_threads = (int) std::thread::hardware_concurrency(); }
    omp_set_num_threads(threading::omp_threads); // Should only need this. Both Eigen (non-Tensor) and MKL listen to this
    //    omp_set_max_active_levels(1);
    tools::log->info("Using OpenMP with {} threads and {} active levels", omp_get_max_threads(), omp_get_max_active_levels());
#endif
    tools::log->info("{}", config::blas::description());

    std::vector<eig::settings> configs(1);
    // https://www.cs.wm.edu/~andreas/software/doc/appendix.html#c.primme_params.eps
    configs[0].tol                         = 1e-12; // 1e-12 is good. This Sets "eps" in primme, see link above.
    configs[0].maxIter                     = 10;    // TODO: Should be thousands
    configs[0].maxNev                      = 1;
    configs[0].maxNcv                      = 4;
    configs[0].maxTime                     = 2 * 60 * 60; // Two hours
    configs[0].lib                         = eig::Lib::PRIMME;
    configs[0].ritz                        = eig::Ritz::SA;
    configs[0].compute_eigvecs             = eig::Vecs::ON;
    configs[0].loglevel                    = 2;
    configs[0].primme_method               = eig::PrimmeMethod::PRIMME_GD_Olsen_plusK; // eig::PrimmeMethod::PRIMME_JDQMR;

    //    configs[0].primme_preconditioner = simps_preconditioner<MatVecMPO<double>>;
    //    configs[0].primme_target_shifts = {0};
    //    configs[0].ritz = eig::Ritz::primme_closest_geq;
    //    configs[0].primme_projection = "primme_proj_refined";

    if(h5pp::fs::exists(filename)) {
        auto                     h5file = h5pp::File(filename, h5pp::FilePermission::READONLY);
        std::vector<std::string> msg;

        for(const auto &group : h5file.findGroups("eigs-")) {
            auto size = h5file.readAttribute<long>(group, "size");
            auto dims = h5file.readAttribute<std::array<long, 3>>(group, "dims");
            tools::log->info("Running group: {} | size {} | dims {}", group, size, dims);
            if(size > 1024) {
                tools::log->warn("Skipping large problem. Comment this line to run more benchmarks.");
                continue;
            }
            for(const auto &config : configs) { msg.push_back(solve(config, h5file, group)); }
        }
        tools::log->debug("Result summary:");
        for(auto &m : msg) tools::log->debug("{}", m);
    }
}
