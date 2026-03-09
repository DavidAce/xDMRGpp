
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#ifdef _OPENMP
    #include <omp.h>
#endif
#include <math/svd.h>
#include <math/tenx.h>
#include <tid/tid.h>

#ifdef OPENBLAS_AVAILABLE
    #include <cblas.h>
    #include <openblas_config.h>
#endif

#ifdef MKL_AVAILABLE
    #include <mkl.h>
    #include <mkl_service.h>
#endif
#include <Eigen/Core>
#include <h5pp/h5pp.h>
#include <math/linalg/matrix/to_string.h>
#include <math/svd.h>
#include <thread>

void test() {
    svd::config svd_settings;
    svd_settings.svd_save = svd::save::NONE;
    svd_settings.loglevel = 2;
    auto filename         = fmt::format("{}/svd-save.h5", BENCH_DATA_DIR);
    if(h5pp::fs::exists(filename)) {
        h5pp::File h5file(filename, h5pp::FilePermission::READONLY, 2);
        for(const auto &svd_group : h5file.findGroups("svd_lapacke")) {
            auto U_original               = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("{}/U", svd_group));
            auto S_original               = h5file.readDataset<Eigen::VectorXcd>(fmt::format("{}/S", svd_group));
            auto V_original               = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("{}/VT", svd_group));
            auto A                        = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("{}/A", svd_group));
            auto rows                     = A.rows();
            auto cols                     = A.cols();
            auto rank_max                 = h5file.readAttribute<long>("rank_max", svd_group);
            svd_settings.truncation_limit = h5file.readAttribute<double>("threshold", svd_group);
            Eigen::MatrixXcd S_mat(rank_max, 4);
            S_mat.setZero();
            {
                svd_settings.svd_lib  = svd::lib::eigen;
                svd_settings.rank_max = rank_max;
                svd::solver svd(svd_settings);
                auto        t_eigen = tid::tic_scope("eigen");
                auto [U, S, V]      = svd.do_svd(A);
                t_eigen.toc();
                //            fmt::print("S\n{}\n", linalg::matrix::to_string(S, 16));
                S_mat.col(0).head(S.size()) = S;
            }
            {
                svd_settings.svd_lib = svd::lib::lapacke;
                svd::solver svd(svd_settings);
                auto        t_lapack = tid::tic_scope("lapack");
                auto [U, S, V]       = svd.do_svd(A);
                t_lapack.toc();
                //            fmt::print("S\n{}\n", linalg::matrix::to_string(S, 16));
                S_mat.col(1).head(S.size()) = S;
            }
            {
                svd_settings.svd_lib  = svd::lib::lapacke;
                svd_settings.svd_rtn  = svd::rtn::gersvd;
                svd_settings.rank_max = rank_max / 10;
                svd::solver svd(svd_settings);
                auto        t_rsvd = tid::tic_scope("rsvd");

                auto [U, S, V] = svd.do_svd(A);
                t_rsvd.toc();
                //            fmt::print("S\n{}\n", linalg::matrix::to_string(S, 16));
                S_mat.col(2).head(S.size()) = S;
            }

            S_mat.col(3)  = (S_mat.col(2) - S_mat.col(0)).cwiseAbs().eval();
            double S_diff = S_mat.col(3).sum().real();
            fmt::print("A {:>4} x {:>4} | rank_max {:4} | eigen {:8.2e} s | lapack {:8.2e} s | rsvd {:8.2e} s | diff {:20.16f}\n", rows, cols, rank_max,
                       tid::get("eigen").get_last_interval(), tid::get("lapack").get_last_interval(), tid::get("rsvd").get_last_interval(), S_diff);
        }
        fmt::print("Total time | eigen {:8.2e} s | lapack {:8.2e} s | rsvd {:8.2e}\n", tid::get("eigen").get_time(), tid::get("lapack").get_time(),
                   tid::get("rsvd").get_time());
    }
}

TEST_CASE("Singular value decomposition in Eigen and Lapacke", "[svd]") {
    SECTION("Bench split functions") {
        svd::config svd_settings;
        svd_settings.truncation_limit = 1e-14;
        svd_settings.loglevel         = 0;
        svd_settings.svd_rtn          = svd::rtn::gejsv;
        svd_settings.svd_save         = svd::save::NONE;
        svd_settings.svd_lib          = svd::lib::lapacke;
        auto filename                 = fmt::format("{}/svd-failed.h5", BENCH_DATA_DIR);
        if(h5pp::fs::exists(filename)) {
            h5pp::File h5file(filename, h5pp::FilePermission::READONLY, 2);
            for(const auto &svd_group : h5file.findGroups("svd_")) {
                auto U_original = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("{}/U", svd_group));
                auto S_original = h5file.readDataset<Eigen::VectorXcd>(fmt::format("{}/S", svd_group));
                auto V_original = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("{}/V", svd_group));
                auto A          = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("{}/A", svd_group));
                fmt::print("S original \n{}\n", linalg::matrix::to_string(S_original, 16));

                svd_settings.rank_max         = h5file.readAttribute<long>("rank_max", svd_group);
                svd_settings.truncation_limit = h5file.readAttribute<long>("threshold", svd_group);

                svd::solver svd(svd_settings);
                auto [U, S, V] = svd.do_svd(A);
                fmt::print("S\n{}\n", linalg::matrix::to_string(S, 16));
            }
        }
    }
}
namespace threading {
    int stl_threads = 4;
    int omp_threads = 4;
}
int main(int argc, char **argv) {
    tools::Logger::setLogger(tools::log, "testfail", 0, true);
    auto filename = fmt::format("{}/svd-save.h5", BENCH_DATA_DIR);
    if(not h5pp::fs::exists(filename)) {
        tools::log->error("File does not exist: {}", filename);
        exit(0);
    }
// Take care of threading
// Set the number of threads to be used
#if defined(EIGEN_USE_THREADS)
    if(threading::stl_threads <= 0) { threading::stl_threads = (int) std::thread::hardware_concurrency(); }
    tenx::threads::setNumThreads(threading::stl_threads);
    tools::log->info("Eigen3 Tensor | stl threads {}", tenx::threads::getNumThreads());
#else
    if(threading::stl_threads > 1)
        tools::log->warn("EIGEN_USE_THREADS is not defined: "
                         "Failed to enable threading in Eigen::Tensor with stl_threads = {}",
                         threading::stl_threads);
#endif

#if defined(_OPENMP)
    if(threading::omp_threads <= 0) { threading::omp_threads = (int) std::thread::hardware_concurrency(); }
    omp_set_num_threads(threading::omp_threads); // Should only need this. Both Eigen (non-Tensor) and MKL listen to this
    tools::log->info("OpenMP | threads {} | max active levels {}", omp_get_max_threads(), omp_get_max_active_levels());
#endif

#if defined(OPENBLAS_AVAILABLE)
    auto        openblas_parallel_mode = openblas_get_parallel();
    std::string openblas_parallel_str;
    if(openblas_parallel_mode == 0) openblas_parallel_str = "seq";
    if(openblas_parallel_mode == 1) openblas_parallel_str = "threads";
    if(openblas_parallel_mode == 2) openblas_parallel_str = "openmp";
    if(openblas_parallel_mode == 1) openblas_set_num_threads(threading::omp_threads); // Use the omp_threads level for blas-related threading
    tools::log->info("{} threads {} | parallel mode [{}:{}] | core type {} | config {} | multithread threshold {}", OPENBLAS_VERSION,
                     openblas_get_num_threads(), openblas_parallel_mode, openblas_parallel_str, openblas_get_corename(), openblas_get_config(),
                     OPENBLAS_GEMM_MULTITHREAD_THRESHOLD);
#endif
#if defined(MKL_AVAILABLE)
    tools::log->info("Intel MKL | threads {}", mkl_get_max_threads());
#endif

#if defined(EIGEN_USE_MKL_ALL)
    tools::log->info("Eigen3 | threads {} | EIGEN_USE_MKL_ALL", Eigen::nbThreads());
#elif defined(EIGEN_USE_BLAS)
    tools::log->info("Eigen3 | threads {} | EIGEN_USE_BLAS", Eigen::nbThreads());
#else
    tools::log->info("Eigen3 | threads {} | No BLAS backend", Eigen::nbThreads());
#endif
    test();
    return 0;
    return Catch::Session().run(argc, argv);
}