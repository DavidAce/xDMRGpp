
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#ifdef _OPENMP
    #include <omp.h>
#endif
#include "config/blas_backend.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tid/tid.h"
#include "tools/common/split.h"
#include "math/svd.h"
#include <Eigen/Core>
#include <h5pp/h5pp.h>
#include <thread>

TEST_CASE("Singular value decomposition in Eigen and Lapacke", "[svd]") {
    SECTION("Bench split functions") {
        svd::config svd_settings;
        svd_settings.truncation_limit = 1e-16;
        svd_settings.loglevel         = 2;
        auto filename                 = fmt::format("{}/svd-benchmark.h5", BENCH_DATA_DIR);
        if(h5pp::fs::exists(filename)) {
            h5pp::File h5file(filename, h5pp::FilePermission::READONLY, 2);
            double     t_split_sum = 0;
            for(const auto &multisite_tensor_name : h5file.findDatasets("multisite_tensor_cplx")) {
                auto dims = h5file.getDatasetDimensions(multisite_tensor_name);
                auto size = dims[0] * dims[1] * dims[2];
                if(size < 131072) continue;

                auto multisite_tensor = h5file.readDataset<Eigen::Tensor<std::complex<double>, 3>>(multisite_tensor_name);

                auto spin_dims        = h5file.readAttribute<std::vector<long>>("spin_dims", multisite_tensor_name);
                auto positions        = h5file.readAttribute<std::vector<size_t>>("positions", multisite_tensor_name);
                auto center_position  = h5file.readAttribute<long>("center_position", multisite_tensor_name);
                auto chi_limit        = h5file.readAttribute<long>("chi_limit", multisite_tensor_name);
                auto t_split          = h5file.readAttribute<double>("t_split", multisite_tensor_name);
                svd_settings.rank_max = chi_limit;
                t_split_sum += t_split;
                {
                    svd_settings.svd_lib = svd::lib::eigen;
                    svd_settings.svd_rtn = svd::rtn::gersvd;
                    auto t_eig           = tid::tic_scope("eig");
                    auto mps_list_eig    = tools::common::split::split_mps<cx64>(multisite_tensor, spin_dims, positions, center_position, svd_settings);
                }
                {
                    svd_settings.svd_lib = svd::lib::lapack;
                    auto t_eig           = tid::tic_scope("lpk");
                    auto mps_list_lpk    = tools::common::split::split_mps<cx64>(multisite_tensor, spin_dims, positions, center_position, svd_settings);
                }

                // Test what happens if center_position is now the same as positions.front()
                center_position = static_cast<long>(positions.front());
                {
                    svd_settings.svd_lib = svd::lib::eigen;
                    svd_settings.svd_rtn = svd::rtn::gersvd;
                    auto t_eig           = tid::tic_scope("eigA");
                    auto mps_list_eig    = tools::common::split::split_mps<cx64>(multisite_tensor, spin_dims, positions, center_position, svd_settings);
                }
                {
                    svd_settings.svd_lib = svd::lib::lapack;
                    auto t_eig           = tid::tic_scope("lpkA");
                    auto mps_list_lpk    = tools::common::split::split_mps<cx64>(multisite_tensor, spin_dims, positions, center_position, svd_settings);
                }

                // Test what happens if center_position is now the same as positions.back()
                center_position = static_cast<long>(positions.back());
                {
                    svd_settings.svd_lib = svd::lib::eigen;
                    svd_settings.svd_rtn = svd::rtn::gersvd;
                    auto t_eig           = tid::tic_scope("eigB");
                    auto mps_list_eig    = tools::common::split::split_mps<cx64>(multisite_tensor, spin_dims, positions, center_position, svd_settings);
                }
                {
                    svd_settings.svd_lib = svd::lib::lapack;
                    auto t_eig           = tid::tic_scope("lpkB");
                    auto mps_list_lpk    = tools::common::split::split_mps<cx64>(multisite_tensor, spin_dims, positions, center_position, svd_settings);
                }

                tools::log->info("eig +{:8.2e} {:8.2e} | lpk +{:8.2e} {:8.2e} | eigA +{:8.2e} {:8.2e} | lpkA +{:8.2e} {:8.2e} | eigB +{:8.2e} {:8.2e} | lpkB "
                                 "+{:8.2e} {:8.2e} | original +{:8.2e} {:8.2e} | {:<32} | dims {}",
                                 tid::get_unscoped("eig.split").get_last_interval(), tid::get_unscoped("eig.split").get_time(),
                                 tid::get_unscoped("lpk.split").get_last_interval(), tid::get_unscoped("lpk.split").get_time(),
                                 tid::get_unscoped("eigA.split").get_last_interval(), tid::get_unscoped("eigA.split").get_time(),
                                 tid::get_unscoped("lpkA.split").get_last_interval(), tid::get_unscoped("lpkA.split").get_time(),
                                 tid::get_unscoped("eigB.split").get_last_interval(), tid::get_unscoped("eigB.split").get_time(),
                                 tid::get_unscoped("lpkB.split").get_last_interval(), tid::get_unscoped("lpkB.split").get_time(), t_split, t_split_sum,
                                 multisite_tensor_name, dims);
            }
            tools::log->info("eig {:8.2e} | lpk {:8.2e} | eigA {:8.2e} | lpkA {:8.2e} | eigB {:8.2e} | lpkB {:8.2e} | original {:8.2e} | TOTAL",
                             tid::get_unscoped("eig.split").get_time(), tid::get_unscoped("lpk.split").get_time(), tid::get_unscoped("eigA.split").get_time(),
                             tid::get_unscoped("lpkA.split").get_time(), tid::get_unscoped("eigB.split").get_time(), tid::get_unscoped("lpkB.split").get_time(),

                             t_split_sum);
            for(const auto &t : tid::get_tree()) tools::log->info("{}", t.str());
        }
    }
}
namespace threading {
    int stl_threads = 4;
    int omp_threads = 4;
}
int main(int argc, char **argv) {
    tools::Logger::setLogger(tools::log, "bench", 0, true);
    auto filename = fmt::format("{}/svd-benchmark.h5", BENCH_DATA_DIR);
    if(not h5pp::fs::exists(filename)) {
        tools::log->error("File does not exist: {}", filename);
        exit(0);
    }
// Take care of threading
// Set the number of threads to be used
#if defined(EIGEN_USE_THREADS)
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

    config::blas::set_num_threads(threading::omp_threads);
    tools::log->info("{}", config::blas::description());

#if defined(EIGEN_USE_MKL_ALL)
    tools::log->info("Eigen3 | threads {} | EIGEN_USE_MKL_ALL", Eigen::nbThreads());
#elif defined(EIGEN_USE_BLAS)
    tools::log->info("Eigen3 | threads {} | EIGEN_USE_BLAS", Eigen::nbThreads());
#else
    tools::log->info("Eigen3 | threads {} | No BLAS backend", Eigen::nbThreads());
#endif

    return Catch::Session().run(argc, argv);
}
