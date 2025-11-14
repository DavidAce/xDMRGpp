#include "threading.h"
#include "math/tenx/threads.h"
#include "settings.h"
#include "tools/common/log.h"
#include <cstdlib>
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include <optional>

#if defined(MKL_AVAILABLE)
    #include <mkl_service.h>
#elif defined(OPENBLAS_AVAILABLE)
    #include <openblas/cblas.h>
    #include <openblas/openblas_config.h>
#elif defined(FLEXIBLAS_AVAILABLE)
    #include <flexiblas/flexiblas_api.h>
#elif __has_include(<cblas.h>) && __has_include(<openblas_config.h>)
    #include <cblas.h>
#endif

namespace settings {

    inline std::optional<std::string> get_env(std::string_view key) {
        if(key.empty()) throw std::invalid_argument("Value requested for the empty-name environment variable");
        const char *ev_val = std::getenv(key.data());
        if(ev_val == nullptr) return std::nullopt;
        return std::string(ev_val);
    }

    void configure_threads() {
        // Set the number of threads to be used
//        unsigned int omp_threads = 1;
#if defined(_OPENMP) and defined(EIGEN_USE_THREADS)
        auto get_omp_proc_bind = []() -> std::string {
            switch(omp_get_proc_bind()) {
                case 0: return "false";
                case 1: return "true";
                case 2: return "primary";
                case 3: return "close";
                case 4: return "spread";
                default: return "unknown";
            }
        };
        if(auto omp_proc_bind = get_omp_proc_bind(); omp_proc_bind != "false") {
            throw except::runtime_error("\n \t Detected OMP_PROC_BIND: {}.\n"
                                        "\t OpenMP core pinning interacts poorly with std::thread in Eigen::Tensor when EIGEN_USE_THREADS is defined.\n"
                                        "\t Please unset environment variables OMP_PROC_BIND and OMP_PLACES, or unset preprocessor variable EIGEN_USE_THREADS",
                                        omp_proc_bind);
        }



        tools::log->info("OpenMP | omp_max_threads {} | omp_max_active_levels {} | omp_dynamic {} | omp_num_procs {}",
                         omp_get_max_threads(), omp_get_max_active_levels(), omp_get_dynamic(), omp_get_num_procs());

//        omp_threads = safe_cast<unsigned int>(omp_get_max_threads());
#endif
        std::string eigen_msg;
#if defined(EIGEN_USE_MKL_ALL)
        eigen_msg.append(" | EIGEN_USE_MKL_ALL");
#endif
#if defined(EIGEN_USE_BLAS)
        eigen_msg.append(" | EIGEN_USE_BLAS");
#endif
#if defined(EIGEN_USE_THREADS)
        eigen_msg.append(" | EIGEN_USE_THREADS");
        unsigned int cxx11_threads = settings::threading::num_threads;
        //        if (omp_threads <= 1) stl_threads = std::clamp(settings::threading::num_threads, stl_threads, settings::threading::max_threads);
        //        else if(settings::threading::num_threads > omp_threads){
        //            stl_threads = std::clamp(settings::threading::num_threads - omp_threads, stl_threads, settings::threading::max_threads);
        //        }
        tenx::threads::setNumThreads(cxx11_threads);
#else
        if(settings::threading::num_threads > 1)
            tools::log->warn("EIGEN_USE_THREADS is not defined: "
                             "Failed to enable threading in Eigen::Tensor with stl_threads = {}",
                             settings::threading::num_threads);
#endif
        tools::log->info("Eigen3 | omp_threads {} | cxx11_threads {} | max_threads {}{}", Eigen::nbThreads(), tenx::threads::getNumThreads(),
                         settings::threading::max_threads, eigen_msg);
#if defined(OPENBLAS_AVAILABLE)
        auto envcoretype = get_env("OPENBLAS_CORETYPE");
        if(envcoretype) tools::log->info("Detected environment variable: OPENBLAS_CORETYPE={}", envcoretype.value());
        auto        openblas_parallel_mode = openblas_get_parallel();
        std::string openblas_parallel_str;
        if(openblas_parallel_mode == 0) openblas_parallel_str = "sequential";
        if(openblas_parallel_mode == 1) openblas_parallel_str = "threads";
        if(openblas_parallel_mode == 2) openblas_parallel_str = "openmp";
        if(openblas_parallel_mode == 1) openblas_set_num_threads(safe_cast<int>(num_threads)); // Use this for blas-related threading
        tools::log->info("{} NUM_THREADS={} | parallel_mode={} | corename={}", openblas_get_config(), openblas_get_num_threads(), openblas_parallel_str,
                         openblas_get_corename());
#endif
#if defined(MKL_AVAILABLE)
        tools::log->info("Intel MKL | max_threads {}", mkl_get_max_threads());
#endif
#if defined(FLEXIBLAS_AVAILABLE)

        char buffer[32] = {0};
        int  size       = flexiblas_current_backend(buffer, 32);
        if(size > 0) {
            tools::log->info("Flexiblas backend [{}] | num_threads {}", buffer, flexiblas_get_num_threads());
        } else {
            tools::log->info("Flexiblas backend read failed: size {}", size);
        }

#endif
        if(settings::threading::show_threads) exit(0);
    }
}