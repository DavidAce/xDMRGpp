#include "threading.h"
#include "blas_backend.h"
#include "debug/exceptions.h"
#include "math/tenx/threads.h"
#include "settings.h"
#include "tools/common/log.h"
#include <cstdlib>
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include <optional>

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

        tools::log->info("OpenMP | omp_max_threads {} | omp_max_active_levels {} | omp_dynamic {} | omp_num_procs {}", omp_get_max_threads(),
                         omp_get_max_active_levels(), omp_get_dynamic(), omp_get_num_procs());

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
        if(auto envcoretype = get_env("OPENBLAS_CORETYPE"); envcoretype)
            tools::log->info("Detected environment variable: OPENBLAS_CORETYPE={}", envcoretype.value());
        config::blas::set_num_threads(static_cast<int>(settings::threading::num_threads));
        tools::log->info("{}", config::blas::description());
        if(settings::threading::show_threads) exit(0);
    }
}
