#include "blas_backend.h"

/**
 * @file
 * Implementation of the BLAS backend abstraction declared in
 * config/blas_backend.h.
 *
 * The rest of the project calls config::blas::* without needing to know which
 * vendor backend is active. This file is the only place that sees the backend
 * preprocessor selection from SetupBlasBackend.cmake.
 */

#include <string>
#include <string_view>

#if defined(DMRG_BLAS_BACKEND_MKL)
    #include <mkl_service.h>
#elif defined(DMRG_BLAS_BACKEND_OPENBLAS)
    #include <openblas/cblas.h>
    #include <openblas/openblas_config.h>
#elif defined(DMRG_BLAS_BACKEND_FLEXIBLAS)
    #include <cstdio>
    #include <flexiblas/flexiblas_api.h>
#elif defined(DMRG_BLAS_BACKEND_GENERIC)
    #if __has_include(<cblas.h>)
        #include <cblas.h>
    #endif
#else
    #error "No DMRG_BLAS_BACKEND_* macro defined"
#endif

namespace config::blas {
    std::string_view backend_name() {
#if defined(DMRG_BLAS_BACKEND_MKL)
        return "mkl";
#elif defined(DMRG_BLAS_BACKEND_OPENBLAS)
        return "openblas";
#elif defined(DMRG_BLAS_BACKEND_FLEXIBLAS)
        return "flexiblas";
#else
        return "generic";
#endif
    }

    void set_num_threads(int num_threads) {
        // Some backends expose thread setters while others are effectively
        // controlled externally or do not offer a stable runtime API here.
#if defined(DMRG_BLAS_BACKEND_MKL)
        mkl_set_num_threads(num_threads);
#elif defined(DMRG_BLAS_BACKEND_OPENBLAS)
        openblas_set_num_threads(num_threads);
#elif defined(DMRG_BLAS_BACKEND_FLEXIBLAS)
        flexiblas_set_num_threads(num_threads);
#else
        (void) num_threads;
#endif
    }

    std::string description() {
        // Keep this concise and single-line so it is suitable for logs,
        // benchmark banners, and metadata written to HDF5.
#if defined(DMRG_BLAS_BACKEND_MKL)
        MKLVersion version {};
        mkl_get_version(&version);
        return "BLAS backend [mkl] | version " + std::to_string(version.MajorVersion) + "." + std::to_string(version.MinorVersion) + "." +
               std::to_string(version.UpdateVersion) + " | threads " + std::to_string(mkl_get_max_threads());
#elif defined(DMRG_BLAS_BACKEND_OPENBLAS)
        std::string parallel_mode = "unknown";
        switch(openblas_get_parallel()) {
            case 0: parallel_mode = "sequential"; break;
            case 1: parallel_mode = "threads"; break;
            case 2: parallel_mode = "openmp"; break;
            default: break;
        }
        return std::string("BLAS backend [openblas] | threads ") + std::to_string(openblas_get_num_threads()) + " | parallel " + parallel_mode +
               " | core " + openblas_get_corename() + " | config " + openblas_get_config();
#elif defined(DMRG_BLAS_BACKEND_FLEXIBLAS)
        char buffer[64] = {0};
        int  size       = flexiblas_current_backend(buffer, 64);
        auto backend    = size > 0 ? std::string_view(buffer) : std::string_view("unknown");
        return "BLAS backend [flexiblas] | active " + std::string(backend) + " | threads " + std::to_string(flexiblas_get_num_threads());
#else
        return "BLAS backend [generic]";
#endif
    }
}
