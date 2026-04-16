#include "blas_backend.h"

/**
 * @file
 * Implementation of the configured BLAS backend metadata declared in
 * config/blas_backend.h.
 */

#include "config/blas_backend_info.h"
#include <string>
#include <string_view>

#ifndef DMRG_BLAS_BACKEND_NAME
    #error "DMRG_BLAS_BACKEND_NAME is not defined"
#endif

#ifndef DMRG_BLAS_BACKEND_DESCRIPTION
    #error "DMRG_BLAS_BACKEND_DESCRIPTION is not defined"
#endif

namespace config::blas {
    std::string_view backend_name() { return DMRG_BLAS_BACKEND_NAME; }

    std::string description() { return DMRG_BLAS_BACKEND_DESCRIPTION; }
}
