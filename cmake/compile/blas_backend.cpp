/**
 * @file
 * Configure-time BLAS backend probe used by SetupBlasBackend.cmake.
 *
 * This file is compiled with `try_compile()` to verify that the selected
 * backend macro, vendor headers, and imported BLAS target are mutually
 * consistent before the real project build starts.
 *
 * Each branch touches one backend-specific API symbol so configuration fails
 * early if headers are missing or the selected backend does not match the
 * advertised include path layout.
 */

#if defined(DMRG_BLAS_BACKEND_MKL)
    #include <mkl_service.h>
int main() {
    return mkl_get_max_threads() < 0;
}
#elif defined(DMRG_BLAS_BACKEND_OPENBLAS)
    #include <openblas/cblas.h>
    #include <openblas/openblas_config.h>
int main() {
    return openblas_get_num_threads() < 0;
}
#elif defined(DMRG_BLAS_BACKEND_FLEXIBLAS)
    #include <cstdio>
    #include <flexiblas/flexiblas_api.h>
int main() {
    return flexiblas_get_num_threads() < 0;
}
#elif defined(DMRG_BLAS_BACKEND_GENERIC)
    #if __has_include(<cblas.h>)
        #include <cblas.h>
    #endif
int main() {
    return 0;
}
#else
    #error "No DMRG_BLAS_BACKEND_* macro selected"
#endif
