cmake_minimum_required(VERSION 3.24)

# Derive one intentional BLAS backend from BLA_VENDOR and expose a compact,
# compile-time description string. Runtime thread control and runtime vendor
# introspection are intentionally left to OpenMP and the user's BLAS setup.

set(DMRG_BLAS_BACKEND_NAME "generic")
set(DMRG_BLAS_BACKEND_VENDOR "${BLA_VENDOR}")
if(NOT DMRG_BLAS_BACKEND_VENDOR)
    set(DMRG_BLAS_BACKEND_VENDOR "unspecified")
endif()

if(DEFINED BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "")
    if(BLA_VENDOR MATCHES "^Intel")
        set(DMRG_BLAS_BACKEND_NAME "mkl")
    elseif(BLA_VENDOR MATCHES "OpenBLAS")
        set(DMRG_BLAS_BACKEND_NAME "openblas")
    elseif(BLA_VENDOR MATCHES "FlexiBLAS")
        set(DMRG_BLAS_BACKEND_NAME "flexiblas")
    endif()
endif()

set(DMRG_BLAS_BACKEND_DESCRIPTION "BLAS backend [${DMRG_BLAS_BACKEND_NAME}] | vendor ${DMRG_BLAS_BACKEND_VENDOR}")
message(STATUS "Configured BLAS backend: ${DMRG_BLAS_BACKEND_NAME}")
