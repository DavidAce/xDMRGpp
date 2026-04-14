cmake_minimum_required(VERSION 3.24)

# Derive one intentional BLAS backend from BLA_VENDOR and validate that the
# matching vendor headers are available. The runtime/backend-specific logic is
# intentionally confined to source/config/blas_backend.cpp; this file only
# decides which backend implementation should be compiled.
#
# This file assumes find_package(BLAS REQUIRED) has already run. Since the
# project minimum CMake version is 3.24 and FindBLAS has provided the imported
# target BLAS::BLAS since 3.18, the imported target is expected to exist here.

include(${CMAKE_CURRENT_LIST_DIR}/CheckCompile.cmake)

set(DMRG_BLAS_BACKEND_DEFINE DMRG_BLAS_BACKEND_GENERIC)
set(DMRG_BLAS_BACKEND_NAME "generic")
set(DMRG_BLAS_BACKEND_INCLUDE_DIRS)

if(DEFINED BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "")
    if(BLA_VENDOR MATCHES "^Intel")
        set(DMRG_BLAS_BACKEND_DEFINE DMRG_BLAS_BACKEND_MKL)
        set(DMRG_BLAS_BACKEND_NAME "mkl")
    elseif(BLA_VENDOR MATCHES "OpenBLAS")
        set(DMRG_BLAS_BACKEND_DEFINE DMRG_BLAS_BACKEND_OPENBLAS)
        set(DMRG_BLAS_BACKEND_NAME "openblas")
    elseif(BLA_VENDOR MATCHES "FlexiBLAS")
        set(DMRG_BLAS_BACKEND_DEFINE DMRG_BLAS_BACKEND_FLEXIBLAS)
        set(DMRG_BLAS_BACKEND_NAME "flexiblas")
    endif()
endif()

# Vendor libraries and vendor headers are not discovered by the same CMake
# machinery. Use BLASROOT/CMAKE_PREFIX_PATH to locate the headers that the
# selected backend implementation needs. This keeps backend selection
# intentional instead of inferring it from whichever headers happen to be
# visible on the host.
if(NOT DMRG_BLAS_BACKEND_NAME STREQUAL "generic")
    set(_dmrg_blas_header_name)
    set(_dmrg_blas_header_suffixes include)
    if(DMRG_BLAS_BACKEND_NAME STREQUAL "mkl")
        set(_dmrg_blas_header_name mkl_service.h)
    elseif(DMRG_BLAS_BACKEND_NAME STREQUAL "openblas")
        set(_dmrg_blas_header_name openblas/cblas.h)
    elseif(DMRG_BLAS_BACKEND_NAME STREQUAL "flexiblas")
        set(_dmrg_blas_header_name flexiblas/flexiblas_api.h)
    endif()

    find_path(DMRG_BLAS_BACKEND_INCLUDE_DIR
              NAMES ${_dmrg_blas_header_name}
              PATHS ${BLASROOT} ${CMAKE_PREFIX_PATH}
              PATH_SUFFIXES ${_dmrg_blas_header_suffixes}
              NO_DEFAULT_PATH)
    if(NOT DMRG_BLAS_BACKEND_INCLUDE_DIR)
        message(FATAL_ERROR "Could not find headers for BLAS backend [${DMRG_BLAS_BACKEND_NAME}] using BLASROOT='${BLASROOT}'")
    endif()
    list(APPEND DMRG_BLAS_BACKEND_INCLUDE_DIRS "${DMRG_BLAS_BACKEND_INCLUDE_DIR}")
endif()

# Compile a tiny backend-specific probe to fail early during configure if the
# chosen backend cannot be compiled as configured.
check_compile(blas_backend BLAS::BLAS REQUIRED
              COMPILE_DEFINITIONS ${DMRG_BLAS_BACKEND_DEFINE}
              INCLUDE_DIRECTORIES ${DMRG_BLAS_BACKEND_INCLUDE_DIRS})
message(STATUS "Configured BLAS backend: ${DMRG_BLAS_BACKEND_NAME}")
