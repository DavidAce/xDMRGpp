include(CheckLanguage)
check_language(Fortran)
if(NOT CMAKE_Fortran_COMPILER)
    message(FATAL_ERROR "A Fortran compiler is required to build/install arpack-ng and generate the LAPACK Fortran interface. "
                        "Install gfortran or set FC/CMAKE_Fortran_COMPILER to a working Fortran compiler.")
endif()
enable_language(Fortran)

include(FortranCInterface)

set(DMRG_LAPACK_FORTRAN_SYMBOLS_HEADER "${CMAKE_BINARY_DIR}/source/math/lapack/lapack_fortran_symbols.h")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/source/math/lapack")

FortranCInterface_HEADER(
        "${DMRG_LAPACK_FORTRAN_SYMBOLS_HEADER}"
        MACRO_NAMESPACE "DMRG_FC_"
        SYMBOL_NAMESPACE "DMRG_LAPACK_"
        SYMBOLS
        sgeev
        dgeev
        cgeev
        zgeev
        ssyevd
        dsyevd
        cheev
        zheev
        cheevd
        zheevd
        ssyevr
        dsyevr
        cheevr
        zheevr
        ssyevx
        dsyevx
        ssygvd
        dsygvd
        ssygvx
        dsygvx
        sgesvd
        dgesvd
        cgesvd
        zgesvd
        sgesvj
        dgesvj
        cgesvj
        zgesvj
        sgejsv
        dgejsv
        cgejsv
        zgejsv
        sgesdd
        dgesdd
        cgesdd
        zgesdd
        sgesvdx
        dgesvdx
        cgesvdx
        zgesvdx
        slamch
        dlamch
)

target_include_directories(xdmrg++-flags INTERFACE "${CMAKE_BINARY_DIR}/source")
