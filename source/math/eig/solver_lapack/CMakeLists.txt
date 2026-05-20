set(eig_lapack_sources)
dmrg_append_if_scalar_enabled(eig_lapack_sources fp32
        sgeev.cpp
        ssyevd.cpp
        ssyevx.cpp
        ssyevr.cpp
        ssygvd.cpp
        ssygvx.cpp
        )
dmrg_append_if_scalar_enabled(eig_lapack_sources fp64
        dgeev.cpp
        dsyevd.cpp
        dsyevx.cpp
        dsyevr.cpp
        dsygvd.cpp
        dsygvx.cpp
        )
dmrg_append_if_scalar_enabled(eig_lapack_sources fp128
        qgeev_eigen.cpp
        qsyevd_eigen.cpp
        qsygvd_eigen.cpp
        )
dmrg_append_if_scalar_enabled(eig_lapack_sources cx32
        cgeev.cpp
        cheevd.cpp
        cheevr.cpp
        )
dmrg_append_if_scalar_enabled(eig_lapack_sources cx64
        zgeev.cpp
        zheevd.cpp
        zheevr.cpp
        )
dmrg_append_if_scalar_enabled(eig_lapack_sources cx128
        wgeev_eigen.cpp
        wheevd_eigen.cpp
        whegvd_eigen.cpp
        )

add_source_target(xdmrg++-math-eig-lapack
        TARGET_SOURCES
        ${eig_lapack_sources}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-math-eig-o
        )
