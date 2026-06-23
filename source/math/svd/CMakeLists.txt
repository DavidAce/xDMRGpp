set(svd_lapack_sources)
dmrg_append_if_scalar_enabled(svd_lapack_sources fp32
        lapack_wrappers/sgesvd.cpp
        lapack_wrappers/sgesvj.cpp
        lapack_wrappers/sgejsv.cpp
        lapack_wrappers/sgesdd.cpp
        lapack_wrappers/sgesvdx.cpp
        )
dmrg_append_if_scalar_enabled(svd_lapack_sources fp64
        lapack_wrappers/dgesvd.cpp
        lapack_wrappers/dgesvj.cpp
        lapack_wrappers/dgejsv.cpp
        lapack_wrappers/dgesdd.cpp
        lapack_wrappers/dgesvdx.cpp
        )
dmrg_append_if_scalar_enabled(svd_lapack_sources cx32
        lapack_wrappers/cgesvd.cpp
        lapack_wrappers/cgesvj.cpp
        lapack_wrappers/cgejsv.cpp
        lapack_wrappers/cgesdd.cpp
        lapack_wrappers/cgesvdx.cpp
        )
dmrg_append_if_scalar_enabled(svd_lapack_sources cx64
        lapack_wrappers/zgesvd.cpp
        lapack_wrappers/zgesvj.cpp
        lapack_wrappers/zgejsv.cpp
        lapack_wrappers/zgesdd.cpp
        lapack_wrappers/zgesvdx.cpp
        )
dmrg_filter_enabled_scalars(svd_lapack_scalars fp32 fp64 cx32 cx64)

if(svd_lapack_scalars)
    add_source_target(xdmrg++-math-svd-lapack
            TARGET_SOURCES
            ${svd_lapack_sources}
            INST_TEMPLATE_FILES
            inst/svd_lapack.inst.cpp.in
            INST_SCALARS
            ${svd_lapack_scalars}
            OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
            INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            )
else()
    add_source_target(xdmrg++-math-svd-lapack
            OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
            INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            )
endif()

add_source_target(xdmrg++-math-svd
        TARGET_SOURCES
        svd.cpp
        INST_TEMPLATE_FILES
        inst/svd.inst.cpp.in
        inst/svd_dump.inst.cpp.in
        inst/svd_eigen.inst.cpp.in
        inst/svd_rsvd.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config xdmrg++-tid xdmrg++-math-svd-lapack
        )
