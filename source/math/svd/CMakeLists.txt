add_source_target(xdmrg++-math-svd-lapacke
        TARGET_SOURCES
        lapack_wrappers/sgesvd.cpp
        lapack_wrappers/dgesvd.cpp
        lapack_wrappers/cgesvd.cpp
        lapack_wrappers/zgesvd.cpp
        lapack_wrappers/sgesvj.cpp
        lapack_wrappers/dgesvj.cpp
        lapack_wrappers/cgesvj.cpp
        lapack_wrappers/zgesvj.cpp
        lapack_wrappers/sgejsv.cpp
        lapack_wrappers/dgejsv.cpp
        lapack_wrappers/cgejsv.cpp
        lapack_wrappers/zgejsv.cpp
        lapack_wrappers/sgesdd.cpp
        lapack_wrappers/dgesdd.cpp
        lapack_wrappers/cgesdd.cpp
        lapack_wrappers/zgesdd.cpp
        lapack_wrappers/sgesvdx.cpp
        lapack_wrappers/dgesvdx.cpp
        lapack_wrappers/cgesvdx.cpp
        lapack_wrappers/zgesvdx.cpp
        INST_TEMPLATE_FILES
        inst/svd_lapacke.inst.cpp.in
        INST_SCALARS
        fp32 fp64 cx32 cx64
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        )

add_source_target(xdmrg++-math-svd
        TARGET_SOURCES
        svd.cpp
        INST_TEMPLATE_FILES
        inst/svd.inst.cpp.in
        inst/svd_dump.inst.cpp.in
        inst/svd_eigen.inst.cpp.in
        inst/svd_rsvd.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid xdmrg++-math-svd-lapacke
        )
