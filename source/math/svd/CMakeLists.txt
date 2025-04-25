add_source_target(xdmrg++-math-svd
        TARGET_SOURCES
        svd.cpp
        svd_dump.cpp
        svd_lapacke.cpp
        svd_eigen.fp32.cpp
        svd_eigen.fp64.cpp
        svd_eigen.fp128.cpp
        svd_eigen.cx32.cpp
        svd_eigen.cx64.cpp
        svd_eigen.cx128.cpp
        svd_rsvd.fp32.cpp
        svd_rsvd.fp64.cpp
        svd_rsvd.fp128.cpp
        svd_rsvd.cx32.cpp
        svd_rsvd.cx64.cpp
        svd_rsvd.cx128.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        )

