add_source_target(xdmrg++-tools-finite-env
        TARGET_SOURCES
        cx32.cpp
        cx64.cpp
        cx128.cpp
        fp32.cpp
        fp64.cpp
        fp128.cpp
        fp32.cx32.cpp
        fp64.cx64.cpp
        fp128.cx128.cpp

        residual_expansion/cx32.cpp
        residual_expansion/cx64.cpp
        residual_expansion/cx128.cpp
        residual_expansion/fp32.cpp
        residual_expansion/fp64.cpp
        residual_expansion/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tensors-finite
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction
        )

