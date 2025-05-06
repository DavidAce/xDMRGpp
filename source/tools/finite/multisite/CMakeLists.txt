add_source_target(xdmrg++-tools-finite-multisite
        TARGET_SOURCES
        fp32.cpp
        fp64.cpp
        fp128.cpp
        cx32.cpp
        cx64.cpp
        cx128.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-mps
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction
        )

