add_source_target(xdmrg++-tools-infinite-h5
        TARGET_SOURCES
        cx32.cpp
        cx64.cpp
        cx128.cpp
        fp32.cpp
        fp64.cpp
        fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-algorithms-status
        xdmrg++-tensors-infinite
        xdmrg++-tools-infinite-measure
        )

