add_source_target(xdmrg++-tools-infinite-mps
        TARGET_SOURCES
        merge/cx32.cpp
        merge/cx64.cpp
        merge/cx128.cpp
        merge/fp32.cpp
        merge/fp64.cpp
        merge/fp128.cpp


        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-tensors-state-infinite
        xdmrg++-tools-common-split
        )

