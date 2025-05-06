add_source_target(xdmrg++-tools-infinite-opt
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
        xdmrg++-math-eig
        xdmrg++-math-tenx
        xdmrg++-tensors-infinite
        xdmrg++-tools-common-split
        )

