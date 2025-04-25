add_source_target(xdmrg++-tensors-edges-finite
        TARGET_SOURCES
        EdgesFinite.cpp
        EdgesFinite.fp32.cpp
        EdgesFinite.fp64.cpp
        EdgesFinite.fp128.cpp
        EdgesFinite.cx32.cpp
        EdgesFinite.cx64.cpp
        EdgesFinite.cx128.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-tensors-env
        xdmrg++-tools-common-log
        )
add_source_target(xdmrg++-tensors-edges-infinite
        TARGET_SOURCES
        EdgesInfinite.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-env
        xdmrg++-tools-common-log
        )

add_library(xdmrg++-tensors-edges INTERFACE)
target_link_libraries(xdmrg++-tensors-edges INTERFACE xdmrg++-tensors-edges-finite xdmrg++-tensors-edges-infinite)