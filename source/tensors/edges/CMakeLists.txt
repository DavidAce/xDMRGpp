add_source_target(xdmrg++-tensors-edges-finite
        TARGET_SOURCES
        EdgesFinite/fp32.cpp
        EdgesFinite/fp64.cpp
        EdgesFinite/fp128.cpp
        EdgesFinite/cx32.cpp
        EdgesFinite/cx64.cpp
        EdgesFinite/cx128.cpp

        EdgesFinite/fp32/fp32.cpp
        EdgesFinite/fp32/fp64.cpp
        EdgesFinite/fp32/fp128.cpp
        EdgesFinite/fp32/cx32.cpp
        EdgesFinite/fp32/cx64.cpp
        EdgesFinite/fp32/cx128.cpp
        EdgesFinite/fp64/fp32.cpp
        EdgesFinite/fp64/fp64.cpp
        EdgesFinite/fp64/fp128.cpp
        EdgesFinite/fp64/cx32.cpp
        EdgesFinite/fp64/cx64.cpp
        EdgesFinite/fp64/cx128.cpp
        EdgesFinite/fp128/fp32.cpp
        EdgesFinite/fp128/fp64.cpp
        EdgesFinite/fp128/fp128.cpp
        EdgesFinite/fp128/cx32.cpp
        EdgesFinite/fp128/cx64.cpp
        EdgesFinite/fp128/cx128.cpp
        EdgesFinite/cx32/fp32.cpp
        EdgesFinite/cx32/fp64.cpp
        EdgesFinite/cx32/fp128.cpp
        EdgesFinite/cx32/cx32.cpp
        EdgesFinite/cx32/cx64.cpp
        EdgesFinite/cx32/cx128.cpp
        EdgesFinite/cx64/fp32.cpp
        EdgesFinite/cx64/fp64.cpp
        EdgesFinite/cx64/fp128.cpp
        EdgesFinite/cx64/cx32.cpp
        EdgesFinite/cx64/cx64.cpp
        EdgesFinite/cx64/cx128.cpp
        EdgesFinite/cx128/fp32.cpp
        EdgesFinite/cx128/fp64.cpp
        EdgesFinite/cx128/fp128.cpp
        EdgesFinite/cx128/cx32.cpp
        EdgesFinite/cx128/cx64.cpp
        EdgesFinite/cx128/cx128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-tensors-env
        xdmrg++-tools-common-log
        )
add_source_target(xdmrg++-tensors-edges-infinite
        TARGET_SOURCES
        EdgesInfinite/fp32.cpp
        EdgesInfinite/fp64.cpp
        EdgesInfinite/fp128.cpp
        EdgesInfinite/cx32.cpp
        EdgesInfinite/cx64.cpp
        EdgesInfinite/cx128.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-env
        xdmrg++-tools-common-log
        )

add_library(xdmrg++-tensors-edges INTERFACE)
target_link_libraries(xdmrg++-tensors-edges INTERFACE xdmrg++-tensors-edges-finite xdmrg++-tensors-edges-infinite)