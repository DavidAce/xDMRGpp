add_source_target(xdmrg++-tensors-edges-finite
        INST_TEMPLATE_FILES
        EdgesFinite/EdgesFinite.inst.cpp.in

        INST_PAIR_TEMPLATE_FILES
        EdgesFinite/EdgesFinite.pair.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-tensors-env
        xdmrg++-tools-common-log
        )
add_source_target(xdmrg++-tensors-edges-infinite
        INST_TEMPLATE_FILES
        EdgesInfinite/EdgesInfinite.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-env
        xdmrg++-tools-common-log
        )

add_library(xdmrg++-tensors-edges INTERFACE)
target_link_libraries(xdmrg++-tensors-edges INTERFACE xdmrg++-tensors-edges-finite xdmrg++-tensors-edges-infinite)
