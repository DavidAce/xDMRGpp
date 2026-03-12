add_source_target(xdmrg++-tensors-state-finite
        INST_TEMPLATE_FILES
        StateFinite/StateFinite.inst.cpp.in
        StateFinite/contract.inst.cpp.in

        INST_PAIR_TEMPLATE_FILES
        StateFinite/StateFinite.pair.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-debug
        xdmrg++-tensors-mps
        xdmrg++-math-tenx
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-multisite
        xdmrg++-tools-common-contraction
        xdmrg++-tools-common-log
        )
add_source_target(xdmrg++-tensors-state-infinite
        INST_TEMPLATE_FILES
        StateInfinite/StateInfinite.inst.cpp.in

        INST_PAIR_TEMPLATE_FILES
        StateInfinite/StateInfinite.pair.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-debug
        xdmrg++-math-tenx
        xdmrg++-tensors-mps
        xdmrg++-tools-common-contraction
        xdmrg++-tools-infinite-measure
        xdmrg++-tools-infinite-mps
        xdmrg++-tools-common-log
        xdmrg++-tools-common-split
        xdmrg++-tools-common-views
        )

add_library(xdmrg++-tensors-state INTERFACE)
target_link_libraries(xdmrg++-tensors-state INTERFACE
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-state-infinite)
