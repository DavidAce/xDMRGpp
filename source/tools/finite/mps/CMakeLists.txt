add_source_target(xdmrg++-tools-finite-mps
        TARGET_SOURCES
        gates/gates.cpp

        init.cpp
        INST_TEMPLATE_FILES
        add/add.inst.cpp.in
        gates/gates.inst.cpp.in
        merge/merge.inst.cpp.in
        move/move.inst.cpp.in
        normalize/normalize.inst.cpp.in
        init_entangled_state/init_entangled_state.inst.cpp.in
        init_product_state/init_product_state.inst.cpp.in
        to_tensor/to_tensor.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-qm-gate
        xdmrg++-qm-mpo
        xdmrg++-qm-spin
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-mps
        xdmrg++-tools-common-contraction
        xdmrg++-tools-common-log
        xdmrg++-tools-common-split
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-ops
        )
