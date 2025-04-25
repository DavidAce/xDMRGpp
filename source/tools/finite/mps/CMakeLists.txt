add_source_target(xdmrg++-tools-finite-mps
        TARGET_SOURCES
        mps.cpp
        add.cpp
        move.cpp
        gates.cpp
        mps_init_entangled_state.cpp
        mps_init_product_state.cpp
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

