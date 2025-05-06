add_source_target(xdmrg++-tools-finite-mps
        TARGET_SOURCES
        add/cx32.cpp
        add/cx64.cpp
        add/cx128.cpp
        add/fp32.cpp
        add/fp64.cpp
        add/fp128.cpp

        gates/cx32.cpp
        gates/cx64.cpp
        gates/cx128.cpp
        gates/fp32.cpp
        gates/fp64.cpp
        gates/fp128.cpp
        gates/gates.cpp

        merge/cx32.cpp
        merge/cx64.cpp
        merge/cx128.cpp
        merge/fp32.cpp
        merge/fp64.cpp
        merge/fp128.cpp

        move/cx32.cpp
        move/cx64.cpp
        move/cx128.cpp
        move/fp32.cpp
        move/fp64.cpp
        move/fp128.cpp

        normalize/cx32.cpp
        normalize/cx64.cpp
        normalize/cx128.cpp
        normalize/fp32.cpp
        normalize/fp64.cpp
        normalize/fp128.cpp

        init.cpp

        init_entangled_state/cx32.cpp
        init_entangled_state/cx64.cpp
        init_entangled_state/cx128.cpp
        init_entangled_state/fp32.cpp
        init_entangled_state/fp64.cpp
        init_entangled_state/fp128.cpp

        init_product_state/cx32.cpp
        init_product_state/cx64.cpp
        init_product_state/cx128.cpp
        init_product_state/fp32.cpp
        init_product_state/fp64.cpp
        init_product_state/fp128.cpp

        to_tensor/cx32.cpp
        to_tensor/cx64.cpp
        to_tensor/cx128.cpp
        to_tensor/fp32.cpp
        to_tensor/fp64.cpp
        to_tensor/fp128.cpp

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

