set(tools_finite_mps_state_deps
        xdmrg++-tid
        xdmrg++-math-tenx
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-mps
        xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-tools-finite-mps-add
        INST_TEMPLATE_FILES
        add/add.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        )

add_source_target(xdmrg++-tools-finite-mps-gates
        TARGET_SOURCES
        gates/gates.cpp
        INST_TEMPLATE_FILES
        gates/gates.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-qm-gate
        xdmrg++-qm-mpo
        xdmrg++-qm-spin
        xdmrg++-tools-finite-mps-merge
        )

add_source_target(xdmrg++-tools-finite-mps-init
        TARGET_SOURCES
        init.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-tools-finite-mps-init-entangled-state
        INST_TEMPLATE_FILES
        init_entangled_state/init_entangled_state.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        xdmrg++-config
        xdmrg++-math-rnd
        xdmrg++-qm-spin
        xdmrg++-tools-finite-measure-spin
        xdmrg++-tools-finite-mps-init
        xdmrg++-tools-finite-mps-merge
        )

add_source_target(xdmrg++-tools-finite-mps-init-product-state
        INST_TEMPLATE_FILES
        init_product_state/init_product_state.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        xdmrg++-config
        xdmrg++-math-rnd
        xdmrg++-qm-spin
        xdmrg++-tools-finite-mps-add
        xdmrg++-tools-finite-mps-init
        )

add_source_target(xdmrg++-tools-finite-mps-merge
        INST_TEMPLATE_FILES
        merge/merge.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-qm-mpo
        xdmrg++-qm-spin
        xdmrg++-tools-common-split
        xdmrg++-tools-finite-measure-dimensions
        xdmrg++-tools-finite-measure-norm
        xdmrg++-tools-finite-ops-apply
        )

add_source_target(xdmrg++-tools-finite-mps-move
        INST_TEMPLATE_FILES
        move/move.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-tools-common-contraction
        xdmrg++-tools-finite-mps-merge
        )

add_source_target(xdmrg++-tools-finite-mps-normalize
        INST_TEMPLATE_FILES
        normalize/normalize.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-tools-finite-measure-dimensions
        xdmrg++-tools-finite-measure-norm
        xdmrg++-tools-finite-mps-merge
        )

add_source_target(xdmrg++-tools-finite-mps-to-tensor
        INST_TEMPLATE_FILES
        to_tensor/to_tensor.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_mps_state_deps}
        )

add_library(xdmrg++-tools-finite-mps INTERFACE)
target_link_libraries(xdmrg++-tools-finite-mps INTERFACE
        xdmrg++-tools-finite-mps-add
        xdmrg++-tools-finite-mps-gates
        xdmrg++-tools-finite-mps-init
        xdmrg++-tools-finite-mps-init-entangled-state
        xdmrg++-tools-finite-mps-init-product-state
        xdmrg++-tools-finite-mps-merge
        xdmrg++-tools-finite-mps-move
        xdmrg++-tools-finite-mps-normalize
        xdmrg++-tools-finite-mps-to-tensor
        )
