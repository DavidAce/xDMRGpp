set(tools_finite_ops_state_deps
        xdmrg++-tid
        xdmrg++-math-tenx
        xdmrg++-tensors-state-finite
        xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-tools-finite-ops-apply
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            ops/ops_apply.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_ops_state_deps}
            xdmrg++-config
            xdmrg++-math-svd
            xdmrg++-qm-mpo
            xdmrg++-tensors-mpo
            xdmrg++-tools-common-contraction
            xdmrg++-tools-finite-measure-dimensions
            xdmrg++-tools-finite-measure-entanglement-entropy
            xdmrg++-tools-finite-measure-norm
            xdmrg++-tools-finite-measure-spin
            xdmrg++-tools-finite-mpo
        )

add_source_target(xdmrg++-tools-finite-ops-overlap
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            ops/ops_overlap.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_ops_state_deps}
            xdmrg++-tools-common-contraction
        )

add_source_target(xdmrg++-tools-finite-ops-project
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            ops/ops_project.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_ops_state_deps}
            xdmrg++-config
            xdmrg++-math-rnd
            xdmrg++-math-svd
            xdmrg++-qm-mpo
            xdmrg++-qm-spin
            xdmrg++-tools-finite-measure-dimensions
            xdmrg++-tools-finite-measure-norm
            xdmrg++-tools-finite-measure-spin
            xdmrg++-tools-finite-mps-normalize
            xdmrg++-tools-finite-ops-apply
        )

add_library(xdmrg++-tools-finite-ops INTERFACE)
target_link_libraries(xdmrg++-tools-finite-ops INTERFACE
        xdmrg++-tools-finite-ops-apply
        xdmrg++-tools-finite-ops-overlap
        xdmrg++-tools-finite-ops-project
        )

add_source_target(xdmrg++-tools-finite-print
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            print/print.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tensors-finite
            xdmrg++-tools-common-log
        )


add_subdirectory(h5)
add_subdirectory(ed)
add_subdirectory(measure)
add_subdirectory(multisite)
add_subdirectory(env)
add_subdirectory(bex)
add_subdirectory(mps)
add_subdirectory(pos)
add_subdirectory(opt)
add_subdirectory(mpo)

add_library(xdmrg++-tools-finite INTERFACE)
target_link_libraries(xdmrg++-tools-finite INTERFACE
        xdmrg++-tools-finite-h5
        xdmrg++-tools-finite-ed
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-mps
        xdmrg++-tools-finite-env
        xdmrg++-tools-finite-bex
        xdmrg++-tools-finite-pos
        xdmrg++-tools-finite-multisite
        xdmrg++-tools-finite-opt
        xdmrg++-tools-finite-mpo
        xdmrg++-tools-finite-ops
        xdmrg++-tools-finite-print
        )
target_link_libraries(xdmrg++-tools-finite INTERFACE xdmrg++-deps xdmrg++-flags)
