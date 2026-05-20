add_source_target(xdmrg++-algorithms-status
        TARGET_SOURCES
        AlgorithmStatus.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        )


add_source_target(xdmrg++-algorithms-base
        TARGET_SOURCES
        AlgorithmBase.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-debug-info
        xdmrg++-debug-stacktrace
        xdmrg++-algorithms-status
        xdmrg++-tools-common-h5
        xdmrg++-tools-common-log
        )

set(algorithms_finite_deps
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-debug-info
        xdmrg++-debug-stacktrace
        xdmrg++-qm-spin
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tensors-edges-finite
        xdmrg++-algorithms-status
        xdmrg++-algorithms-base
        xdmrg++-tools-finite-h5
        xdmrg++-tools-finite-measure-dimensions
        xdmrg++-tools-finite-measure-entanglement-entropy
        xdmrg++-tools-finite-measure-expectation-value
        xdmrg++-tools-finite-measure-hamiltonian
        xdmrg++-tools-finite-measure-information
        xdmrg++-tools-finite-measure-norm
        xdmrg++-tools-finite-measure-number-entropy
        xdmrg++-tools-finite-measure-residual
        xdmrg++-tools-finite-measure-spin
        xdmrg++-tools-finite-mps-normalize
        xdmrg++-tools-finite-ops-apply
        xdmrg++-tools-finite-ops-overlap
        xdmrg++-tools-finite-pos
        xdmrg++-tools-finite-print
        xdmrg++-tools-common-log
        )

set(algorithms_finite_scalar_targets)
foreach(scalar IN LISTS DMRG_ENABLED_SCALARS)
    add_source_target(xdmrg++-algorithms-finite-${scalar}
            INST_TEMPLATE_FILES
            AlgorithmFinite/AlgorithmFinite.inst.cpp.in
            INST_SCALARS
            ${scalar}

            OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
            INTERFACE_LINK_LIBRARIES
            ${algorithms_finite_deps}
            )
    list(APPEND algorithms_finite_scalar_targets xdmrg++-algorithms-finite-${scalar})
endforeach()

add_library(xdmrg++-algorithms-finite INTERFACE)
target_link_libraries(xdmrg++-algorithms-finite INTERFACE
        ${algorithms_finite_scalar_targets}
        )

add_source_target(xdmrg++-algorithms-finite-opt-meta
        INST_TEMPLATE_FILES
        AlgorithmFinite/AlgorithmFinite.opt_meta.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-finite
        xdmrg++-tools-finite-multisite
        xdmrg++-tools-finite-opt
        xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-algorithms-infinite
        INST_TEMPLATE_FILES
        AlgorithmInfinite/AlgorithmInfinite.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-debug-info
        xdmrg++-algorithms-status
        xdmrg++-algorithms-base
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tools-infinite-h5
        xdmrg++-tools-infinite-measure
        xdmrg++-tools-common-log
        xdmrg++-tools-common-h5

        )
add_source_target(xdmrg++-algorithms-fdmrg
        INST_TEMPLATE_FILES
        fdmrg/fdmrg.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-finite
        xdmrg++-algorithms-finite-opt-meta
        xdmrg++-tools-finite-bex
        xdmrg++-tools-finite-opt
        xdmrg++-tools-common-h5
        xdmrg++-tools-common-prof

        )
if(cx64 IN_LIST DMRG_ENABLED_SCALARS)
    add_source_target(xdmrg++-algorithms-flbit
            INST_TEMPLATE_FILES
            flbit/flbit.inst.cpp.in
            flbit/flbit.tmpl.inst.cpp.in
            INST_SCALARS
            cx64

            OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
            INTERFACE_LINK_LIBRARIES
            xdmrg++-algorithms-finite-cx64
            xdmrg++-math-tenx
            xdmrg++-qm-lbit
            xdmrg++-tools-common-plot
            xdmrg++-tools-finite-h5
            xdmrg++-tools-finite-mps-gates
            xdmrg++-tools-finite-mps-init-product-state
            xdmrg++-tools-finite-mps-merge
            xdmrg++-tools-finite-mps-normalize
            xdmrg++-tools-finite-mps-to-tensor
            xdmrg++-tools-finite-ops-overlap
            xdmrg++-tools-common-h5
            xdmrg++-tools-common-prof

            )
else()
    add_library(xdmrg++-algorithms-flbit INTERFACE)
endif()

add_source_target(xdmrg++-algorithms-xdmrg
        INST_TEMPLATE_FILES
        xdmrg/xdmrg.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-finite
        xdmrg++-qm-time
        xdmrg++-math-rnd
        xdmrg++-algorithms-fdmrg
        xdmrg++-algorithms-finite-opt-meta
        xdmrg++-tools-finite-bex
        xdmrg++-tools-finite-opt
        )

add_source_target(xdmrg++-algorithms-idmrg
        INST_TEMPLATE_FILES
        idmrg/idmrg.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-infinite
        xdmrg++-tensors-edges-infinite
        xdmrg++-tools-infinite-opt
        )

if(cx64 IN_LIST DMRG_ENABLED_SCALARS)
    add_source_target(xdmrg++-algorithms-itebd
            INST_TEMPLATE_FILES
            itebd/itebd.inst.cpp.in
            INST_SCALARS
            cx64

            OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
            INTERFACE_LINK_LIBRARIES
            xdmrg++-qm-time
            xdmrg++-algorithms-infinite
            xdmrg++-tools-infinite-opt

            )
else()
    add_library(xdmrg++-algorithms-itebd INTERFACE)
endif()

add_source_target(xdmrg++-algorithms-launcher
        TARGET_SOURCES
        AlgorithmLauncher.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-fdmrg
        xdmrg++-algorithms-flbit
        xdmrg++-algorithms-itebd
        xdmrg++-algorithms-idmrg
        xdmrg++-algorithms-xdmrg
        )

add_library(xdmrg++-algorithms INTERFACE)
target_link_libraries(xdmrg++-algorithms INTERFACE
        xdmrg++-algorithms-base
        xdmrg++-algorithms-finite
        xdmrg++-algorithms-infinite
        xdmrg++-algorithms-launcher
        xdmrg++-algorithms-fdmrg
        xdmrg++-algorithms-flbit
        xdmrg++-algorithms-itebd
        xdmrg++-algorithms-idmrg
        xdmrg++-algorithms-xdmrg
        )
