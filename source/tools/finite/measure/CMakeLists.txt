set(tools_finite_measure_state_deps
        xdmrg++-tid
        xdmrg++-math-tenx
        xdmrg++-tensors-state-finite
        xdmrg++-tools-common-log
        )

set(tools_finite_measure_expectation_deps
        ${tools_finite_measure_state_deps}
        xdmrg++-math-svd
        xdmrg++-tensors-env
        xdmrg++-tensors-mpo
        xdmrg++-tensors-mps
        xdmrg++-tools-common-contraction
        xdmrg++-tools-common-split
        xdmrg++-tools-finite-mpo
        )

add_source_target(xdmrg++-tools-finite-measure-correlation
        INST_TEMPLATE_FILES
        correlation/correlation.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-tools-finite-measure-expectation-value
        )

add_source_target(xdmrg++-tools-finite-measure-dimensions
        INST_TEMPLATE_FILES
        dimensions/dimensions.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-tensors-state-finite
        )

add_source_target(xdmrg++-tools-finite-measure-entanglement-entropy
        INST_TEMPLATE_FILES
        entanglement_entropy/entanglement_entropy.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        )

add_source_target(xdmrg++-tools-finite-measure-expectation-value
        INST_TEMPLATE_FILES
        expectation_value/contract.inst.cpp.in
        INST_PAIR_TEMPLATE_FILES
        expectation_value/expectation_value.pair.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_expectation_deps}
        )

add_source_target(xdmrg++-tools-finite-measure-hamiltonian
        INST_TEMPLATE_FILES
        hamiltonian/hamiltonian.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-tensors-finite
        xdmrg++-tools-common-contraction
        xdmrg++-tools-finite-measure-expectation-value
        )

add_source_target(xdmrg++-tools-finite-measure-information
        TARGET_SOURCES
        infopolicy.cpp
        INST_TEMPLATE_FILES
        information/information.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-math-eig
        xdmrg++-math-svd
        xdmrg++-qm-mpo
        xdmrg++-tools-common-split
        xdmrg++-tools-finite-measure-dimensions
        xdmrg++-tools-finite-measure-entanglement-entropy
        )

add_source_target(xdmrg++-tools-finite-measure-norm
        INST_TEMPLATE_FILES
        norm/norm.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-tools-common-contraction
        )

add_source_target(xdmrg++-tools-finite-measure-number-entropy
        INST_TEMPLATE_FILES
        number_entropy/number_entropy.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-tools-finite-measure-dimensions
        xdmrg++-tools-finite-mps-move
        )

add_source_target(xdmrg++-tools-finite-measure-opdm
        INST_TEMPLATE_FILES
        opdm/opdm.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-math-eig
        xdmrg++-qm-spin
        xdmrg++-tools-finite-measure-expectation-value
        )

add_source_target(xdmrg++-tools-finite-measure-residual
        INST_TEMPLATE_FILES
        residual/residual.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-tensors-finite
        xdmrg++-tools-common-contraction
        xdmrg++-tools-finite-measure-hamiltonian
        xdmrg++-tools-finite-measure-norm
        xdmrg++-tools-finite-ops-apply
        )

add_source_target(xdmrg++-tools-finite-measure-spin
        INST_TEMPLATE_FILES
        spin/spin.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_measure_state_deps}
        xdmrg++-qm-mpo
        xdmrg++-qm-spin
        xdmrg++-tools-common-contraction
        xdmrg++-tools-finite-measure-correlation
        xdmrg++-tools-finite-measure-expectation-value
        )

add_source_target(xdmrg++-tools-finite-measure-truncation
        INST_TEMPLATE_FILES
        truncation/truncation.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-tensors-state-finite
        )

add_library(xdmrg++-tools-finite-measure INTERFACE)
target_link_libraries(xdmrg++-tools-finite-measure INTERFACE
        xdmrg++-tools-finite-measure-correlation
        xdmrg++-tools-finite-measure-dimensions
        xdmrg++-tools-finite-measure-entanglement-entropy
        xdmrg++-tools-finite-measure-expectation-value
        xdmrg++-tools-finite-measure-hamiltonian
        xdmrg++-tools-finite-measure-information
        xdmrg++-tools-finite-measure-norm
        xdmrg++-tools-finite-measure-number-entropy
        xdmrg++-tools-finite-measure-opdm
        xdmrg++-tools-finite-measure-residual
        xdmrg++-tools-finite-measure-spin
        xdmrg++-tools-finite-measure-truncation
        )
