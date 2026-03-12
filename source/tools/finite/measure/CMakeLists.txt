add_source_target(xdmrg++-tools-finite-measure
        TARGET_SOURCES

        infopolicy.cpp

        INST_TEMPLATE_FILES
        correlation/correlation.inst.cpp.in
        dimensions/dimensions.inst.cpp.in
        entanglement_entropy/entanglement_entropy.inst.cpp.in
        expectation_value/contract.inst.cpp.in
        hamiltonian/hamiltonian.inst.cpp.in
        information/information.inst.cpp.in
        norm/norm.inst.cpp.in
        number_entropy/number_entropy.inst.cpp.in
        opdm/opdm.inst.cpp.in
        residual/residual.inst.cpp.in
        spin/spin.inst.cpp.in
        truncation/truncation.inst.cpp.in

        INST_PAIR_TEMPLATE_FILES
        expectation_value/expectation_value.pair.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-qm-spin
        xdmrg++-qm-mpo
        xdmrg++-tensors-finite
        xdmrg++-math-eig
        xdmrg++-math-tenx
        xdmrg++-tools-finite-mps
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction

        )
