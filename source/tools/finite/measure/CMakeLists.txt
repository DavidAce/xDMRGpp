add_source_target(xdmrg++-tools-finite-measure
        TARGET_SOURCES
        measure.cpp
        dimensions.cpp
        correlation.cpp
        entropy.cpp
        entanglement_entropy.cpp
#        expectation_value.cpp
        expectation_value.fp32.cpp
        expectation_value.fp64.cpp
        expectation_value.fp128.cpp
        expectation_value.cx32.cpp
        expectation_value.cx64.cpp
        expectation_value.cx128.cpp
        hamiltonian.fp32.cpp
        hamiltonian.fp64.cpp
        hamiltonian.fp128.cpp
        hamiltonian.cx32.cpp
        hamiltonian.cx64.cpp
        hamiltonian.cx128.cpp
        infopolicy.cpp
        information.cpp
        number_entropy.cpp
        opdm.cpp
        residual.cpp
        spin.cpp
        truncation.cpp


        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-qm-spin
        xdmrg++-qm-mpo
        xdmrg++-tensors-finite
        xdmrg++-math-eig
        xdmrg++-math-tenx
        xdmrg++-math-linalg
        xdmrg++-tools-finite-mps
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction

        )

