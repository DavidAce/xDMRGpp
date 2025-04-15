add_source_target(xdmrg++-tools-finite-measure
        TARGET_SOURCES
        measure.cpp
        dimensions.cpp
        correlation.cpp
        entropy.cpp
        entanglement_entropy.cpp
        expectation_value.cpp
        hamiltonian.cpp
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

