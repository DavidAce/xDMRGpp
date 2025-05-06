add_source_target(xdmrg++-tools-finite-measure
        TARGET_SOURCES

        correlation/fp32.cpp
        correlation/fp64.cpp
        correlation/fp128.cpp
        correlation/cx32.cpp
        correlation/cx64.cpp
        correlation/cx128.cpp

        dimensions/fp32.cpp
        dimensions/fp64.cpp
        dimensions/fp128.cpp
        dimensions/cx32.cpp
        dimensions/cx64.cpp
        dimensions/cx128.cpp

        entanglement_entropy/fp32.cpp
        entanglement_entropy/fp64.cpp
        entanglement_entropy/fp128.cpp
        entanglement_entropy/cx32.cpp
        entanglement_entropy/cx64.cpp
        entanglement_entropy/cx128.cpp

        expectation_value/contract.cx32.cpp
        expectation_value/contract.cx64.cpp
        expectation_value/contract.cx128.cpp
        expectation_value/contract.fp32.cpp
        expectation_value/contract.fp64.cpp
        expectation_value/contract.fp128.cpp
        expectation_value/cx32.cx32.cpp
        expectation_value/cx32.cx64.cpp
        expectation_value/cx32.cx128.cpp
        expectation_value/cx32.fp32.cpp
        expectation_value/cx32.fp64.cpp
        expectation_value/cx32.fp128.cpp
        expectation_value/cx64.cx32.cpp
        expectation_value/cx64.cx64.cpp
        expectation_value/cx64.cx128.cpp
        expectation_value/cx64.fp32.cpp
        expectation_value/cx64.fp64.cpp
        expectation_value/cx64.fp128.cpp
        expectation_value/cx128.cx32.cpp
        expectation_value/cx128.cx64.cpp
        expectation_value/cx128.cx128.cpp
        expectation_value/cx128.fp32.cpp
        expectation_value/cx128.fp64.cpp
        expectation_value/cx128.fp128.cpp
        expectation_value/fp32.cx32.cpp
        expectation_value/fp32.cx64.cpp
        expectation_value/fp32.cx128.cpp
        expectation_value/fp32.fp32.cpp
        expectation_value/fp32.fp64.cpp
        expectation_value/fp32.fp128.cpp
        expectation_value/fp64.cx32.cpp
        expectation_value/fp64.cx64.cpp
        expectation_value/fp64.cx128.cpp
        expectation_value/fp64.fp32.cpp
        expectation_value/fp64.fp64.cpp
        expectation_value/fp64.fp128.cpp
        expectation_value/fp128.cx32.cpp
        expectation_value/fp128.cx64.cpp
        expectation_value/fp128.cx128.cpp
        expectation_value/fp128.fp32.cpp
        expectation_value/fp128.fp64.cpp
        expectation_value/fp128.fp128.cpp

        hamiltonian/fp32.cpp
        hamiltonian/fp64.cpp
        hamiltonian/fp128.cpp
        hamiltonian/cx32.cpp
        hamiltonian/cx64.cpp
        hamiltonian/cx128.cpp

        infopolicy.cpp

        information/fp32.cpp
        information/fp64.cpp
        information/fp128.cpp
        information/cx32.cpp
        information/cx64.cpp
        information/cx128.cpp

        norm/fp32.cpp
        norm/fp64.cpp
        norm/fp128.cpp
        norm/cx32.cpp
        norm/cx64.cpp
        norm/cx128.cpp

        number_entropy/fp32.cpp
        number_entropy/fp64.cpp
        number_entropy/fp128.cpp
        number_entropy/cx32.cpp
        number_entropy/cx64.cpp
        number_entropy/cx128.cpp

        opdm/fp32.cpp
        opdm/fp64.cpp
        opdm/fp128.cpp
        opdm/cx32.cpp
        opdm/cx64.cpp
        opdm/cx128.cpp

        residual/fp32.cpp
        residual/fp64.cpp
        residual/fp128.cpp
        residual/cx32.cpp
        residual/cx64.cpp
        residual/cx128.cpp

        spin/fp32.cpp
        spin/fp64.cpp
        spin/fp128.cpp
        spin/cx32.cpp
        spin/cx64.cpp
        spin/cx128.cpp

        truncation/fp32.cpp
        truncation/fp64.cpp
        truncation/fp128.cpp
        truncation/cx32.cpp
        truncation/cx64.cpp
        truncation/cx128.cpp

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

