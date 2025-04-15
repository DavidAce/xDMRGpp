add_source_target(xdmrg++-tools-finite-mpo
        TARGET_SOURCES
            mpo.cpp mpo_inv.cpp deproject.cpp merge.cpp
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-math-svd
            xdmrg++-math-tenx
            xdmrg++-tensors-mpo
            xdmrg++-tools-common-log
            xdmrg++-tools-common-contraction
            xdmrg++-qm-spin
        )