add_source_target(xdmrg++-tools-finite-pos
        INST_TEMPLATE_FILES
        pos.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tensors-finite
        xdmrg++-tools-finite-mps
        xdmrg++-tools-finite-mpo
        xdmrg++-tools-finite-multisite
        xdmrg++-tools-common-log
        )
