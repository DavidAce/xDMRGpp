add_source_target(xdmrg++-tools-finite-multisite
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        multisite.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-mps
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction
        )
