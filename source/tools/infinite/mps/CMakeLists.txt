add_source_target(xdmrg++-tools-infinite-mps
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        merge/merge.inst.cpp.in


        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-tensors-state-infinite
        xdmrg++-tools-common-split
        )
