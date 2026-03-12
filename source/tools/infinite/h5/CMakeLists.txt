add_source_target(xdmrg++-tools-infinite-h5
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        h5.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-algorithms-status
        xdmrg++-tensors-infinite
        xdmrg++-tools-infinite-measure
        )
