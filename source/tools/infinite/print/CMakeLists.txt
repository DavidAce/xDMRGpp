add_source_target(xdmrg++-tools-infinite-print
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        print.inst.cpp.in


        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-infinite
        )
