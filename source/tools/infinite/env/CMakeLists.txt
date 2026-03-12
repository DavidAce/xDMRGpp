add_source_target(xdmrg++-tools-infinite-env
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        env.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-infinite
        )
