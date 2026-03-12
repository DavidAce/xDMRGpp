add_source_target(xdmrg++-qm-time
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        time.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-math-tenx
        xdmrg++-tools-common-log
        )
