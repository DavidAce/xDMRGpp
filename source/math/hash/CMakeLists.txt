add_source_target(xdmrg++-math-hash
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        hash.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        )
