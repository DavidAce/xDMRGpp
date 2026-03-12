add_source_target(xdmrg++-qm-mpo
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        mpo.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-math-rnd
        xdmrg++-math-tenx
        )
