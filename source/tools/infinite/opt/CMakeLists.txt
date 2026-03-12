add_source_target(xdmrg++-tools-infinite-opt
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        opt.inst.cpp.in


        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig
        xdmrg++-math-tenx
        xdmrg++-tensors-infinite
        xdmrg++-tools-common-split
        )
