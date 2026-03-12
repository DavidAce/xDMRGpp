add_source_target(xdmrg++-tools-finite-ed
        INST_TEMPLATE_FILES
        ed.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-algorithms-status
        xdmrg++-math-eig
        xdmrg++-math-tenx
        xdmrg++-tensors-finite
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-opt
        xdmrg++-tools-common-split
        )
