add_source_target(xdmrg++-tools-infinite-measure
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        ham/ham.inst.cpp.in
        mom/mom.inst.cpp.in
        mpo/mpo.inst.cpp.in
        measure/measure.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-tenx
        xdmrg++-algorithms-status
        xdmrg++-tensors-infinite
        xdmrg++-tools-common-contraction
        xdmrg++-tools-common-log
        )
