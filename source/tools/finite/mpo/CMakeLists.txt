add_source_target(xdmrg++-tools-finite-mpo
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            compress/compress.inst.cpp.in
            edge/edge.inst.cpp.in
            merge/merge.inst.cpp.in
            swap/swap.inst.cpp.in

        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-math-svd
            xdmrg++-math-tenx
            xdmrg++-tensors-mpo
            xdmrg++-tools-common-log
            xdmrg++-tools-common-contraction
            xdmrg++-qm-spin
        )
