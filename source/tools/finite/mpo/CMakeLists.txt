add_source_target(xdmrg++-tools-finite-mpo-compress-svd
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            compress/compress.svd.inst.cpp.in
        INST_SCALARS
            ${DMRG_ENABLED_SCALARS}
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-math-svd
            xdmrg++-math-tenx
            xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-tools-finite-mpo-compress-dpl
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            compress/compress.dpl.inst.cpp.in
        INST_SCALARS
            ${DMRG_ENABLED_SCALARS}
            ${DMRG_HIGHPREC_REAL}
            ${DMRG_HIGHPREC_CPLX}
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-math-tenx
            xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-tools-finite-mpo
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            edge/edge.inst.cpp.in
            merge/merge.inst.cpp.in
            swap/swap.inst.cpp.in
        INST_SCALARS
            ${DMRG_ENABLED_SCALARS}

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
            xdmrg++-tools-finite-mpo-compress-svd
            xdmrg++-tools-finite-mpo-compress-dpl
        )
