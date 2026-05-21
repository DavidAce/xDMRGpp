set(xdmrg_tools_finite_mpo_inst_scalars ${DMRG_ENABLED_SCALARS} ${DMRG_HIGHPREC_SCALARS})

add_source_target(xdmrg++-tools-finite-mpo
        TARGET_SOURCES
        INST_TEMPLATE_FILES
            compress/compress.inst.cpp.in
            edge/edge.inst.cpp.in
            merge/merge.inst.cpp.in
            swap/swap.inst.cpp.in
        INST_SCALARS
            ${xdmrg_tools_finite_mpo_inst_scalars}

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
