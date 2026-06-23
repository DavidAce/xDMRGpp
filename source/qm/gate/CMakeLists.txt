add_source_target(xdmrg++-qm-gate
        INST_PAIR_TEMPLATE_FILES
        internal/exp_internal.pair.inst.cpp.in
        INST_PAIR_DIAGONAL
        INST_PAIR_OUTER_SCALARS
        ${DMRG_ENABLED_SCALARS}
        ${DMRG_HIGHPREC_CPLX}
        INST_PAIR_INNER_SCALARS
        ${DMRG_ENABLED_SCALARS}
        ${DMRG_HIGHPREC_CPLX}
        TARGET_SOURCES
        gate.cpp
        swap.cpp
        circuit.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-config-enums
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-math-tenx
        xdmrg++-tools-common-log
#        COMPILE_OPTIONS
#        -mlong-double-128
        )
