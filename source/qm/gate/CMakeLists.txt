set(qm_gate_exp_scalars cx64 cx128)
add_source_target(xdmrg++-qm-gate
        INST_PAIR_TEMPLATE_FILES
        internal/exp_internal.pair.inst.cpp.in
        INST_PAIR_DIAGONAL
        INST_PAIR_OUTER_SCALARS
        ${qm_gate_exp_scalars}
        INST_PAIR_INNER_SCALARS
        ${qm_gate_exp_scalars}
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
