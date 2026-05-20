add_source_target(xdmrg++-qm-lbit-h5tb
        TARGET_SOURCES
        h5tb.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-deps xdmrg++-flags
        )

dmrg_filter_enabled_scalars(qm_lbit_cls_scalars cx32 cx64 cx128)
if(qm_lbit_cls_scalars)
    add_source_target(xdmrg++-qm-lbit
            INST_TEMPLATE_FILES
            cls.inst.cpp.in
            INST_SCALARS
            ${qm_lbit_cls_scalars}
            TARGET_SOURCES
            lbit.cpp
            cls.cpp
            OBJECT_LINK_LIBRARIES
            xdmrg++-qm-lbit-h5tb
            xdmrg++-deps xdmrg++-flags
            )
else()
    add_source_target(xdmrg++-qm-lbit
            TARGET_SOURCES
            lbit.cpp
            cls.cpp
            OBJECT_LINK_LIBRARIES
            xdmrg++-qm-lbit-h5tb
            xdmrg++-deps xdmrg++-flags
            )
endif()
