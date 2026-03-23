add_source_target(xdmrg++-qm-lbit-h5tb
        TARGET_SOURCES
        h5tb.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-deps xdmrg++-flags
        )
add_source_target(xdmrg++-qm-lbit
        TARGET_SOURCES
        lbit.cpp
        cls.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-qm-lbit-h5tb
        xdmrg++-deps xdmrg++-flags
        )
