add_subdirectory(h5)
add_subdirectory(contraction)

add_source_target(xdmrg++-tools-common-log TARGET_SOURCES log.cpp OBJECT_LINK_LIBRARIES xdmrg++-deps xdmrg++-flags)

add_source_target(xdmrg++-tools-common-prof
        TARGET_SOURCES
        prof.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-tools-common-log
        )
add_source_target(xdmrg++-tools-common-split
        TARGET_SOURCES
        split/split.fp32.cpp
        split/split.fp64.cpp
        split/split.fp128.cpp
        split/split.cx32.cpp
        split/split.cx64.cpp
        split/split.cx128.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-tensors-mps
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction
        )
add_source_target(xdmrg++-tools-common-views
        TARGET_SOURCES
        views.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-math-eig
        xdmrg++-math-tenx
        xdmrg++-tensors-state
        xdmrg++-tools-common-log
        )
add_source_target(xdmrg++-tools-common-plot
        TARGET_SOURCES
        plot/plot.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        )


add_library(xdmrg++-tools-common INTERFACE)
target_link_libraries(xdmrg++-tools-common INTERFACE xdmrg++-deps xdmrg++-flags)
target_link_libraries(xdmrg++-tools-common INTERFACE
        xdmrg++-tools-common-log
        xdmrg++-tools-common-prof
        xdmrg++-tools-common-split
        xdmrg++-tools-common-views
        xdmrg++-tools-common-plot
        xdmrg++-tools-common-h5
        xdmrg++-tools-common-contraction
        )
