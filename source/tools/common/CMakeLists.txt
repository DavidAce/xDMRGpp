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
        split/cx32/cx32.cpp
        split/cx32/cx64.cpp
        split/cx32/cx128.cpp
        split/cx32/fp32.cpp
        split/cx32/fp64.cpp
        split/cx32/fp128.cpp
        split/cx64/cx32.cpp
        split/cx64/cx64.cpp
        split/cx64/cx128.cpp
        split/cx64/fp32.cpp
        split/cx64/fp64.cpp
        split/cx64/fp128.cpp
        split/cx128/cx32.cpp
        split/cx128/cx64.cpp
        split/cx128/cx128.cpp
        split/cx128/fp32.cpp
        split/cx128/fp64.cpp
        split/cx128/fp128.cpp
        split/fp32/cx32.cpp
        split/fp32/cx64.cpp
        split/fp32/cx128.cpp
        split/fp32/fp32.cpp
        split/fp32/fp64.cpp
        split/fp32/fp128.cpp
        split/fp64/cx32.cpp
        split/fp64/cx64.cpp
        split/fp64/cx128.cpp
        split/fp64/fp32.cpp
        split/fp64/fp64.cpp
        split/fp64/fp128.cpp
        split/fp128/cx32.cpp
        split/fp128/cx64.cpp
        split/fp128/cx128.cpp
        split/fp128/fp32.cpp
        split/fp128/fp64.cpp
        split/fp128/fp128.cpp


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
        views/cx32.cpp
        views/cx64.cpp
        views/cx128.cpp
        views/fp32.cpp
        views/fp64.cpp
        views/fp128.cpp

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
