add_source_target(xdmrg++-tools-finite-ops
        TARGET_SOURCES
            ops/fp32.cpp
            ops/fp64.cpp
            ops/fp128.cpp
            ops/cx32.cpp
            ops/cx64.cpp
            ops/cx128.cpp
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-qm-spin
            xdmrg++-qm-mpo
            xdmrg++-tensors-state-finite
            xdmrg++-tensors-mpo
            xdmrg++-math-tenx
            xdmrg++-tools-finite-mps
            xdmrg++-tools-finite-measure
            xdmrg++-tools-common-log
            xdmrg++-tools-common-contraction
        )


add_source_target(xdmrg++-tools-finite-print
        TARGET_SOURCES
            print.cpp
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tensors-finite
            xdmrg++-tools-common-log
        )


add_subdirectory(h5)
add_subdirectory(ed)
add_subdirectory(measure)
add_subdirectory(multisite)
add_subdirectory(env)
add_subdirectory(mps)
add_subdirectory(opt)
add_subdirectory(mpo)

add_library(xdmrg++-tools-finite INTERFACE
mps/add.cpp)
target_link_libraries(xdmrg++-tools-finite INTERFACE
        xdmrg++-tools-finite-h5
        xdmrg++-tools-finite-ed
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-mps
        xdmrg++-tools-finite-env
        xdmrg++-tools-finite-multisite
        xdmrg++-tools-finite-opt
        xdmrg++-tools-finite-mpo
        xdmrg++-tools-finite-ops
        xdmrg++-tools-finite-print
        )
target_link_libraries(xdmrg++-tools-finite INTERFACE xdmrg++-deps xdmrg++-flags)
