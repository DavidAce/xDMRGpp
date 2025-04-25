add_source_target(xdmrg++-tensors-model-finite
        TARGET_SOURCES
        ModelFinite.cpp
        ModelFinite.fp32.cpp
        ModelFinite.fp64.cpp
        ModelFinite.fp128.cpp
        ModelFinite.cx32.cpp
        ModelFinite.cx64.cpp
        ModelFinite.cx128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-qm-spin
        xdmrg++-math-svd
        xdmrg++-tensors-mpo
        xdmrg++-tools-finite-multisite
        xdmrg++-tools-common-log
        )
add_source_target(xdmrg++-tensors-model-infinite
        TARGET_SOURCES
        ModelInfinite.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tensors-mpo

        )

add_library(xdmrg++-tensors-model INTERFACE)
target_link_libraries(xdmrg++-tensors-model INTERFACE xdmrg++-tensors-model-finite xdmrg++-tensors-model-infinite)