add_source_target(xdmrg++-tensors-model-finite
        TARGET_SOURCES
        ModelFinite/fp32.cpp
        ModelFinite/fp64.cpp
        ModelFinite/fp128.cpp
        ModelFinite/cx32.cpp
        ModelFinite/cx64.cpp
        ModelFinite/cx128.cpp

        ModelFinite/fp32/fp32.cpp
        ModelFinite/fp32/fp64.cpp
        ModelFinite/fp32/fp128.cpp
        ModelFinite/fp32/cx32.cpp
        ModelFinite/fp32/cx64.cpp
        ModelFinite/fp32/cx128.cpp
        ModelFinite/fp64/fp32.cpp
        ModelFinite/fp64/fp64.cpp
        ModelFinite/fp64/fp128.cpp
        ModelFinite/fp64/cx32.cpp
        ModelFinite/fp64/cx64.cpp
        ModelFinite/fp64/cx128.cpp
        ModelFinite/fp128/fp32.cpp
        ModelFinite/fp128/fp64.cpp
        ModelFinite/fp128/fp128.cpp
        ModelFinite/fp128/cx32.cpp
        ModelFinite/fp128/cx64.cpp
        ModelFinite/fp128/cx128.cpp
        ModelFinite/cx32/fp32.cpp
        ModelFinite/cx32/fp64.cpp
        ModelFinite/cx32/fp128.cpp
        ModelFinite/cx32/cx32.cpp
        ModelFinite/cx32/cx64.cpp
        ModelFinite/cx32/cx128.cpp
        ModelFinite/cx64/fp32.cpp
        ModelFinite/cx64/fp64.cpp
        ModelFinite/cx64/fp128.cpp
        ModelFinite/cx64/cx32.cpp
        ModelFinite/cx64/cx64.cpp
        ModelFinite/cx64/cx128.cpp
        ModelFinite/cx128/fp32.cpp
        ModelFinite/cx128/fp64.cpp
        ModelFinite/cx128/fp128.cpp
        ModelFinite/cx128/cx32.cpp
        ModelFinite/cx128/cx64.cpp
        ModelFinite/cx128/cx128.cpp

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
        ModelInfinite/fp32.cpp
        ModelInfinite/fp64.cpp
        ModelInfinite/fp128.cpp
        ModelInfinite/cx32.cpp
        ModelInfinite/cx64.cpp
        ModelInfinite/cx128.cpp
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