add_source_target(xdmrg++-tensors-model-finite
        INST_TEMPLATE_FILES
        ModelFinite/ModelFinite.inst.cpp.in

        INST_PAIR_TEMPLATE_FILES
        ModelFinite/ModelFinite.pair.inst.cpp.in

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
        INST_TEMPLATE_FILES
        ModelInfinite/ModelInfinite.inst.cpp.in
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
