add_source_target(xdmrg++-algorithms-status
        TARGET_SOURCES
        AlgorithmStatus.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        )


add_source_target(xdmrg++-algorithms-base
        TARGET_SOURCES
        AlgorithmBase.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-config
        xdmrg++-debug
        xdmrg++-stacktrace
        xdmrg++-algorithms-status
        xdmrg++-tools-common-h5
        xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-algorithms-finite
        INST_TEMPLATE_FILES
        AlgorithmFinite/AlgorithmFinite.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-debug
        xdmrg++-stacktrace
        xdmrg++-qm-spin
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tensors-edges-finite
        xdmrg++-algorithms-status
        xdmrg++-algorithms-base
        xdmrg++-tools-finite-env
        xdmrg++-tools-finite-h5
        xdmrg++-tools-finite-measure
        xdmrg++-tools-finite-ops
        xdmrg++-tools-finite-print
        xdmrg++-tools-common-log
        )

add_source_target(xdmrg++-algorithms-infinite
        INST_TEMPLATE_FILES
        AlgorithmInfinite/AlgorithmInfinite.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-debug
        xdmrg++-algorithms-status
        xdmrg++-algorithms-base
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tools-infinite-h5
        xdmrg++-tools-infinite-measure
        xdmrg++-tools-common-log
        xdmrg++-tools-common-h5

        )
add_source_target(xdmrg++-algorithms-fdmrg
        INST_TEMPLATE_FILES
        fdmrg/fdmrg.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-finite
        xdmrg++-tools-finite-opt
        xdmrg++-tools-common-h5
        xdmrg++-tools-common-prof

        )
add_source_target(xdmrg++-algorithms-flbit
        INST_TEMPLATE_FILES
        flbit/flbit.inst.cpp.in
        flbit/flbit.tmpl.inst.cpp.in
        INST_SCALARS
        cx64

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-finite
        xdmrg++-math-tenx
        xdmrg++-qm-lbit
        xdmrg++-tools-finite-opt
        xdmrg++-tools-finite-mps
        xdmrg++-tools-common-h5
        xdmrg++-tools-common-prof
        xdmrg++-tools-common-plot

        )

add_source_target(xdmrg++-algorithms-xdmrg
        INST_TEMPLATE_FILES
        xdmrg/xdmrg.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-finite
        xdmrg++-qm-time
        xdmrg++-math-rnd
        xdmrg++-algorithms-fdmrg
        xdmrg++-tools-finite-opt
        )

add_source_target(xdmrg++-algorithms-idmrg
        INST_TEMPLATE_FILES
        idmrg/idmrg.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-infinite
        xdmrg++-tensors-edges-infinite
        xdmrg++-tools-infinite-opt
        )

add_source_target(xdmrg++-algorithms-itebd
        INST_TEMPLATE_FILES
        itebd/itebd.inst.cpp.in
        INST_SCALARS
        cx64

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-qm-time
        xdmrg++-algorithms-infinite
        xdmrg++-tools-infinite-opt

        )

add_source_target(xdmrg++-algorithms-launcher
        TARGET_SOURCES
        AlgorithmLauncher.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-fdmrg
        xdmrg++-algorithms-flbit
        xdmrg++-algorithms-itebd
        xdmrg++-algorithms-idmrg
        xdmrg++-algorithms-xdmrg
        )

add_library(xdmrg++-algorithms INTERFACE)
target_link_libraries(xdmrg++-algorithms INTERFACE
        xdmrg++-algorithms-base
        xdmrg++-algorithms-finite
        xdmrg++-algorithms-infinite
        xdmrg++-algorithms-launcher
        xdmrg++-algorithms-fdmrg
        xdmrg++-algorithms-flbit
        xdmrg++-algorithms-itebd
        xdmrg++-algorithms-idmrg
        xdmrg++-algorithms-xdmrg
        )
