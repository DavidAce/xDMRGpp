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
        TARGET_SOURCES

        AlgorithmFinite/cx32.cpp
        AlgorithmFinite/cx64.cpp
        AlgorithmFinite/cx128.cpp
        AlgorithmFinite/fp32.cpp
        AlgorithmFinite/fp64.cpp
        AlgorithmFinite/fp128.cpp

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
        TARGET_SOURCES
        AlgorithmInfinite/cx32.cpp
        AlgorithmInfinite/cx64.cpp
        AlgorithmInfinite/cx128.cpp
        AlgorithmInfinite/fp32.cpp
        AlgorithmInfinite/fp64.cpp
        AlgorithmInfinite/fp128.cpp

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
        TARGET_SOURCES
        fdmrg/cx32.cpp
        fdmrg/cx64.cpp
        fdmrg/cx128.cpp
        fdmrg/fp32.cpp
        fdmrg/fp64.cpp
        fdmrg/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-finite
        xdmrg++-tools-finite-opt
        xdmrg++-tools-common-h5
        xdmrg++-tools-common-prof

        )
add_source_target(xdmrg++-algorithms-flbit
        TARGET_SOURCES
        flbit/cx32.cpp
        flbit/cx64.cpp
        flbit/cx128.cpp
        flbit/fp32.cpp
        flbit/fp64.cpp
        flbit/fp128.cpp
        flbit/tmpl.cx32.cpp
        flbit/tmpl.cx64.cpp
        flbit/tmpl.cx128.cpp
        flbit/tmpl.fp32.cpp
        flbit/tmpl.fp64.cpp
        flbit/tmpl.fp128.cpp

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
        TARGET_SOURCES
        xdmrg/cx32.cpp
        xdmrg/cx64.cpp
        xdmrg/cx128.cpp
        xdmrg/fp32.cpp
        xdmrg/fp64.cpp
        xdmrg/fp128.cpp

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
        TARGET_SOURCES
        idmrg/cx32.cpp
        idmrg/cx64.cpp
        idmrg/cx128.cpp
        idmrg/fp32.cpp
        idmrg/fp64.cpp
        idmrg/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-algorithms-infinite
        xdmrg++-tensors-edges-infinite
        xdmrg++-tools-infinite-opt
        )

add_source_target(xdmrg++-algorithms-itebd
        TARGET_SOURCES
        itebd/cx32.cpp
        itebd/cx64.cpp
        itebd/cx128.cpp
        itebd/fp32.cpp
        itebd/fp64.cpp
        itebd/fp128.cpp

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