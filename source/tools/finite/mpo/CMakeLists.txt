add_source_target(xdmrg++-tools-finite-mpo
        TARGET_SOURCES
            compress/cx32.cpp
            compress/cx64.cpp
            compress/cx128.cpp
            compress/fp32.cpp
            compress/fp64.cpp
            compress/fp128.cpp

            edge/cx32.cpp
            edge/cx64.cpp
            edge/cx128.cpp
            edge/fp32.cpp
            edge/fp64.cpp
            edge/fp128.cpp

            merge/cx32.cpp
            merge/cx64.cpp
            merge/cx128.cpp
            merge/fp32.cpp
            merge/fp64.cpp
            merge/fp128.cpp

            swap/cx32.cpp
            swap/cx64.cpp
            swap/cx128.cpp
            swap/fp32.cpp
            swap/fp64.cpp
            swap/fp128.cpp

        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-math-svd
            xdmrg++-math-tenx
            xdmrg++-tensors-mpo
            xdmrg++-tools-common-log
            xdmrg++-tools-common-contraction
            xdmrg++-qm-spin
        )