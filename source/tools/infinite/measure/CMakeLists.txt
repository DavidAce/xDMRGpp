add_source_target(xdmrg++-tools-infinite-measure
        TARGET_SOURCES
        ham/cx32.cpp
        ham/cx64.cpp
        ham/cx128.cpp
        ham/fp32.cpp
        ham/fp64.cpp
        ham/fp128.cpp

        mom/cx32.cpp
        mom/cx64.cpp
        mom/cx128.cpp
        mom/fp32.cpp
        mom/fp64.cpp
        mom/fp128.cpp

        mpo/cx32.cpp
        mpo/cx64.cpp
        mpo/cx128.cpp
        mpo/fp32.cpp
        mpo/fp64.cpp
        mpo/fp128.cpp

        measure/cx32.cpp
        measure/cx64.cpp
        measure/cx128.cpp
        measure/fp32.cpp
        measure/fp64.cpp
        measure/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-tenx
        xdmrg++-algorithms-status
        xdmrg++-tensors-infinite
        xdmrg++-tools-common-contraction
        xdmrg++-tools-common-log
        )

