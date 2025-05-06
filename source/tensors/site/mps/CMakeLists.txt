add_source_target(xdmrg++-tensors-mps
        TARGET_SOURCES
        MpsSite/cx32.cpp
        MpsSite/cx64.cpp
        MpsSite/cx128.cpp
        MpsSite/fp32.cpp
        MpsSite/fp64.cpp
        MpsSite/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-debug
        xdmrg++-math-tenx
        xdmrg++-math-hash
        xdmrg++-tools-common-contraction
        xdmrg++-tools-common-log
        )