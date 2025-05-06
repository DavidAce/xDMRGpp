add_source_target(xdmrg++-tensors-env
        TARGET_SOURCES
        EnvPair.cpp
        EnvBase/cx32.cpp
        EnvBase/cx64.cpp
        EnvBase/cx128.cpp
        EnvBase/fp32.cpp
        EnvBase/fp64.cpp
        EnvBase/fp128.cpp

        EnvEne/cx32.cpp
        EnvEne/cx64.cpp
        EnvEne/cx128.cpp
        EnvEne/fp32.cpp
        EnvEne/fp64.cpp
        EnvEne/fp128.cpp

        EnvVar/cx32.cpp
        EnvVar/cx64.cpp
        EnvVar/cx128.cpp
        EnvVar/fp32.cpp
        EnvVar/fp64.cpp
        EnvVar/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-math-hash
        xdmrg++-tensors-mps
        xdmrg++-tensors-mpo
        xdmrg++-tensors-env
        xdmrg++-tools-common-log
        )