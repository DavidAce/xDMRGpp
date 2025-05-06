add_source_target(xdmrg++-math-eig-matvec
        TARGET_SOURCES
        matvec_dense/cx32.cpp
        matvec_dense/cx64.cpp
        matvec_dense/cx128.cpp
        matvec_dense/fp32.cpp
        matvec_dense/fp64.cpp
        matvec_dense/fp128.cpp

        matvec_sparse/cx32.cpp
        matvec_sparse/cx64.cpp
        matvec_sparse/cx128.cpp
        matvec_sparse/fp32.cpp
        matvec_sparse/fp64.cpp
        matvec_sparse/fp128.cpp

        matvec_zero/cx32.cpp
        matvec_zero/cx64.cpp
        matvec_zero/cx128.cpp
        matvec_zero/fp32.cpp
        matvec_zero/fp64.cpp
        matvec_zero/fp128.cpp

        matvec_mpo/cx32.cpp
        matvec_mpo/cx64.cpp
        matvec_mpo/cx128.cpp
        matvec_mpo/fp32.cpp
        matvec_mpo/fp64.cpp
        matvec_mpo/fp128.cpp

        matvec_mpos/fp32.cpp
        matvec_mpos/fp64.cpp
        matvec_mpos/fp128.cpp
        matvec_mpos/cx32.cpp
        matvec_mpos/cx64.cpp
        matvec_mpos/cx128.cpp


        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-tensors-mps
        xdmrg++-tensors-mpo
        xdmrg++-tensors-env
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tools-common-contraction
        xdmrg++-math-eig-o
        )

