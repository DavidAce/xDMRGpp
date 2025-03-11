add_source_target(xdmrg++-math-eig-matvec
        TARGET_SOURCES
        matvec_dense.cpp
        matvec_sparse.cpp
        matvec_mpo.cpp
        matvec_mpos.cpp
#        matvec_mpos_jcb_fp32.cpp
        matvec_zero.cpp
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

