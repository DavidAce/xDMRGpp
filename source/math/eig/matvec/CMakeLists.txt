add_source_target(xdmrg++-math-eig-matvec
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        matvec_dense/matvec_dense.inst.cpp.in
        matvec_sparse/matvec_sparse.inst.cpp.in
        matvec_zero/matvec_zero.inst.cpp.in
        matvec_mpo/matvec_mpo.inst.cpp.in
        matvec_mpos/matvec_mpos.inst.cpp.in
        INST_PAIR_TEMPLATE_FILES
        matvec_mpos/matvec_mpos.pair.inst.cpp.in

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
