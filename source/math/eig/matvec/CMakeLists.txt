set(math_eig_matvec_common_deps
        xdmrg++-tid
        xdmrg++-tensors-mps
        xdmrg++-tensors-mpo
        xdmrg++-tensors-env
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tools-common-contraction
        xdmrg++-math-eig-o
        )

add_source_target(xdmrg++-math-eig-matvec-dense
        INST_TEMPLATE_FILES
        matvec_dense/matvec_dense.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${math_eig_matvec_common_deps}
        )

add_source_target(xdmrg++-math-eig-matvec-sparse
        INST_TEMPLATE_FILES
        matvec_sparse/matvec_sparse.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${math_eig_matvec_common_deps}
        )

add_source_target(xdmrg++-math-eig-matvec-zero
        INST_TEMPLATE_FILES
        matvec_zero/matvec_zero.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${math_eig_matvec_common_deps}
        )

add_source_target(xdmrg++-math-eig-matvec-mpo
        INST_TEMPLATE_FILES
        matvec_mpo/matvec_mpo.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${math_eig_matvec_common_deps}
        )

add_source_target(xdmrg++-math-eig-matvec-mpos
        INST_TEMPLATE_FILES
        matvec_mpos/matvec_mpos.inst.cpp.in
        matvec_mpos/internal/get_op_norm.inst.cpp.in
        INST_PAIR_TEMPLATE_FILES
        matvec_mpos/matvec_mpos.pair.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${math_eig_matvec_common_deps}
        )

add_library(xdmrg++-math-eig-matvec INTERFACE)
target_link_libraries(xdmrg++-math-eig-matvec INTERFACE
        xdmrg++-math-eig-matvec-dense
        xdmrg++-math-eig-matvec-sparse
        xdmrg++-math-eig-matvec-zero
        xdmrg++-math-eig-matvec-mpo
        xdmrg++-math-eig-matvec-mpos
        )
