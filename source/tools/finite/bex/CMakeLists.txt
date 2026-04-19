set(tools_finite_bex_common_deps
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction
        )

set(tools_finite_bex_state_deps
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tensors-edges-finite
        )

add_source_target(xdmrg++-tools-finite-bex-expansion-terms
        INST_TEMPLATE_FILES
            expansion_terms/expansion_terms.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_bex_common_deps}
            xdmrg++-tensors-state-finite
        )

add_source_target(xdmrg++-tools-finite-bex-mixing-terms
        INST_TEMPLATE_FILES
            mixing_terms/mixing_terms.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_bex_common_deps}
            xdmrg++-tensors-state-finite
        )

add_source_target(xdmrg++-tools-finite-bex-bond-expansion
        INST_TEMPLATE_FILES
            bond_expansion_postopt_1site/bond_expansion_postopt_1site.inst.cpp.in
            bond_expansion_preopt_1site/bond_expansion_preopt_1site.inst.cpp.in
            bond_expansion_preopt_nsite/bond_expansion_preopt_nsite.inst.cpp.in
            bond_expansion_dmrg3s/bond_expansion_dmrg3s.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_bex_common_deps}
            ${tools_finite_bex_state_deps}
            xdmrg++-tools-finite-env-edges
            xdmrg++-tools-finite-bex-expansion-terms
            xdmrg++-tools-finite-bex-mixing-terms
        )

add_source_target(xdmrg++-tools-finite-bex-mixer
        INST_TEMPLATE_FILES
            mixer/mixer.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_bex_common_deps}
            xdmrg++-tensors-finite
            xdmrg++-tools-finite-env-edges
            xdmrg++-tools-finite-opt
        )

add_source_target(xdmrg++-tools-finite-bex-density-perturbation
        INST_TEMPLATE_FILES
            density_matrix_perturbation_preopt_1site/density_matrix_perturbation_preopt_1site.inst.cpp.in
            density_matrix_perturbation_postopt_1site/density_matrix_perturbation_postopt_1site.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            ${tools_finite_bex_common_deps}
            xdmrg++-tensors-finite
            xdmrg++-tools-finite-env-edges
            xdmrg++-tools-finite-bex-expansion-terms
            xdmrg++-tools-finite-bex-mixer
        )

add_source_target(xdmrg++-tools-finite-bex
        INST_TEMPLATE_FILES
            expand_bonds/expand_bonds.inst.cpp.in
        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-config
            xdmrg++-tensors-finite
            xdmrg++-tools-finite-bex-bond-expansion
            xdmrg++-tools-finite-bex-density-perturbation
            xdmrg++-tools-common-log
        )
