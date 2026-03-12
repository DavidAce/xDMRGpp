add_source_target(xdmrg++-tools-finite-env
        INST_TEMPLATE_FILES
        env/env.inst.cpp.in
        env_x2/env_x2.inst.cpp.in
        bond_expansion_postopt_1site/bond_expansion_postopt_1site.inst.cpp.in
        bond_expansion_preopt_1site/bond_expansion_preopt_1site.inst.cpp.in
        bond_expansion_preopt_nsite/bond_expansion_preopt_nsite.inst.cpp.in
        bond_expansion_dmrg3s/bond_expansion_dmrg3s.inst.cpp.in
        density_matrix_perturbation_preopt_1site/density_matrix_perturbation_preopt_1site.inst.cpp.in
        density_matrix_perturbation_postopt_1site/density_matrix_perturbation_postopt_1site.inst.cpp.in
        expansion_terms/expansion_terms.inst.cpp.in
        mixing_terms/mixing_terms.inst.cpp.in
        mixer/mixer.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tensors-finite
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction
        )
