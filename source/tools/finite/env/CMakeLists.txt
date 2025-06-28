add_source_target(xdmrg++-tools-finite-env
        TARGET_SOURCES
        cx32.cpp
        cx64.cpp
        cx128.cpp
        fp32.cpp
        fp64.cpp
        fp128.cpp

        bond_expansion_postopt_1site/cx32.cpp
        bond_expansion_postopt_1site/cx64.cpp
        bond_expansion_postopt_1site/cx128.cpp
        bond_expansion_postopt_1site/fp32.cpp
        bond_expansion_postopt_1site/fp64.cpp
        bond_expansion_postopt_1site/fp128.cpp

        bond_expansion_preopt_1site/cx32.cpp
        bond_expansion_preopt_1site/cx64.cpp
        bond_expansion_preopt_1site/cx128.cpp
        bond_expansion_preopt_1site/fp32.cpp
        bond_expansion_preopt_1site/fp64.cpp
        bond_expansion_preopt_1site/fp128.cpp

        bond_expansion_preopt_nsite/cx32.cpp
        bond_expansion_preopt_nsite/cx64.cpp
        bond_expansion_preopt_nsite/cx128.cpp
        bond_expansion_preopt_nsite/fp32.cpp
        bond_expansion_preopt_nsite/fp64.cpp
        bond_expansion_preopt_nsite/fp128.cpp

        bond_expansion_dmrg3s/cx32.cpp
        bond_expansion_dmrg3s/cx64.cpp
        bond_expansion_dmrg3s/cx128.cpp
        bond_expansion_dmrg3s/fp32.cpp
        bond_expansion_dmrg3s/fp64.cpp
        bond_expansion_dmrg3s/fp128.cpp

        density_matrix_perturbation_preopt_1site/cx32.cpp
        density_matrix_perturbation_preopt_1site/cx64.cpp
        density_matrix_perturbation_preopt_1site/cx128.cpp
        density_matrix_perturbation_preopt_1site/fp32.cpp
        density_matrix_perturbation_preopt_1site/fp64.cpp
        density_matrix_perturbation_preopt_1site/fp128.cpp

        density_matrix_perturbation_postopt_1site/cx32.cpp
        density_matrix_perturbation_postopt_1site/cx64.cpp
        density_matrix_perturbation_postopt_1site/cx128.cpp
        density_matrix_perturbation_postopt_1site/fp32.cpp
        density_matrix_perturbation_postopt_1site/fp64.cpp
        density_matrix_perturbation_postopt_1site/fp128.cpp

        expansion_terms/cx32.cpp
        expansion_terms/cx64.cpp
        expansion_terms/cx128.cpp
        expansion_terms/fp32.cpp
        expansion_terms/fp64.cpp
        expansion_terms/fp128.cpp

        mixing_terms/cx32.cpp
        mixing_terms/cx64.cpp
        mixing_terms/cx128.cpp
        mixing_terms/fp32.cpp
        mixing_terms/fp64.cpp
        mixing_terms/fp128.cpp

        mixer/cx32.cpp
        mixer/cx64.cpp
        mixer/cx128.cpp
        mixer/fp32.cpp
        mixer/fp64.cpp
        mixer/fp128.cpp

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

