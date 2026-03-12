add_source_target(xdmrg++-tools-finite-opt
        TARGET_SOURCES
            opt/opt.cpp
            opt_meta.cpp

        INST_TEMPLATE_FILES
            filter_subspace/filter_subspace.inst.cpp.in
            find_subspace/find_subspace.inst.cpp.in
            optimize_energy/eig/optimize_energy_eig.inst.cpp.in
            optimize_energy/eigs/optimize_energy.inst.cpp.in
            optimize_folded_spectrum/eig/optimize_folded_spectrum_eig.inst.cpp.in
            optimize_folded_spectrum/eigs/optimize_folded_spectrum.inst.cpp.in
            optimize_generalized_shift_invert/eig/optimize_generalized_shift_invert_eig.inst.cpp.in
            optimize_generalized_shift_invert/eigs/optimize_generalized_shift_invert.inst.cpp.in
            optimize_overlap/optimize_overlap.inst.cpp.in
            optimize_subspace_variance/optimize_subspace_variance.inst.cpp.in
            precond/common/common.inst.cpp.in
            precond/standard_basis_change/standard_basis_change.inst.cpp.in
            precond/generalized_basis_change/generalized_basis_change.inst.cpp.in
            opt_mps/opt_mps.inst.cpp.in
            opt/opt.inst.cpp.in
            report/report.inst.cpp.in

        INST_PAIR_TEMPLATE_FILES
            extract_results/extract_results.pair.inst.cpp.in

        OBJECT_LINK_LIBRARIES
            xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
            xdmrg++-tid
            xdmrg++-config
            xdmrg++-math-eig
            xdmrg++-algorithms-status
            xdmrg++-tensors-finite
            xdmrg++-tools-finite-measure
            xdmrg++-tools-common-log
            xdmrg++-tools-common-contraction
        )
