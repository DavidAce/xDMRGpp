add_source_target(xdmrg++-tools-finite-opt
        TARGET_SOURCES

            extract_results/fp32/cx32.cpp
            extract_results/fp32/cx64.cpp
            extract_results/fp32/cx128.cpp
            extract_results/fp32/fp32.cpp
            extract_results/fp32/fp64.cpp
            extract_results/fp32/fp128.cpp
            extract_results/fp64/cx32.cpp
            extract_results/fp64/cx64.cpp
            extract_results/fp64/cx128.cpp
            extract_results/fp64/fp32.cpp
            extract_results/fp64/fp64.cpp
            extract_results/fp64/fp128.cpp
            extract_results/fp128/cx32.cpp
            extract_results/fp128/cx64.cpp
            extract_results/fp128/cx128.cpp
            extract_results/fp128/fp32.cpp
            extract_results/fp128/fp64.cpp
            extract_results/fp128/fp128.cpp
            extract_results/cx32/cx32.cpp
            extract_results/cx32/cx64.cpp
            extract_results/cx32/cx128.cpp
            extract_results/cx32/fp32.cpp
            extract_results/cx32/fp64.cpp
            extract_results/cx32/fp128.cpp
            extract_results/cx64/cx32.cpp
            extract_results/cx64/cx64.cpp
            extract_results/cx64/cx128.cpp
            extract_results/cx64/fp32.cpp
            extract_results/cx64/fp64.cpp
            extract_results/cx64/fp128.cpp
            extract_results/cx128/cx32.cpp
            extract_results/cx128/cx64.cpp
            extract_results/cx128/cx128.cpp
            extract_results/cx128/fp32.cpp
            extract_results/cx128/fp64.cpp
            extract_results/cx128/fp128.cpp


            filter_subspace/cx32.cpp
            filter_subspace/cx64.cpp
            filter_subspace/cx128.cpp
            filter_subspace/fp32.cpp
            filter_subspace/fp64.cpp
            filter_subspace/fp128.cpp

            find_subspace/cx32.cpp
            find_subspace/cx64.cpp
            find_subspace/cx128.cpp
            find_subspace/fp32.cpp
            find_subspace/fp64.cpp
            find_subspace/fp128.cpp

            optimize_energy/eig/cx32.cpp
            optimize_energy/eig/cx64.cpp
            optimize_energy/eig/cx128.cpp
            optimize_energy/eig/fp32.cpp
            optimize_energy/eig/fp64.cpp
            optimize_energy/eig/fp128.cpp

            optimize_energy/eigs/cx32.cpp
            optimize_energy/eigs/cx64.cpp
            optimize_energy/eigs/cx128.cpp
            optimize_energy/eigs/fp32.cpp
            optimize_energy/eigs/fp64.cpp
            optimize_energy/eigs/fp128.cpp

            optimize_folded_spectrum/eig/cx32.cpp
            optimize_folded_spectrum/eig/cx64.cpp
            optimize_folded_spectrum/eig/cx128.cpp
            optimize_folded_spectrum/eig/fp32.cpp
            optimize_folded_spectrum/eig/fp64.cpp
            optimize_folded_spectrum/eig/fp128.cpp

            optimize_folded_spectrum/eigs/cx32.cpp
            optimize_folded_spectrum/eigs/cx64.cpp
            optimize_folded_spectrum/eigs/cx128.cpp
            optimize_folded_spectrum/eigs/fp32.cpp
            optimize_folded_spectrum/eigs/fp64.cpp
            optimize_folded_spectrum/eigs/fp128.cpp

            optimize_generalized_shift_invert/eig/cx32.cpp
            optimize_generalized_shift_invert/eig/cx64.cpp
            optimize_generalized_shift_invert/eig/cx128.cpp
            optimize_generalized_shift_invert/eig/fp32.cpp
            optimize_generalized_shift_invert/eig/fp64.cpp
            optimize_generalized_shift_invert/eig/fp128.cpp

            optimize_generalized_shift_invert/eigs/cx32.cpp
            optimize_generalized_shift_invert/eigs/cx64.cpp
            optimize_generalized_shift_invert/eigs/cx128.cpp
            optimize_generalized_shift_invert/eigs/fp32.cpp
            optimize_generalized_shift_invert/eigs/fp64.cpp
            optimize_generalized_shift_invert/eigs/fp128.cpp

            optimize_overlap/cx32.cpp
            optimize_overlap/cx64.cpp
            optimize_overlap/cx128.cpp
            optimize_overlap/fp32.cpp
            optimize_overlap/fp64.cpp
            optimize_overlap/fp128.cpp

            optimize_subspace_variance/cx32.cpp
            optimize_subspace_variance/cx64.cpp
            optimize_subspace_variance/cx128.cpp
            optimize_subspace_variance/fp32.cpp
            optimize_subspace_variance/fp64.cpp
            optimize_subspace_variance/fp128.cpp

            optimize_lanczos_h1h2/cx32.cpp
            optimize_lanczos_h1h2/cx64.cpp
            optimize_lanczos_h1h2/cx128.cpp
            optimize_lanczos_h1h2/fp32.cpp
            optimize_lanczos_h1h2/fp64.cpp
            optimize_lanczos_h1h2/fp128.cpp

            optimize_lanczos_h1h2/SolverBase/cx32.cpp
            optimize_lanczos_h1h2/SolverBase/cx64.cpp
            optimize_lanczos_h1h2/SolverBase/cx128.cpp
            optimize_lanczos_h1h2/SolverBase/fp32.cpp
            optimize_lanczos_h1h2/SolverBase/fp64.cpp
            optimize_lanczos_h1h2/SolverBase/fp128.cpp

            optimize_lanczos_h1h2/LOBPCG/cx32.cpp
            optimize_lanczos_h1h2/LOBPCG/cx64.cpp
            optimize_lanczos_h1h2/LOBPCG/cx128.cpp
            optimize_lanczos_h1h2/LOBPCG/fp32.cpp
            optimize_lanczos_h1h2/LOBPCG/fp64.cpp
            optimize_lanczos_h1h2/LOBPCG/fp128.cpp

            optimize_lanczos_h1h2/GD/cx32.cpp
            optimize_lanczos_h1h2/GD/cx64.cpp
            optimize_lanczos_h1h2/GD/cx128.cpp
            optimize_lanczos_h1h2/GD/fp32.cpp
            optimize_lanczos_h1h2/GD/fp64.cpp
            optimize_lanczos_h1h2/GD/fp128.cpp

            optimize_lanczos_h1h2/BlockLanczos/cx32.cpp
            optimize_lanczos_h1h2/BlockLanczos/cx64.cpp
            optimize_lanczos_h1h2/BlockLanczos/cx128.cpp
            optimize_lanczos_h1h2/BlockLanczos/fp32.cpp
            optimize_lanczos_h1h2/BlockLanczos/fp64.cpp
            optimize_lanczos_h1h2/BlockLanczos/fp128.cpp

            opt_mps/cx32.cpp
            opt_mps/cx64.cpp
            opt_mps/cx128.cpp
            opt_mps/fp32.cpp
            opt_mps/fp64.cpp
            opt_mps/fp128.cpp

            opt/opt.cpp
            opt/cx32.cpp
            opt/cx64.cpp
            opt/cx128.cpp
            opt/fp32.cpp
            opt/fp64.cpp
            opt/fp128.cpp

            report/cx32.cpp
            report/cx64.cpp
            report/cx128.cpp
            report/fp32.cpp
            report/fp64.cpp
            report/fp128.cpp

            opt_meta.cpp



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

