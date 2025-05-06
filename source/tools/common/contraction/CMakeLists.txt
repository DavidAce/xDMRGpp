add_source_target(xdmrg++-tools-common-contraction
        TARGET_SOURCES
        contract_bnd_mps.cx32.cpp
        contract_bnd_mps.cx64.cpp
        contract_bnd_mps.cx128.cpp
        contract_bnd_mps.fp32.cpp
        contract_bnd_mps.fp64.cpp
        contract_bnd_mps.fp128.cpp

        contract_env_mps_mpo.cx32.cpp
        contract_env_mps_mpo.cx64.cpp
        contract_env_mps_mpo.cx128.cpp
        contract_env_mps_mpo.fp32.cpp
        contract_env_mps_mpo.fp64.cpp
        contract_env_mps_mpo.fp128.cpp

        contract_mps_bnd.cx32.cpp
        contract_mps_bnd.cx64.cpp
        contract_mps_bnd.cx128.cpp
        contract_mps_bnd.fp32.cpp
        contract_mps_bnd.fp64.cpp
        contract_mps_bnd.fp128.cpp

        contract_mps_mpo_env.cx32.cpp
        contract_mps_mpo_env.cx64.cpp
        contract_mps_mpo_env.cx128.cpp
        contract_mps_mpo_env.fp32.cpp
        contract_mps_mpo_env.fp64.cpp
        contract_mps_mpo_env.fp128.cpp

        contract_mps_mps.cx32.cpp
        contract_mps_mps.cx64.cpp
        contract_mps_mps.cx128.cpp
        contract_mps_mps.fp32.cpp
        contract_mps_mps.fp64.cpp
        contract_mps_mps.fp128.cpp

        contract_mps_mps_overlap.cx32.cpp
        contract_mps_mps_overlap.cx64.cpp
        contract_mps_mps_overlap.cx128.cpp
        contract_mps_mps_overlap.fp32.cpp
        contract_mps_mps_overlap.fp64.cpp
        contract_mps_mps_overlap.fp128.cpp

        contract_mps_mps_partial.cx32.cpp
        contract_mps_mps_partial.cx64.cpp
        contract_mps_mps_partial.cx128.cpp
        contract_mps_mps_partial.fp32.cpp
        contract_mps_mps_partial.fp64.cpp
        contract_mps_mps_partial.fp128.cpp

        contraction_tblis.cpp
        expectation_value.cx32.cpp
        expectation_value.cx64.cpp
        expectation_value.cx128.cpp
        expectation_value.fp32.cpp
        expectation_value.fp64.cpp
        expectation_value.fp128.cpp

        matrix_inverse_vector_product.cx32.cpp
        matrix_inverse_vector_product.cx64.cpp
        matrix_inverse_vector_product.cx128.cpp
        matrix_inverse_vector_product.fp32.cpp
        matrix_inverse_vector_product.fp64.cpp
        matrix_inverse_vector_product.fp128.cpp

        matrix_vector_product.cx32.cpp
        matrix_vector_product.cx64.cpp
        matrix_vector_product.cx128.cpp
        matrix_vector_product.fp32.cpp
        matrix_vector_product.fp64.cpp
        matrix_vector_product.fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags

        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-tenx
        xdmrg++-tools-common-log
        )

target_sources(xdmrg++-tools-common-contraction-o PRIVATE contraction_tblis.cpp)

if (DMRG_ENABLE_TBLIS)
    target_compile_definitions(xdmrg++-tools-common-contraction-o PRIVATE DMRG_ENABLE_TBLIS)
endif()

if (DMRG_BENCH_CONTRACTION)
    target_compile_definitions(xdmrg++-tools-common-contraction-o PRIVATE DMRG_BENCH_CONTRACTION)
endif()

if (DMRG_SAVE_CONTRACTION)
    target_compile_definitions(xdmrg++-tools-common-contraction-o PRIVATE DMRG_SAVE_CONTRACTION)
endif()
