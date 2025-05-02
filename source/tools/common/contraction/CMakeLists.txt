add_source_target(xdmrg++-tools-common-contraction
        TARGET_SOURCES
        contract_bnd_mps.cpp
        contract_mps_bnd.cpp

        contract_env_mps_mpo.cx32.cpp
        contract_env_mps_mpo.cx64.cpp
        contract_env_mps_mpo.cx128.cpp
        contract_env_mps_mpo.fp32.cpp
        contract_env_mps_mpo.fp64.cpp
        contract_env_mps_mpo.fp128.cpp

        contract_mps_mpo_env.cx32.cpp
        contract_mps_mpo_env.cx64.cpp
        contract_mps_mpo_env.cx128.cpp
        contract_mps_mpo_env.fp32.cpp
        contract_mps_mpo_env.fp64.cpp
        contract_mps_mpo_env.fp128.cpp

        contract_mps_mps.cpp
        contract_mps_mps_overlap.cpp
        contract_mps_mps_partial.cpp
        expectation_value.cpp
        matrix_inverse_vector_product.cpp
        matrix_vector_product.cpp

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
