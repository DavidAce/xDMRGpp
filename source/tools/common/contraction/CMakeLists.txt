add_source_target(xdmrg++-tools-common-contraction
        TARGET_SOURCES
        contraction_policy.cpp
        contraction_tblis.cpp
        INST_TEMPLATE_FILES
        contract_bnd_mps.inst.cpp.in
        contract_mps_bnd.inst.cpp.in
        env/contract_env_mps_mpo.inst.cpp.in
        contract_mps_mps.inst.cpp.in
        contract_mps_mps_overlap.inst.cpp.in
        contract_mps_mps_partial.inst.cpp.in
        expectation_value/expectation_value.inst.cpp.in
        matrix_inverse_vector_product.inst.cpp.in
        matrix_vector_product/matrix_vector_product.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags

        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-tenx
        xdmrg++-tools-common-log
        )

# fp80/cx80 are only distinct from fp128/cx128 when true float128 support is
# enabled, and these wrappers intentionally instantiate a smaller overload set
# than the normal matrix_vector_product scalar template.
if(DMRG_USE_FLOAT128 OR DMRG_USE_QUADMATH)
    generate_scalar_instantiations(matrix_vector_product_ldbl_sources
            TEMPLATE matrix_vector_product/matrix_vector_product.ext80.inst.cpp.in
            SCALARS fp80 cx80)
    target_sources(xdmrg++-tools-common-contraction-o PRIVATE ${matrix_vector_product_ldbl_sources})
endif()

if (DMRG_ENABLE_TBLIS)
    target_compile_definitions(xdmrg++-tools-common-contraction-o PRIVATE DMRG_ENABLE_TBLIS)
endif()

if (DMRG_BENCH_CONTRACTION)
    target_compile_definitions(xdmrg++-tools-common-contraction-o PRIVATE DMRG_BENCH_CONTRACTION)
endif()

if (DMRG_SAVE_CONTRACTION)
    target_compile_definitions(xdmrg++-tools-common-contraction-o PRIVATE DMRG_SAVE_CONTRACTION)
endif()
