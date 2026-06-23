set(tools_finite_env_common_deps
        xdmrg++-tid
        xdmrg++-config
        xdmrg++-math-svd
        xdmrg++-math-tenx
        xdmrg++-tools-common-log
        xdmrg++-tools-common-contraction
        )

set(tools_finite_env_state_deps
        xdmrg++-tensors-state-finite
        xdmrg++-tensors-model-finite
        xdmrg++-tensors-edges-finite
        )

add_source_target(xdmrg++-tools-finite-env-edges
        INST_TEMPLATE_FILES
        env/env.inst.cpp.in
        env_x2/env_x2.inst.cpp.in
        INST_SCALARS
        ${DMRG_ENABLED_SCALARS}

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        ${tools_finite_env_common_deps}
        ${tools_finite_env_state_deps}
        )

add_library(xdmrg++-tools-finite-env INTERFACE)
target_link_libraries(xdmrg++-tools-finite-env INTERFACE
        xdmrg++-tools-finite-env-edges
        )
