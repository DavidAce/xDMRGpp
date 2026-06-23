add_source_target(xdmrg++-math-eig-primme
        TARGET_SOURCES
        solver_primme.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig-matvec
        xdmrg++-math-eig-o
        )

generate_scalar_instantiation_include(xdmrg_math_eig_primme_inst_include
        TEMPLATE solver_primme.inst.cpp.in
        SCALARS ${DMRG_ENABLED_SCALARS}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/solver_primme
        )
target_include_directories(xdmrg++-math-eig-primme-o PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/generated/solver_primme)
