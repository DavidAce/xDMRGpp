add_source_target(xdmrg++-math-eig-arpack
        TARGET_SOURCES
        solver_arpack.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig-matvec
        xdmrg++-math-eig-o
        )

generate_scalar_instantiation_include(xdmrg_math_eig_arpack_inst_include
        TEMPLATE solver_arpack.inst.cpp.in
        SCALARS ${DMRG_ENABLED_SCALARS}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/solver_arpack
        )
target_include_directories(xdmrg++-math-eig-arpack-o PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/generated/solver_arpack)

# This one is special, since it absolutely can't have BLAS/LAPACK headers during compile for some reason
set_source_files_properties(solver_arpack.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
