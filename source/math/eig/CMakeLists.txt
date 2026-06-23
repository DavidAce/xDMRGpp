
add_subdirectory(solver_arpack)
add_subdirectory(solver_spectra)
add_subdirectory(solver_primme)
add_subdirectory(solver_lapack)
add_subdirectory(solver_eigsmpo)
add_subdirectory(matvec)

generate_scalar_instantiation_include(xdmrg_math_eig_solver_inst_include
        TEMPLATE solver.inst.cpp.in
        SCALARS ${DMRG_ENABLED_SCALARS}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/solver
        )

add_source_target(xdmrg++-math-eig
        TARGET_SOURCES
        log.cpp
        settings.cpp
        solver.cpp
        solution.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig-matvec
        xdmrg++-math-eig-arpack
        xdmrg++-math-eig-lapack
        xdmrg++-math-eig-primme
        xdmrg++-math-eig-spectra
        xdmrg++-math-eig-eigsmpo
        )

target_include_directories(xdmrg++-math-eig-o PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/generated/solver)
