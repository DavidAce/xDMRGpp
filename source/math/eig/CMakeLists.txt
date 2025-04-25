
add_subdirectory(solver_arpack)
add_subdirectory(solver_spectra)
add_subdirectory(solver_primme)
add_subdirectory(solver_lapacke)
add_subdirectory(matvec)

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
        xdmrg++-math-eig-lapacke
        xdmrg++-math-eig-primme
        xdmrg++-math-eig-spectra
        )
