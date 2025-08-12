
unset(PKG_IS_RUNNING CACHE) # Remove from cache when this file is included

function(pkg_install_dependencies  package_name)
    if(NOT PKG_INSTALL_SUCCESS AND NOT PKG_IS_RUNNING)
        message(STATUS "pkg_install_dependencies running with package_name: ${package_name}")
        unset(PKG_INSTALL_SUCCESS CACHE)
        set(PKG_IS_RUNNING TRUE CACHE INTERNAL "" FORCE)
        include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PKGInstall.cmake)

        # For the PCG random number generator
        pkg_install(pcg-cpp)

        # Iterative Eigenvalue solver for a few eigenvalues/eigenvectors using Arnoldi method.
        pkg_install(arpack-ng)

        # C++ frontend for arpack-ng. Custom find module.
        pkg_install(arpack++)

        # Eigen3 numerical library
        pkg_install(Eigen3)

        # C++ re-implementation of arpack
        pkg_install(spectra)

        # cli11 for parsing cli arguments
        pkg_install(cli11)

        # h5pp for writing to file binary in format
        pkg_install(h5pp)

        # Backward for printing pretty stack traces
        pkg_install(Backward)

#         For parsing toml config files (work in progress)
#        pkg_install(tomlplusplus)

        # For parsing toml config files (work in progress)
        pkg_install(toml11)

        # ceres-solver (for L-BFGS routine)
#        pkg_install(gflags)
#        pkg_install(glog)
        pkg_install(Ceres)

        pkg_install(primme)

        if(DMRG_ENABLE_TBLIS)
#            pkg_install(tblis)
            pkg_install(TBLIS)
        endif()

        set(PKG_INSTALL_SUCCESS TRUE CACHE BOOL "PKG dependency install has been invoked and was successful")
        set(PKG_IS_RUNNING FALSE CACHE INTERNAL "" FORCE)
    endif()

    find_package(${ARGN} BYPASS_PROVIDER)
endfunction()