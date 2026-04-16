cmake_minimum_required(VERSION 3.15)
project(external-spectra)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
    get_filename_component(DMRG_PARENT_CMAKETMP_DIR "${INIT_CACHE_FILE}" DIRECTORY)
    get_filename_component(DMRG_PARENT_CMAKEFILES_DIR "${DMRG_PARENT_CMAKETMP_DIR}" DIRECTORY)
    get_filename_component(DMRG_PARENT_BUILD_DIR "${DMRG_PARENT_CMAKEFILES_DIR}" DIRECTORY)
    set(DMRG_CONAN_CONFIG_DIR "${DMRG_PARENT_BUILD_DIR}/conan")
    if(EXISTS "${DMRG_CONAN_CONFIG_DIR}/Eigen3Config.cmake")
        set(DMRG_CONAN_EIGEN3_DIR "${DMRG_CONAN_CONFIG_DIR}")
        set(DMRG_SPECTRA_CMAKE_PREFIX_PATH_ARG -DCMAKE_PREFIX_PATH:PATH=${DMRG_CONAN_EIGEN3_DIR})
        set(DMRG_SPECTRA_EIGEN3_DIR_ARG -DEigen3_DIR:PATH=${DMRG_CONAN_EIGEN3_DIR})
    endif()
endif()
include(ExternalProject)
ExternalProject_Add(external_spectra
        URL  https://github.com/yixuan/spectra/archive/refs/tags/v1.1.0.tar.gz
        URL_MD5 aec60b8f72a511c5a4fda6d8311bb05d
        PREFIX      ${CMAKE_BINARY_DIR}
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}

        #Give Ninja access to the terminal.
        USES_TERMINAL_DOWNLOAD TRUE
        USES_TERMINAL_UPDATE TRUE
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE
        USES_TERMINAL_TEST TRUE

        BUILD_ALWAYS TRUE
        CMAKE_ARGS
        ${INIT_CACHE_ARG}
        -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_MESSAGE=NEVER
        -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=TRUE
        -DEigen3_ROOT:PATH=<INSTALL_DIR>
        ${DMRG_SPECTRA_CMAKE_PREFIX_PATH_ARG}
        ${DMRG_SPECTRA_EIGEN3_DIR_ARG}
        # spectra flags
        )
