cmake_minimum_required(VERSION 3.15)
project(external-eigen3)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(ExternalProject)
ExternalProject_Add(external_eigen3
        URL  https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
        URL_MD5 27cabe8e2f67230a5da6504775725ed7
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
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_MESSAGE=NEVER
        # Eigen3 flags
        -DEIGEN_TEST_CXX11:BOOL=ON
        -DBUILD_TESTING:BOOL=OFF
        -DEIGEN_BUILD_DOC:BOOL=OFF
        -DEIGEN_BUILD_PKGCONFIG:BOOL=OFF
        )