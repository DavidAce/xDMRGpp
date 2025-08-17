cmake_minimum_required(VERSION 3.15)
project(external-LBFGSpp)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(ExternalProject)
ExternalProject_Add(external_LBFGSpp
        URL  https://github.com/yixuan/LBFGSpp/archive/refs/tags/v0.4.0.tar.gz
        URL_MD5 6edcda923aa0bfdbb2ef7f5fe7613944
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
        # LBFGSpp flags
        )