cmake_minimum_required(VERSION 3.21)
project(external-toml11)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)

ExternalProject_Add(external_toml11
#        URL https://github.com/ToruNiina/toml11/archive/refs/tags/v4.3.0.tar.gz
        URL https://github.com/ToruNiina/toml11/archive/refs/tags/v4.4.0.tar.gz
#        URL_MD5 0235d45af6aa3ef4e70f22d99598b92e
        URL_MD5 b576218153c21742e54920c061ff33cb
        BUILD_ALWAYS TRUE
        PREFIX ${CMAKE_BINARY_DIR}
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
        #Give Ninja access to the terminal.
        USES_TERMINAL_DOWNLOAD TRUE
        USES_TERMINAL_UPDATE TRUE
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE
        USES_TERMINAL_TEST TRUE

        CMAKE_ARGS
        ${INIT_CACHE_ARG}
        -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_RPATH:PATH=<INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
        -DCMAKE_INSTALL_MESSAGE=LAZY
        -DTOML11_PRECOMPILE:BOOL=TRUE
        -DTOML11_COMPILE_SOURCES:BOOL=TRUE
        )

