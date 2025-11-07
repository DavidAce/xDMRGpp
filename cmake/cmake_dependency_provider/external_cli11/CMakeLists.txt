cmake_minimum_required(VERSION 3.15)
project(external-cli11)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(ExternalProject)
ExternalProject_Add(external_cli11
#                    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.3.2.tar.gz
                    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.0.tar.gz
                    # URL_MD5 86607d04f99045d8596be0376b27fc18 # v2.1.1
                    # URL_MD5 b80cb645dee25982110b068b426363ff # v2.3.2
                    URL_MD5 ae448d3184dd6906aee2ab97d6e39c9a # v2.6.0
                    PREFIX ${CMAKE_BINARY_DIR}
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
                    # cli11 flags
                    -DCLI11_BUILD_EXAMPLES:BOOL=OFF
                    -DCLI11_BUILD_TESTS:BOOL=OFF
                    -DCLI11_BUILD_DOC:BOOL=OFF
                    )