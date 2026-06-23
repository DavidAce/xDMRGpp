cmake_minimum_required(VERSION 3.15)
project(external-grit)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(ExternalProject)

option(GRIT_ENABLE_32BIT "Build explicit instantiations for fp32/cx32" FALSE)
option(GRIT_ENABLE_64BIT "Build explicit instantiations for fp64/cx64" TRUE)
option(GRIT_ENABLE_80BIT "Build explicit instantiations for fp80/cx80" FALSE)
option(GRIT_ENABLE_128BIT "Build explicit instantiations for std::float128_t/cx128" FALSE)

ExternalProject_Add(external_grit
                    GIT_REPOSITORY https://github.com/DavidAce/grit.git
                    GIT_TAG main
                    GIT_PROGRESS TRUE
                    GIT_SHALLOW TRUE
                    GIT_SUBMODULES_RECURSE TRUE
                    PREFIX ${CMAKE_BINARY_DIR}
                    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
                    DOWNLOAD_EXTRACT_TIMESTAMP TRUE

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
                    -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=TRUE
                    -DGRIT_PACKAGE_MANAGER:STRING=find
                    -DGRIT_ENABLE_TESTS:BOOL=OFF
                    -DGRIT_BUILD_EXAMPLES:BOOL=OFF
                    -DGRIT_ENABLE_BENCH:BOOL=OFF
                    -DGRIT_ENABLE_32BIT:BOOL=${GRIT_ENABLE_32BIT}
                    -DGRIT_ENABLE_64BIT:BOOL=${GRIT_ENABLE_64BIT}
                    -DGRIT_ENABLE_80BIT:BOOL=${GRIT_ENABLE_80BIT}
                    -DGRIT_ENABLE_128BIT:BOOL=${GRIT_ENABLE_128BIT}
                    )
