cmake_minimum_required(VERSION 3.15)
project(external-primme)
cmake_host_system_information(RESULT NUM_THREADS QUERY NUMBER_OF_PHYSICAL_CORES)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(ExternalProject)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(cflags "-O0 -g -std=c17 -fPIC ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -DPRIMME_BLASINT_SIZE=32 -DPRIMME_INT_SIZE=0")
    set(cxxflags "-O0 -g -std=c++23 -fPIC ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -DPRIMME_BLASINT_SIZE=32 -DPRIMME_INT_SIZE=0")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    set(cflags "-O3 -DNDEBUG -std=c17 -fPIC -g ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -DPRIMME_BLASINT_SIZE=32 -DPRIMME_INT_SIZE=0")
    set(cxxflags "-O3 -DNDEBUG -std=c++23 -fPIC -g ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE} -DPRIMME_BLASINT_SIZE=32 -DPRIMME_INT_SIZE=0")
endif()


if(BUILD_SHARED_LIBS)
    set(type solib)
else()
    set(type lib)
endif()

externalproject_add(external_primme
                    URL https://github.com/primme/primme/archive/refs/tags/v3.2.tar.gz
                    URL_MD5 b26968d0ea8aa2e6feefc89f3c863062
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
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND
                    COMMAND cd <SOURCE_DIR>
                    COMMAND make ${type} -j${NUM_THREADS} VERBOSE=1 CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER} CXXFLAGS=${cxxflags} CFLAGS=${cflags}

                    INSTALL_COMMAND
                    COMMAND cd <SOURCE_DIR>
                    COMMAND make install includedir=<INSTALL_DIR>/include/primme libdir=<INSTALL_DIR>/lib
                    COMMAND make clean
                    )

if(NOT BUILD_SHARED_LIBS)
    externalproject_add_step(
            external_primme
            install_static_lib
            DEPENDEES build install
            COMMENT "Installing static lib"
            COMMAND cmake -E copy <SOURCE_DIR>/lib/libprimme${CMAKE_STATIC_LIBRARY_SUFFIX} <INSTALL_DIR>/lib/libprimme${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()
