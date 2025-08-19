cmake_minimum_required(VERSION 3.15)
project(external-TBLIS)
cmake_host_system_information(RESULT NUM_THREADS QUERY NUMBER_OF_PHYSICAL_CORES)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(ExternalProject)

option(ENABLE_TESTS "Build tests" OFF)
option(ENABLE_SHARED "Build a shared library" ON)
option(ENABLE_STATIC "Build a static library" ON)
option(ENABLE_COMPAT "Enable compatibility with the TBLIS 1.x interface" ON)
option(ENABLE_MEMKIND OFF "Enable use of the memkind library for MCDRAM allocation if supported.")
option(ENABLE_HWLOC ON "Enable use of the memkind library for MCDRAM allocation if supported.")


set(cc ${CMAKE_C_COMPILER})
set(cxx ${CMAKE_CXX_COMPILER})
set(cflags "-O3 -DNDEBUG -std=c17 -fPIC ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
set(cxxflags "-O3 -DNDEBUG -std=c++23 -fPIC ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
set(ldflags "${CMAKE_EXE_LINKER_FLAGS}")
set(libs "-lm -ldl")
if(TBLIS_THREAD_MODEL)
    set(thread_model "--enable-thread-model=${TBLIS_THREAD_MODEL}")
endif()

set(ENV CC ${cc})
set(ENV CXX ${cxx})
set(ENV LIBS ${libs})
set(ENV CFLAGS ${cflags})
set(ENV CXXFLAGS ${cxxflags})
set(ENV LDFLAGS ${ldflags})


ExternalProject_Add(external_TBLIS
                    GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
                    GIT_TAG 9b95712966cb8804be51c62bfd6207957f62bc6f # develop on 2025-08-19
                    GIT_PROGRESS TRUE
                    GIT_SHALLOW TRUE
                    GIT_SUBMODULES_RECURSE TRUE
                    PREFIX ${CMAKE_BINARY_DIR}
                    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
                    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                    PATCH_COMMAND patch -p1 -i ${PROJECT_SOURCE_DIR}/tblis-develop-9b95712.patch || true

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
                    # TBLIS flags
                    -DENABLE_TESTS:BOOL=${ENABLE_TESTS}
                    -DENABLE_SHARED:BOOL=${ENABLE_SHARED}
                    -DENABLE_STATIC:BOOL=${ENABLE_STATIC}
                    -DENABLE_COMPAT:BOOL=${ENABLE_COMPAT}
                    -DENABLE_MEMKIND:BOOL=${ENABLE_MEMKIND}
                    -DENABLE_HWLOC:BOOL=${ENABLE_HWLOC}
                    -DENABLE_THREAD_MODEL=${TBLIS_THREAD_MODEL}
                    )