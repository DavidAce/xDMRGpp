cmake_minimum_required(VERSION 3.15)
project(external-arpack-ng)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)

enable_language(C Fortran)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_EQUAL 12)
    message(WARNING "FortranCInterface check is broken on GCC 12. ")
endif()

include(FortranCInterface)
fortrancinterface_verify()

externalproject_add(external_arpack-ng
                    URL https://github.com/opencollab/arpack-ng/archive/refs/tags/3.9.0.tar.gz
                    URL_MD5 e28fdbe33ee44a16e2733c180ec2a2bd
                    BUILD_ALWAYS TRUE
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

                    UPDATE_COMMAND ""

                    CMAKE_ARGS
                    ${INIT_CACHE_ARG}
                    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
                    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                    -DCMAKE_INSTALL_RPATH:PATH=<INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}
                    #-DCMAKE_SKIP_BUILD_RPATH:BOOL=OFF
                    #-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=OFF
                    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
                    -DCMAKE_INSTALL_MESSAGE=LAZY
                    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

                    # Arpack flags
                    -DEXAMPLES=OFF
                    -DTESTS=OFF
                    -DMPI=OFF
                    -DINTERFACE64=OFF

                    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
                    )