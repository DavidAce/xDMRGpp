cmake_minimum_required(VERSION 3.15)
project(external-libunwind)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)

set(cc ${CMAKE_C_COMPILER})
set(cxx ${CMAKE_CXX_COMPILER})
set(cflags "-O3 -DNDEBUG -std=c17 -fPIC -g -fno-strict-aliasing -fno-omit-frame-pointer ${CMAKE_C_FLAGS_RELEASE}")
set(cxxflags "-O3 -DNDEBUG -std=c++23 -fPIC -g -fno-strict-aliasing -fno-omit-frame-pointer ${CMAKE_CXX_FLAGS_RELEASE}")
set(libs "-lm -ldl")
if(ZLIB_ROOT)
    set(ldflags -L${ZLIB_ROOT}/lib)
endif()

externalproject_add(external_libunwind
                    URL https://github.com/libunwind/libunwind/releases/download/v1.6.2/libunwind-1.6.2.tar.gz
                    URL_MD5 f625b6a98ac1976116c71708a73dc44a
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

        BUILD_ALWAYS TRUE
        # libunwind flags

        CONFIGURE_COMMAND
        COMMAND export CC=${cc}
        COMMAND export CXX=${cxx}
        COMMAND export LIBS=${libs}
        COMMAND export CFLAGS=${cflags}
        COMMAND export CXXFLAGS=${cxxflags}
        COMMAND export LDFLAGS=${ldflags}
        COMMAND cd <SOURCE_DIR>
        COMMAND ./configure
        --prefix=<INSTALL_DIR>
        --enable-shared=yes
        --enable-static=yes
        --enable-coredump=yes
        --enable-ptrace=yes
        --enable-setjmp=yes
        --enable-minidebuginfo=no # Avoid linking to lzma
        --enable-zlibdebuginfo=no # Avoid linking to zlib
        --disable-tests
        --disable-documentation
        BUILD_COMMAND
        COMMAND cd <SOURCE_DIR>
        COMMAND make -j${NUM_THREADS}

        INSTALL_COMMAND
        COMMAND cd <SOURCE_DIR>
        COMMAND make install

        )
