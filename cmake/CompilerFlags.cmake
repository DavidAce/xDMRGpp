cmake_minimum_required(VERSION 3.15)

cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)
if($ENV{CI} OR $ENV{GITHUB_ACTIONS})
    set(OPENBLAS_TARGET GENERIC CACHE INTERNAL "")
    set(OPENBLAS_DYNAMIC_ARCH ON CACHE INTERNAL "")
else()
    set(OPENBLAS_TARGET HASWELL CACHE INTERNAL "")
    set(OPENBLAS_DYNAMIC_ARCH ON CACHE INTERNAL "")
endif()

if(NOT TARGET xdmrg++-flags)
    add_library(xdmrg++-flags INTERFACE)
endif()

###  Add optional RELEASE/DEBUG compile to flags
target_compile_options(xdmrg++-flags INTERFACE $<$<AND:$<CONFIG:DEBUG>,$<CXX_COMPILER_ID:Clang>>: -Wconversion  -Wdouble-promotion -fstandalone-debug>)
target_compile_options(xdmrg++-flags INTERFACE $<$<AND:$<CONFIG:DEBUG>,$<CXX_COMPILER_ID:GNU>>: -Wconversion  -Wdouble-promotion>)
target_compile_options(xdmrg++-flags INTERFACE $<$<AND:$<CONFIG:RELWITHDEBINFO>,$<CXX_COMPILER_ID:Clang>>: -Wconversion  -Wdouble-promotion -fstandalone-debug>)
target_compile_options(xdmrg++-flags INTERFACE $<$<AND:$<CONFIG:RELWITHDEBINFO>,$<CXX_COMPILER_ID:GNU>>: -Wconversion  -Wdouble-promotion>)
target_compile_options(xdmrg++-flags INTERFACE
                       $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
                       $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wpedantic -Wunused -Wformat>)
###  Enable c++23 support
target_compile_features(xdmrg++-flags INTERFACE cxx_std_23)

###  Enable build profiling with ClangBuildAnalyzer
if(COMPILER_PROFILE_BUILD)
    target_compile_options(xdmrg++-flags INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-ftime-trace>)
endif()

# Settings for sanitizers
if(COMPILER_ENABLE_ASAN)
    set(ASAN_FLAGS -g -fno-omit-frame-pointer -fsanitize=address)
    mark_as_advanced(ASAN_FLAGS)
    target_compile_options(xdmrg++-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>: ${ASAN_FLAGS}>)
    target_link_libraries(xdmrg++-flags INTERFACE ${ASAN_FLAGS})
endif()
if(COMPILER_ENABLE_USAN)
    set(USAN_FLAGS -g -fno-omit-frame-pointer -fsanitize-undefined-trap-on-error -fsanitize=undefined,leak,pointer-compare,pointer-subtract,alignment,bounds,return,enum,float-cast-overflow,float-divide-by-zero,signed-integer-overflow)
    mark_as_advanced(USAN_FLAGS)
    target_compile_options(xdmrg++-flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${USAN_FLAGS}>) #  -fno-omit-frame-pointer
    target_link_libraries(xdmrg++-flags INTERFACE ${USAN_FLAGS})
endif()

if(COMPILER_ENABLE_COVERAGE)
    target_compile_options(xdmrg++-flags INTERFACE --coverage)
    target_link_options(xdmrg++-flags INTERFACE --coverage)
endif()


# Enable static linking
function(target_enable_static_libgcc tgt)
    if(BUILD_SHARED_LIBS)
        return()
    endif()
    message(STATUS "Enabling static linking on target [${tgt}]")
    target_link_options(${tgt} BEFORE PUBLIC
                        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-static-libstdc++ -static-libgcc>
                        $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-static-libgcc>
                        )
endfunction()

### Speed up compilation with precompiled headers
function(target_link_precompiled_headers tgt)
    if(COMPILER_ENABLE_PCH)
        set(PCH_HEADERS
            <string> <string_view> <vector> <array>
            <optional> <complex> <memory> <chrono> <algorithm>
            <utility> <type_traits> <cmath> <iterator> <numeric>
            <stdexcept> <filesystem>
            )

        get_target_property(type ${tgt} TYPE)
        if(type MATCHES "EXECUTABLE")
            if(NOT TARGET pch-exe)
                add_executable(pch-exe ${PROJECT_SOURCE_DIR}/cmake/pch.cpp)
                target_link_libraries(pch-exe PUBLIC xdmrg++-deps xdmrg++-flags)
                target_precompile_headers(pch-exe PUBLIC ${PCH_HEADERS})
                target_enable_static_libgcc(pch-exe)
            endif()
            target_precompile_headers(${tgt} REUSE_FROM pch-exe)
        else()
            if(NOT TARGET pch-obj)
                message(STATUS "First target for PCH: ${tgt}")
                add_library(pch-obj OBJECT ${PROJECT_SOURCE_DIR}/cmake/pch.cpp)
                target_link_libraries(pch-obj PUBLIC xdmrg++-deps xdmrg++-flags)
                target_precompile_headers(pch-obj PUBLIC ${PCH_HEADERS})
            endif()
            target_precompile_headers(${tgt} REUSE_FROM pch-obj)
        endif()
        message(TRACE "Enabled precompiled headers for target ${tgt}")
    endif()
endfunction()

# Settings for ccache
if(COMPILER_ENABLE_CCACHE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0.0")
        message(STATUS "ccache can't be enabled with clang version < 10.0.0 ")
        return()
    endif()
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "Found ccache ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(xdmrg++-flags INTERFACE -Xclang -fno-pch-timestamp -Xclang -fpch-preprocess)
        endif()
        if(COMPILER_ENABLE_PCH)
            message(STATUS "Detected ccache + pch: Remember to set --> sloppiness = include_file_mtime,pch_defines,time_macros <-- in your ccache.conf")
        endif()
    endif()
endif()

# Try to use the mold linker (incompatible with LTO!)
#function(target_enable_mold tgt)
#    if(COMPILER_ENABLE_MOLD)
#        get_target_property(LTO_IS_ON ${tgt} INTERPROCEDURAL_OPTIMIZATION)
#        if(LTO_IS_ON)
#            message(STATUS "Cannot set mold linker: LTO is enabled on target [${tgt}]")
#            return()
#        endif()
#        include(CheckLinkerFlag)
#        cmake_host_system_information(RESULT num_threads QUERY NUMBER_OF_PHYSICAL_CORES)
#        check_linker_flag(CXX "-fuse-ld=mold --thread-count=${num_threads}" LINK_MOLD)
#
#        if(LINK_MOLD)
#            target_link_options(${tgt} PUBLIC -fuse-ld=mold)
#        else()
#            message(STATUS "Cannot set mold linker: -fuse-ld=mold is not supported")
#        endif()
#    endif()
#endfunction()