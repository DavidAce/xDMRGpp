cmake_minimum_required(VERSION 3.24)

# check_compile(<probe_name> <link_target>
#     [REQUIRED]
#     [SOURCE_FILE <path>]
#     [COMPILE_DEFINITIONS <def>...]
#     [INCLUDE_DIRECTORIES <dir>...])
#
# Compile a small translation unit against an imported target to validate that
# the target is usable from CMake's point of view, not merely discoverable.
# This is useful for vendor libraries such as BLAS backends where library
# discovery and header discovery may follow different code paths.
#
# The probe result is cached per <probe_name>/<link_target> pair so repeated
# configure runs do not recompile unless the cache entry is cleared.
function(check_compile pkg tgt)
    set(options REQUIRED)
    set(oneValueArgs SOURCE_FILE)
    set(multiValueArgs COMPILE_DEFINITIONS INCLUDE_DIRECTORIES)
    cmake_parse_arguments(PARSE_ARGV 1 CHECK "${options}" "${oneValueArgs}" "${multiValueArgs}")

    if(CHECK_SOURCE_FILE)
        set(check_source_file "${CHECK_SOURCE_FILE}")
    else()
        set(check_source_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compile/${pkg}.cpp")
    endif()

    # try_compile expects raw compiler flags rather than target-style compile
    # definitions/include directories, so normalize the caller input here.
    set(check_compile_definitions)
    foreach(check_definition IN LISTS CHECK_COMPILE_DEFINITIONS)
        if(check_definition MATCHES "^-D")
            list(APPEND check_compile_definitions "${check_definition}")
        else()
            list(APPEND check_compile_definitions "-D${check_definition}")
        endif()
    endforeach()
    foreach(check_include_directory IN LISTS CHECK_INCLUDE_DIRECTORIES)
        list(APPEND check_compile_definitions "-I${check_include_directory}")
    endforeach()

    string(MAKE_C_IDENTIFIER "${tgt}" tgt_id)
    set(check_cache_var "${pkg}_compiles_${tgt_id}")


    if(NOT DEFINED ${check_cache_var} OR NOT ${${check_cache_var}})
        message(CHECK_START "Test compile -- ${pkg} [${tgt}]")
        try_compile(${check_cache_var}
                ${CMAKE_BINARY_DIR}
                ${check_source_file}
                OUTPUT_VARIABLE compile_out
                LINK_LIBRARIES ${tgt}
                COMPILE_DEFINITIONS ${check_compile_definitions}
                CXX_STANDARD 23
                CXX_EXTENSIONS OFF
                )
        if(${check_cache_var})
            message(CHECK_PASS "Success")
            file(APPEND ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeOutput.log "${compile_out}")
            set(${check_cache_var} "${${check_cache_var}}" CACHE BOOL "" FORCE)
            mark_as_advanced(${check_cache_var})
        else()
            message(CHECK_FAIL "Failed")
            file(APPEND ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeError.log "${compile_out}")
            include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PrintTargetInfo.cmake)
            if(CHECK_REQUIRED)
                print_target_info_recursive(${tgt})
                message(FATAL_ERROR "Failed to compile ${pkg} with target [${tgt}]")
            endif()
        endif()
    endif()
endfunction()
