cmake_minimum_required(VERSION 3.24)
project(xDMRG++
        VERSION 4.0.0
        DESCRIPTION "MPS algorithms for 1D quantum systems"
        HOMEPAGE_URL "https://github.com/DavidAce/DMRG")

include(cmake/Deprecation.cmake) # Warn if user passes deprecated settings

# Make sure we use DMRG's own find modules
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/modules)

# Set options for this build
option(DMRG_USE_QUADMATH            "Use __float128 from quadmath.h for fLBIT time evolution"           FALSE)
option(DMRG_USE_FLOAT128            "Use std::float128_t for fLBIT time evolution"                      FALSE)
option(DMRG_ENABLE_FP32             "Build explicit instantiations for fp32"                            ON)
option(DMRG_ENABLE_FP64             "Build explicit instantiations for fp64"                            ON)
option(DMRG_ENABLE_FP128            "Build explicit instantiations for fp128 aliases"                   ON)
option(DMRG_ENABLE_CX32             "Build explicit instantiations for cx32"                            ON)
option(DMRG_ENABLE_CX64             "Build explicit instantiations for cx64"                            ON)
option(DMRG_ENABLE_CX128            "Build explicit instantiations for cx128 aliases"                   ON)
option(DMRG_ENABLE_TBLIS            "Use TBLIS for faster tensor contractions (FP64) instead of Eigen3" FALSE)
option(DMRG_ENABLE_TESTS            "Enable unit testing with ctest"                                    FALSE)
option(DMRG_BUILD_EXAMPLES          "Build examples"                                                    FALSE)
option(DMRG_BUILD_TOOLS             "Build tools"                                                       FALSE)
option(DMRG_ENABLE_DOCS             "Build documentation"                                               FALSE)
option(DMRG_CMAKE_DEBUG             "Extra information during CMake configuration"                      FALSE)
option(DMRG_BENCH_CONTRACTION       "(Development) For internal development"                            FALSE)
option(DMRG_SAVE_CONTRACTION        "(Development) For internal development"                            FALSE)
option(COMPILER_PROFILE_BUILD       "(Development) Enable -ftime-trace (inspect with ClangBuildAnalyzer)" FALSE)
option(COMPILER_ENABLE_ASAN         "(Development) Enable runtime address sanitizer -fsanitize=address" FALSE)
option(COMPILER_ENABLE_USAN         "(Development) Enable undefined behavior sanitizer -fsanitize=undefined" FALSE)
option(COMPILER_ENABLE_PCH          "(Development) Enable precompiled headers to speed up compilation"  FALSE)
option(COMPILER_ENABLE_COVERAGE     "(Development) Enable test coverage"                                FALSE)
option(EIGEN_USE_THREADS            "Use STL threads to parallelize Eigen::Tensor"                      TRUE)
option(THREADS_PREFER_PTHREAD_FLAG  "Prefer -pthread flag over -lpthread or similar"                    TRUE)

set(DMRG_ENABLED_SCALARS)
foreach(dmrg_scalar IN ITEMS fp32 fp64 fp128 cx32 cx64 cx128)
    string(TOUPPER "${dmrg_scalar}" dmrg_scalar_upper)
    if(DMRG_ENABLE_${dmrg_scalar_upper})
        list(APPEND DMRG_ENABLED_SCALARS "${dmrg_scalar}")
    endif()
endforeach()

if(NOT DMRG_ENABLED_SCALARS)
    message(FATAL_ERROR "At least one explicit-instantiation scalar must be enabled")
endif()

set(DMRG_ENABLED_SCALARS "${DMRG_ENABLED_SCALARS}" CACHE INTERNAL "Enabled explicit-instantiation scalars")



# Make an "enum" for valid package managers
set(DMRG_PACKAGE_MANAGERS_VALID find cmake conan)
set(DMRG_PACKAGE_MANAGER find CACHE STRING "Package manager for external dependencies")
set_property(CACHE DMRG_PACKAGE_MANAGER PROPERTY STRINGS ${DMRG_PACKAGE_MANAGERS_VALID})
if(NOT DMRG_PACKAGE_MANAGER IN_LIST DMRG_PACKAGE_MANAGERS_VALID)
    message(FATAL_ERROR "DMRG_PACKAGE_MANAGER must be one of ${DMRG_PACKAGE_MANAGERS_VALID}")
endif()


###  Check <optional>
include(cmake/CheckCXXOptional.cmake)
checkcxxoptional()

### Print extra info during CMake configure
include(cmake/PrintBuildInfo.cmake)

###  Add targets to collect common settings
add_library(xdmrg++-flags INTERFACE)
add_library(xdmrg++-deps INTERFACE)
target_include_directories(xdmrg++-flags INTERFACE source)

###  Apply compiler flags
include(cmake/CompilerFlags.cmake)

################################################################
### Get environment, host, build, exec and git details       ###
### Generates a header env/environment.h                     ###
### Include it using #include <env/environment.h>            ###
################################################################
include(cmake/environment.cmake)


if(DMRG_USE_FLOAT128)
    include(CheckIncludeFileCXX)
    check_include_file_cxx(stdfloat DMRG_HAS_STDFLOAT)
    if(NOT DMRG_HAS_STDFLOAT)
        message(FATAL_ERROR "Could not find header <stdfloat>")
    endif()
endif()

### Create the main executable
add_executable(xDMRG++)

### Link precompiled headers. Requires DMRG_ENABLE_PCH=ON
target_link_precompiled_headers(xDMRG++)

# Setup paths that find_package should search and
# let cmake find our Find<package>.cmake modules
include(cmake/SetupPaths.cmake)

### Find or install all the dependencies
include(cmake/SetupDependencies.cmake)

###  Add all source files
add_subdirectory(source)

### Link all the things!
target_link_libraries(xDMRG++ PRIVATE xdmrg++-main)

### Enable static linking of libgcc and libstdc++. Requires BUILD_SHARED_LIBS=OFF
target_enable_static_libgcc(xDMRG++)

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DMRGUninstall.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/DMRGUninstall.cmake
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
                      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/DMRGUninstall.cmake)
endif()

if(DMRG_CMAKE_DEBUG)
    # Print summary of CMake configuration
    include(cmake/PrintTargetInfo.cmake)
    print_and_write_project_summary(xDMRG++)
endif()

if(DMRG_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests) # Unit testing with ctest and catch2
endif()
add_subdirectory(bench) # Benchmarks with ctest and nanobench
add_subdirectory(tools) # Build tools for minor DMRG related tasks
add_subdirectory(docs)  # Build documentation with doxygen

#############################
## NINJA TRACING
#############################
if(CMAKE_GENERATOR MATCHES Ninja)
    add_custom_command(
            TARGET xDMRG++ POST_BUILD
            COMMAND ninja -C ${CMAKE_BINARY_DIR} -t recompact
            COMMAND python ${CMAKE_CURRENT_LIST_DIR}/ninjatracing --showall ${CMAKE_BINARY_DIR}/.ninja_log > ${CMAKE_BINARY_DIR}/.ninja_trace.json
            BYPRODUCTS ${CMAKE_BINARY_DIR}/.ninja_trace.json
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Generating ninja build statistics"
    )
endif()
