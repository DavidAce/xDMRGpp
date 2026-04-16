find_package(cuTENSOR CONFIG QUIET HINTS ${cutensor_DIR})
if(cuTENSOR_FOUND)
    include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
    find_package_handle_standard_args(cuTENSOR DEFAULT_MSG cuTENSOR_CONFIG)
    return()
endif()

find_path(cuTENSOR_INCLUDE_DIR NAMES cutensor.h)
find_library(cuTENSOR_LIBRARY NAMES cutensor libcutensor.so)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(cuTENSOR REQUIRED_VARS cuTENSOR_LIBRARY cuTENSOR_INCLUDE_DIR)

if(cuTENSOR_FOUND AND NOT TARGET cuTENSOR::cuTENSOR)
    add_library(cuTENSOR::cuTENSOR UNKNOWN IMPORTED)
    set_target_properties(cuTENSOR::cuTENSOR PROPERTIES
                          IMPORTED_LOCATION "${cuTENSOR_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${cuTENSOR_INCLUDE_DIR}")
endif()
