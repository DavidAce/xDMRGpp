if(arpack-ng_FIND_REQUIRED)
    set(REQUIRED REQUIRED)
endif()

find_package(arpackng ${arpack-ng_PACKAGE_FIND_VERSION_RANGE} ${REQUIRED} CONFIG BYPASS_PROVIDER)

if(TARGET ARPACK::ARPACK)
    message(DEBUG "Detected target ARPACK::ARPACK")
    set(ARPACK_TARGET ARPACK::ARPACK)
    if(TARGET arpack AND NOT TARGET arpack-ng::arpack-ng)
        add_library(arpack-ng::arpack-ng ALIAS arpack)
    endif()
elseif(TARGET arpack-ng::arpack-ng)
    message(DEBUG "Detected target arpack-ng::arpack-ng")
    set(ARPACK_TARGET arpack-ng::arpack-ng)
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(arpack-ng
                                  VERSION_VAR PACKAGE_VERSION
                                  REQUIRED_VARS ARPACK_TARGET
                                  HANDLE_VERSION_RANGE
                                  )


if(TARGET ${ARPACK_TARGET})
    get_target_property(ARPACK_ALIASED_TARGET ${ARPACK_TARGET} ALIASED_TARGET)
    if(ARPACK_ALIASED_TARGET)
        set(ARPACK_TARGET ${ARPACK_ALIASED_TARGET})
    endif()

    find_package(gfortran BYPASS_PROVIDER)
    find_package(quadmath QUIET BYPASS_PROVIDER)

    if(TARGET gfortran::gfortran)
        target_link_libraries(${ARPACK_TARGET} INTERFACE gfortran::gfortran)
    endif()
    if(TARGET quadmath::quadmath)
        target_link_libraries(${ARPACK_TARGET} INTERFACE quadmath::quadmath)
    endif()
endif()
