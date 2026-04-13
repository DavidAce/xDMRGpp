
function(find_arpackpp)
    list(APPEND arpack_target_candidates arpack arpack-ng ARPACK::ARPACK arpack-ng::arpack-ng arpack::arpack)
    foreach(tgt ${arpack_target_candidates})
        if(TARGET ${tgt})
            get_target_property(arpack_aliased ${tgt} ALIASED_TARGET)
            if(arpack_aliased)
                set(arpack_target_name ${arpack_aliased})
            else()
                set(arpack_target_name ${tgt})
            endif()
            break()
        endif()
    endforeach()
    if(NOT arpack_target_name)
        message(FATAL_ERROR "Findarpack++: Could not find target for arpack dependency. Checked: ${arpack_target_candidates}")
    endif()

    list(APPEND lapack_target_candidates LAPACK::LAPACK lapack)
    foreach(tgt ${lapack_target_candidates})
        if(TARGET ${tgt})
            get_target_property(lapack_aliased ${tgt} ALIASED_TARGET)
            if(lapack_aliased)
                set(lapack_target_name ${lapack_aliased})
            else()
                set(lapack_target_name ${tgt})
            endif()
            break()
        endif()
    endforeach()
    if(NOT lapack_target_name)
        message(FATAL_ERROR "Findarpack++: Could not find target for LAPACK dependency. Checked: ${lapack_target_candidates}")
    endif()

    list(APPEND blas_target_candidates BLAS::BLAS blas)
    foreach(tgt ${blas_target_candidates})
        if(TARGET ${tgt})
            get_target_property(blas_aliased ${tgt} ALIASED_TARGET)
            if(blas_aliased)
                set(blas_target_name ${blas_aliased})
            else()
                set(blas_target_name ${tgt})
            endif()
            break()
        endif()
    endforeach()
    if(NOT blas_target_name)
        message(FATAL_ERROR "Findarpack++: Could not find target for BLAS dependency. Checked: ${blas_target_candidates}")
    endif()

    set(ARPACKPP_ARPACK_TARGET ${arpack_target_name} PARENT_SCOPE)
    set(ARPACKPP_LAPACK_TARGET ${lapack_target_name} PARENT_SCOPE)
    set(ARPACKPP_BLAS_TARGET ${blas_target_name} PARENT_SCOPE)

    if(ARPACKPP_NO_DEFAULT_PATH)
        set(NO_DEFAULT_PATH NO_DEFAULT_PATH)
    endif()
    if(ARPACKPP_NO_CMAKE_PACKAGE_REGISTRY)
        set(NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_PACKAGE_REGISTRY)
    endif()

    if (NOT TARGET arpack++::arpack++)
        include(GNUInstallDirs)
        unset(ARPACKPP_LIBRARY)
        unset(ARPACKPP_LIBRARY CACHE)
        unset(ARPACKPP_INCLUDE_DIR)
        unset(ARPACKPP_INCLUDE_DIR CACHE)
        find_library(ARPACKPP_LIBRARY
                NAMES arpackpp arpack++
                ${NO_DEFAULT_PATH}
                ${NO_CMAKE_PACKAGE_REGISTRY}
                )
        find_path(ARPACKPP_INCLUDE_DIR
                NAMES arpack++/arcomp.h arpackpp/arcomp.h
                PATH_SUFFIXES
                    include
                    arpack++ arpackpp
                    arpack++/include arpackpp/include
                    include/arpack++ include/arpackpp
                    arpack++/include/arpack++
                    arpackpp/include/arpackpp
                ${NO_DEFAULT_PATH}
                ${NO_CMAKE_PACKAGE_REGISTRY}
                )

        if(NOT ARPACKPP_LIBRARY)
            set(ARPACKPP_LIBRARY "")
        endif()
        if(NOT ARPACKPP_INCLUDE_DIR)
            set(ARPACKPP_INCLUDE_DIR "")
        endif()
    endif()
endfunction()


find_arpackpp()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(arpack++ DEFAULT_MSG ARPACKPP_INCLUDE_DIR)

if(arpack++_FOUND)
    if(NOT TARGET arpack++::arpack++)
        if(ARPACKPP_LIBRARY)
            add_library(arpack++::arpack++ UNKNOWN IMPORTED)
            set_target_properties(arpack++::arpack++ PROPERTIES IMPORTED_LOCATION "${ARPACKPP_LIBRARY}")
        else()
            add_library(arpack++::arpack++ INTERFACE IMPORTED)
        endif()
    endif()
    target_include_directories(arpack++::arpack++ SYSTEM INTERFACE ${ARPACKPP_INCLUDE_DIR})
    target_link_libraries(arpack++::arpack++ INTERFACE
                 ${ARPACKPP_ARPACK_TARGET}
                 ${ARPACKPP_LAPACK_TARGET}
                 ${ARPACKPP_BLAS_TARGET}
             )
endif()
