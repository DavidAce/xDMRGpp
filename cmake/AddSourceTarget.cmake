cmake_minimum_required(VERSION 3.15)
include(${CMAKE_CURRENT_LIST_DIR}/GenerateInstantiationSources.cmake)

# add_source_target(<target_name>
#     TARGET_SOURCES <src1> <src2> ...
#     INST_TEMPLATE_FILES <tmpl1.inst.cpp.in> <tmpl2.inst.cpp.in> ...
#     INST_SCALARS <fp32> <fp64> ...
#     INST_OUTPUT_DIR <dir>
#     INST_PAIR_TEMPLATE_FILES <tmpl1.pair.inst.cpp.in> <tmpl2.pair.inst.cpp.in> ...
#     INST_PAIR_OUTER_SCALARS <fp32> <fp64> ...
#     INST_PAIR_INNER_SCALARS <fp32> <fp64> ...
#     INST_PAIR_OUTPUT_DIR <dir>
#     OBJECT_LINK_LIBRARIES <libs...>
#     PRIVATE_LINK_LIBRARIES <libs...>
#     INTERFACE_LINK_LIBRARIES <libs...>
#     COMPILE_OPTIONS <opts...>
#     COMPILE_DEFINITIONS <defs...>)
#
# Creates an object library <target_name>-o and an interface library <target_name>
# that forwards the object files and interface link dependencies.
#
# Use TARGET_SOURCES for ordinary translation units.
# Use INST_TEMPLATE_FILES for explicit-instantiation templates that should be
# expanded into one generated source per scalar type.
# INST_SCALARS is optional; when omitted, each template is expanded using the
# project-wide DMRG_ENABLED_SCALARS list defined by the top-level
# DMRG_ENABLE_<SCALAR> options.
# INST_OUTPUT_DIR is optional; when omitted, generated sources are written under
# CMAKE_CURRENT_BINARY_DIR/generated, mirroring the template's relative path.
# Use INST_PAIR_TEMPLATE_FILES for explicit-instantiation templates that should
# be expanded into one generated source per scalar pair.
# INST_PAIR_OUTER_SCALARS and INST_PAIR_INNER_SCALARS are optional; when
# omitted, each side falls back to the project-wide DMRG_ENABLED_SCALARS list.
# INST_PAIR_OUTPUT_DIR is optional; when omitted, generated pair sources are
# written under CMAKE_CURRENT_BINARY_DIR/generated, mirroring the template's
# relative path.
function(add_source_target target_name)
    set(options CONFIG MODULE CHECK QUIET DEBUG INSTALL_PREFIX_PKGNAME)
    set(oneValueArgs VERSION INSTALL_DIR INSTALL_SUBDIR BUILD_DIR BUILD_SUBDIR FIND_NAME TARGET_NAME LINK_TYPE INST_OUTPUT_DIR INST_PAIR_OUTPUT_DIR)
    set(multiValueArgs TARGET_SOURCES INST_TEMPLATE_FILES INST_SCALARS INST_PAIR_TEMPLATE_FILES INST_PAIR_OUTER_SCALARS INST_PAIR_INNER_SCALARS OBJECT_LINK_LIBRARIES PRIVATE_LINK_LIBRARIES INTERFACE_LINK_LIBRARIES COMPILE_OPTIONS COMPILE_DEFINITIONS)
    cmake_parse_arguments(PARSE_ARGV 1 ADD "${options}" "${oneValueArgs}" "${multiValueArgs}")

    # Start from the hand-written source list and append any generated
    # explicit-instantiation translation units requested for this target.
    set(resolved_target_sources ${ADD_TARGET_SOURCES})
    foreach(inst_template IN LISTS ADD_INST_TEMPLATE_FILES)
        set(dmrg_inst_args TEMPLATE "${inst_template}")
        if(ADD_INST_SCALARS)
            list(APPEND dmrg_inst_args SCALARS ${ADD_INST_SCALARS})
        endif()
        if(ADD_INST_OUTPUT_DIR)
            list(APPEND dmrg_inst_args OUTPUT_DIR "${ADD_INST_OUTPUT_DIR}")
        endif()
        generate_scalar_instantiations(generated_sources ${dmrg_inst_args})
        list(APPEND resolved_target_sources ${generated_sources})
    endforeach()

    # Append any generated scalar-pair translation units requested for this
    # target. This covers storage/calc mixtures such as EdgesFinite<fp32>
    # exporting conversions to fp64 or cx64.
    foreach(inst_pair_template IN LISTS ADD_INST_PAIR_TEMPLATE_FILES)
        set(dmrg_pair_args TEMPLATE "${inst_pair_template}")
        if(ADD_INST_PAIR_OUTER_SCALARS)
            list(APPEND dmrg_pair_args OUTER_SCALARS ${ADD_INST_PAIR_OUTER_SCALARS})
        endif()
        if(ADD_INST_PAIR_INNER_SCALARS)
            list(APPEND dmrg_pair_args INNER_SCALARS ${ADD_INST_PAIR_INNER_SCALARS})
        endif()
        if(ADD_INST_PAIR_OUTPUT_DIR)
            list(APPEND dmrg_pair_args OUTPUT_DIR "${ADD_INST_PAIR_OUTPUT_DIR}")
        endif()
        generate_scalar_pair_instantiations(generated_pair_sources ${dmrg_pair_args})
        list(APPEND resolved_target_sources ${generated_pair_sources})
    endforeach()

    # Build the object library that owns the actual compilation units.
    set(object_name ${target_name}-o)
    add_library(${object_name} OBJECT)
    target_sources(${object_name} PRIVATE ${resolved_target_sources})
    target_link_libraries(${object_name} PUBLIC ${ADD_OBJECT_LINK_LIBRARIES})
    target_link_libraries(${object_name} PRIVATE ${ADD_PRIVATE_LINK_LIBRARIES})
    target_compile_definitions(${object_name} PRIVATE ${ADD_COMPILE_DEFINITIONS})
    target_compile_options(${object_name} PRIVATE ${ADD_COMPILE_OPTIONS})
    target_link_precompiled_headers(${object_name})

    # Expose the object files through an interface target so callers can link
    # against a normal logical target name.
    add_library(${target_name} INTERFACE)
    target_sources(${target_name} INTERFACE $<TARGET_OBJECTS:${object_name}>)
    target_link_libraries(${target_name} INTERFACE ${object_name} ${ADD_INTERFACE_LINK_LIBRARIES})
endfunction()
