function(_dmrg_resolve_instantiation_template out_template_path out_template_dir out_output_stem template_arg)
    if(IS_ABSOLUTE "${template_arg}")
        set(template_path "${template_arg}")
    else()
        set(template_path "${CMAKE_CURRENT_SOURCE_DIR}/${template_arg}")
    endif()

    get_filename_component(template_name "${template_path}" NAME)
    if(template_name MATCHES "^(.*)\\.pair\\.inst\\.cpp\\.in$")
        set(output_stem "${CMAKE_MATCH_1}")
    elseif(template_name MATCHES "^(.*)\\.inst\\.cpp\\.in$")
        set(output_stem "${CMAKE_MATCH_1}")
    elseif(template_name MATCHES "^(.*)\\.cpp\\.in$")
        set(output_stem "${CMAKE_MATCH_1}")
    else()
        string(REGEX REPLACE "\\.in$" "" output_stem "${template_name}")
    endif()

    get_filename_component(template_dir "${template_path}" DIRECTORY)

    set(${out_template_path} "${template_path}" PARENT_SCOPE)
    set(${out_template_dir} "${template_dir}" PARENT_SCOPE)
    set(${out_output_stem} "${output_stem}" PARENT_SCOPE)
endfunction()

function(_dmrg_get_default_instantiation_output_dir out_var template_dir)
    file(RELATIVE_PATH template_rel_dir "${CMAKE_CURRENT_SOURCE_DIR}" "${template_dir}")
    if(template_rel_dir STREQUAL "")
        set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    else()
        set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${template_rel_dir}")
    endif()
    set(${out_var} "${output_dir}" PARENT_SCOPE)
endfunction()

# generate_scalar_instantiations(<out_var>
#     TEMPLATE <path/to/file.inst.cpp.in>
#     SCALARS <fp32> <fp64> ...
#     OUTPUT_DIR <dir>)
#
# Expands one instantiation template into one generated translation unit per
# scalar type and returns the generated source list in <out_var>.
#
# TEMPLATE may be absolute or relative to CMAKE_CURRENT_SOURCE_DIR.
# SCALARS is optional; if omitted, the function falls back to the project-wide
# DMRG_ENABLED_SCALARS list assembled from the DMRG_ENABLE_<SCALAR> options in
# the top-level CMakeLists.txt.
# OUTPUT_DIR is optional; if omitted, generated files are placed under
# CMAKE_CURRENT_BINARY_DIR/generated while preserving the template's relative
# source-tree layout.
#
# The template receives @SCALAR@ and the numeric helper flag
# @SCALAR_IS_COMPLEX@ so one file can preserve complex-only explicit
# instantiations without duplicating CMake-side scalar lists.
#
# Each generated source receives the template directory as an include path so
# local includes such as #include "foo.impl.h" continue to work.
function(generate_scalar_instantiations out_var)
    set(options)
    set(oneValueArgs TEMPLATE OUTPUT_DIR)
    set(multiValueArgs SCALARS)
    cmake_parse_arguments(PARSE_ARGV 1 DMRG_INST "${options}" "${oneValueArgs}" "${multiValueArgs}")

    if(NOT DMRG_INST_TEMPLATE)
        message(FATAL_ERROR "generate_scalar_instantiations requires TEMPLATE")
    endif()

    _dmrg_resolve_instantiation_template(template_path template_dir output_stem "${DMRG_INST_TEMPLATE}")

    if(DMRG_INST_SCALARS)
        set(dmrg_inst_scalars ${DMRG_INST_SCALARS})
    else()
        set(dmrg_inst_scalars ${DMRG_ENABLED_SCALARS})
    endif()

    if(DMRG_INST_OUTPUT_DIR)
        set(output_dir "${DMRG_INST_OUTPUT_DIR}")
    else()
        _dmrg_get_default_instantiation_output_dir(output_dir "${template_dir}")
    endif()

    file(MAKE_DIRECTORY "${output_dir}")

    # Materialize one translation unit per scalar and let each generated file
    # resolve local includes relative to the original template directory.
    set(generated_sources)
    foreach(dmrg_scalar IN LISTS dmrg_inst_scalars)
        set(SCALAR "${dmrg_scalar}")
        if(dmrg_scalar MATCHES "^cx")
            set(SCALAR_IS_COMPLEX 1)
        else()
            set(SCALAR_IS_COMPLEX 0)
        endif()
        set(output_path "${output_dir}/${output_stem}.${dmrg_scalar}.cpp")
        configure_file("${template_path}" "${output_path}" @ONLY)
        set_property(SOURCE "${output_path}" APPEND PROPERTY INCLUDE_DIRECTORIES "${template_dir}")
        list(APPEND generated_sources "${output_path}")
    endforeach()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${template_path}")
    set(${out_var} "${generated_sources}" PARENT_SCOPE)
endfunction()

# generate_scalar_pair_instantiations(<out_var>
#     TEMPLATE <path/to/file.pair.inst.cpp.in>
#     OUTER_SCALARS <fp32> <fp64> ...
#     INNER_SCALARS <fp32> <fp64> ...
#     OUTPUT_DIR <dir>)
#
# Expands one instantiation template into one generated translation unit per
# scalar pair and returns the generated source list in <out_var>.
#
# TEMPLATE may be absolute or relative to CMAKE_CURRENT_SOURCE_DIR.
# OUTER_SCALARS and INNER_SCALARS are optional; when omitted they each fall
# back to the project-wide DMRG_ENABLED_SCALARS list from the top-level
# DMRG_ENABLE_<SCALAR> options.
# OUTPUT_DIR is optional; if omitted, generated files are placed under
# CMAKE_CURRENT_BINARY_DIR/generated while preserving the template's relative
# source-tree layout.
#
# The template receives @OUTER_SCALAR@ and @INNER_SCALAR@, as well as the more
# descriptive aliases @STORAGE_SCALAR@ and @CALC_SCALAR@. It also receives the
# numeric helper flags @OUTER_SCALAR_EQUALS_INNER@ and @INNER_SCALAR_IS_FP32@
# so templates can preserve special-case instantiation rules without duplicating
# CMake-side pair lists.
#
# Each generated source receives the template directory as an include path so
# local includes such as #include "../foo.impl.h" continue to work.
function(generate_scalar_pair_instantiations out_var)
    set(options)
    set(oneValueArgs TEMPLATE OUTPUT_DIR)
    set(multiValueArgs OUTER_SCALARS INNER_SCALARS)
    cmake_parse_arguments(PARSE_ARGV 1 DMRG_PAIR "${options}" "${oneValueArgs}" "${multiValueArgs}")

    if(NOT DMRG_PAIR_TEMPLATE)
        message(FATAL_ERROR "generate_scalar_pair_instantiations requires TEMPLATE")
    endif()

    _dmrg_resolve_instantiation_template(template_path template_dir output_stem "${DMRG_PAIR_TEMPLATE}")

    if(DMRG_PAIR_OUTER_SCALARS)
        set(dmrg_outer_scalars ${DMRG_PAIR_OUTER_SCALARS})
    else()
        set(dmrg_outer_scalars ${DMRG_ENABLED_SCALARS})
    endif()

    if(DMRG_PAIR_INNER_SCALARS)
        set(dmrg_inner_scalars ${DMRG_PAIR_INNER_SCALARS})
    else()
        set(dmrg_inner_scalars ${DMRG_ENABLED_SCALARS})
    endif()

    if(DMRG_PAIR_OUTPUT_DIR)
        set(output_dir "${DMRG_PAIR_OUTPUT_DIR}")
    else()
        _dmrg_get_default_instantiation_output_dir(output_dir "${template_dir}")
    endif()

    file(MAKE_DIRECTORY "${output_dir}")

    # Materialize one translation unit per scalar pair. The first scalar is
    # encoded in the output subdirectory so pair expansions mirror the current
    # handwritten layout of <outer>/<inner>.cpp.
    set(generated_sources)
    foreach(dmrg_outer_scalar IN LISTS dmrg_outer_scalars)
        set(pair_output_dir "${output_dir}/${dmrg_outer_scalar}")
        file(MAKE_DIRECTORY "${pair_output_dir}")

        foreach(dmrg_inner_scalar IN LISTS dmrg_inner_scalars)
            set(OUTER_SCALAR "${dmrg_outer_scalar}")
            set(INNER_SCALAR "${dmrg_inner_scalar}")
            set(STORAGE_SCALAR "${dmrg_outer_scalar}")
            set(CALC_SCALAR "${dmrg_inner_scalar}")
            if(dmrg_outer_scalar STREQUAL dmrg_inner_scalar)
                set(OUTER_SCALAR_EQUALS_INNER 1)
            else()
                set(OUTER_SCALAR_EQUALS_INNER 0)
            endif()
            if(dmrg_inner_scalar STREQUAL "fp32")
                set(INNER_SCALAR_IS_FP32 1)
            else()
                set(INNER_SCALAR_IS_FP32 0)
            endif()
            set(output_path "${pair_output_dir}/${dmrg_inner_scalar}.cpp")
            configure_file("${template_path}" "${output_path}" @ONLY)
            set_property(SOURCE "${output_path}" APPEND PROPERTY INCLUDE_DIRECTORIES "${template_dir}")
            list(APPEND generated_sources "${output_path}")
        endforeach()
    endforeach()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${template_path}")
    set(${out_var} "${generated_sources}" PARENT_SCOPE)
endfunction()
