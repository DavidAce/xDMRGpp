cmake_minimum_required(VERSION 3.15)
project(xdmrg++-test-tomlpp-parser)
enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(tomlplusplus 3.4.0 REQUIRED)

set(TOMLPP_PARSER_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(TOMLPP_PARSER_SETTINGS_H ${TOMLPP_PARSER_GENERATED_DIR}/settings_generated.h)
set(TOMLPP_PARSER_SPECS_H ${TOMLPP_PARSER_GENERATED_DIR}/setting_specs_generated.h)
set(TOMLPP_PARSER_ENUM_CHOICES_H ${TOMLPP_PARSER_GENERATED_DIR}/enum_choices_generated.h)
set(TOMLPP_PARSER_PARSE_CPP ${TOMLPP_PARSER_GENERATED_DIR}/parse_generated.cpp)
set(TOMLPP_PARSER_INPUT_TOML ${TOMLPP_PARSER_GENERATED_DIR}/input.test.toml)

add_custom_command(
    OUTPUT
        ${TOMLPP_PARSER_SETTINGS_H}
        ${TOMLPP_PARSER_SPECS_H}
        ${TOMLPP_PARSER_ENUM_CHOICES_H}
        ${TOMLPP_PARSER_PARSE_CPP}
        ${TOMLPP_PARSER_INPUT_TOML}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TOMLPP_PARSER_GENERATED_DIR}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_settings.py
            ${CMAKE_CURRENT_SOURCE_DIR}/settings.def
            ${CMAKE_SOURCE_DIR}/source/config/enums.h
            ${TOMLPP_PARSER_SETTINGS_H}
            ${TOMLPP_PARSER_SPECS_H}
            ${TOMLPP_PARSER_ENUM_CHOICES_H}
            ${TOMLPP_PARSER_PARSE_CPP}
            ${TOMLPP_PARSER_INPUT_TOML}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_settings.py
        ${CMAKE_CURRENT_SOURCE_DIR}/settings.def
        ${CMAKE_SOURCE_DIR}/source/config/enums.h
    VERBATIM)

add_custom_target(${PROJECT_NAME}-generate
                  DEPENDS
                      ${TOMLPP_PARSER_SETTINGS_H}
                      ${TOMLPP_PARSER_SPECS_H}
                      ${TOMLPP_PARSER_ENUM_CHOICES_H}
                      ${TOMLPP_PARSER_PARSE_CPP}
                      ${TOMLPP_PARSER_INPUT_TOML})

add_executable(${PROJECT_NAME}
               main.cpp
               config_io.cpp
               simulation_preview.cpp
               ${TOMLPP_PARSER_PARSE_CPP})

target_link_libraries(${PROJECT_NAME} PRIVATE tomlplusplus::tomlplusplus xdmrg++-deps xdmrg++-flags)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${TOMLPP_PARSER_GENERATED_DIR})
target_link_precompiled_headers(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE TEST_TOML_FILE="${TOMLPP_PARSER_INPUT_TOML}")
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}-generate)

if(DMRG_ENABLE_COVERAGE)
    target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
    target_link_options(${PROJECT_NAME} PRIVATE --coverage)
endif()

add_test(NAME ${PROJECT_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${PROJECT_NAME})
