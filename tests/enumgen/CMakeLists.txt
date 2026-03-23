cmake_minimum_required(VERSION 3.15)
project(xdmrg++-test-enumgen)
enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ENUMGEN_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(ENUMGEN_GENERATED_H ${ENUMGEN_GENERATED_DIR}/demo_enums_generated.h)

add_custom_command(
    OUTPUT ${ENUMGEN_GENERATED_H}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ENUMGEN_GENERATED_DIR}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_enums.py
            ${CMAKE_CURRENT_SOURCE_DIR}/enums.def
            ${ENUMGEN_GENERATED_H}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_enums.py
        ${CMAKE_CURRENT_SOURCE_DIR}/enums.def
        ${CMAKE_SOURCE_DIR}/tests/enum_support.h
    VERBATIM)

add_custom_target(${PROJECT_NAME}-generate DEPENDS ${ENUMGEN_GENERATED_H})

add_executable(${PROJECT_NAME} main.cpp ${ENUMGEN_GENERATED_H})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_include_directories(${PROJECT_NAME} PRIVATE ${ENUMGEN_GENERATED_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE xdmrg++-deps xdmrg++-flags)
target_link_precompiled_headers(${PROJECT_NAME})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}-generate)

if(DMRG_ENABLE_COVERAGE)
    target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
    target_link_options(${PROJECT_NAME} PRIVATE --coverage)
endif()

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
