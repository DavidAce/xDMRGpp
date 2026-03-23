cmake_minimum_required(VERSION 3.15)
project(xdmrg++-test-enumcppgen)
enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ENUMCPPGEN_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(ENUMCPPGEN_GENERATED_DECLS ${ENUMCPPGEN_GENERATED_DIR}/enum_generated_decls.h)
set(ENUMCPPGEN_ENUM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/enums/AlgorithmType.h
    ${CMAKE_CURRENT_SOURCE_DIR}/enums/MpoCompress.h
    ${CMAKE_CURRENT_SOURCE_DIR}/enums/ProjectionPolicy.h)
set(ENUMCPPGEN_GENERATED_CPPS
    ${ENUMCPPGEN_GENERATED_DIR}/AlgorithmType_enum.cpp
    ${ENUMCPPGEN_GENERATED_DIR}/MpoCompress_enum.cpp
    ${ENUMCPPGEN_GENERATED_DIR}/ProjectionPolicy_enum.cpp)

add_custom_command(
    OUTPUT
        ${ENUMCPPGEN_GENERATED_DECLS}
        ${ENUMCPPGEN_GENERATED_CPPS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ENUMCPPGEN_GENERATED_DIR}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_enum_io.py
            ${ENUMCPPGEN_GENERATED_DIR}
            ${ENUMCPPGEN_ENUM_HEADERS}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_enum_io.py
        ${CMAKE_CURRENT_SOURCE_DIR}/enum_api.h
        ${CMAKE_CURRENT_SOURCE_DIR}/enums.h
        ${ENUMCPPGEN_ENUM_HEADERS}
    VERBATIM)

add_custom_target(${PROJECT_NAME}-generate
                  DEPENDS
                      ${ENUMCPPGEN_GENERATED_DECLS}
                      ${ENUMCPPGEN_GENERATED_CPPS})

add_executable(${PROJECT_NAME}
               main.cpp
               ${ENUMCPPGEN_GENERATED_CPPS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${ENUMCPPGEN_GENERATED_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE xdmrg++-deps xdmrg++-flags)
target_link_precompiled_headers(${PROJECT_NAME})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}-generate)

if(DMRG_ENABLE_COVERAGE)
    target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
    target_link_options(${PROJECT_NAME} PRIVATE --coverage)
endif()

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
