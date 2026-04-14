cmake_minimum_required(VERSION 3.15)
project(xdmrg++-test-svd)
enable_testing()
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
                      xdmrg++-config
                      xdmrg++-tid
                      xdmrg++-math-svd
                      xdmrg++-math-tenx
                      xdmrg++-tools-common-log
                      )
target_link_libraries(${PROJECT_NAME} PRIVATE xdmrg++-deps xdmrg++-flags)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_precompiled_headers(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE TEST_MATRIX_DIR="${PROJECT_SOURCE_DIR}/testmatrices")

if(DMRG_ENABLE_COVERAGE)
    target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
    target_link_options(${PROJECT_NAME} PRIVATE --coverage)
endif()

add_test(NAME ${PROJECT_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${PROJECT_NAME})
