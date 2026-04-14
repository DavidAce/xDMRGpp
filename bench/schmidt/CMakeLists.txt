cmake_minimum_required(VERSION 3.15)
project(xdmrg++-bench-schmidt)
enable_testing()
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
                      xdmrg++-config
                      xdmrg++-tid
                      xdmrg++-math-svd
                      xdmrg++-math-tenx
                      xdmrg++-math-hash
                      xdmrg++-tensors-mps
                      xdmrg++-tools-common-contraction
                      xdmrg++-tools-common-log
                      xdmrg++-tools-common-split)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(${PROJECT_NAME} PRIVATE xdmrg++-deps xdmrg++-flags)
target_link_precompiled_headers(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE BENCH_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
