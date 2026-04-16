cmake_minimum_required(VERSION 3.15)
project(xdmrg++-bench-swapgates)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
                      xdmrg++-tid
                      xdmrg++-config
                      xdmrg++-debug-info
                      xdmrg++-qm-gate
                      xdmrg++-math-tenx
                      xdmrg++-tensors-state-finite
                      xdmrg++-tools-finite-mps
                      xdmrg++-tools-finite-h5
                      xdmrg++-tools-common-h5
                      xdmrg++-tools-common-prof)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(${PROJECT_NAME} PRIVATE xdmrg++-deps xdmrg++-flags)
target_link_precompiled_headers(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE BENCH_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
