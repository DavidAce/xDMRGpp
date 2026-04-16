cmake_minimum_required(VERSION 3.15)
project(xdmrg++-bench-matvec-dispatch)
add_executable(${PROJECT_NAME}
               main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(${PROJECT_NAME} PRIVATE xdmrg++-deps xdmrg++-flags)
target_link_libraries(${PROJECT_NAME} PRIVATE
                      xdmrg++-debug-affinity
                      xdmrg++-config-runtime
                      xdmrg++-tools-common-matrix-vector-product
                      xdmrg++-math-tenx
                      xdmrg++-tid
                      xdmrg++-tools-common-log)
if(DMRG_ENABLE_TBLIS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DMRG_ENABLE_TBLIS)
endif()
if(DMRG_ENABLE_CUTENSOR)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DMRG_ENABLE_CUTENSOR)
endif()
target_link_precompiled_headers(${PROJECT_NAME})
