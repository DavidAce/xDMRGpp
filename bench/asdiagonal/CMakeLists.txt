cmake_minimum_required(VERSION 3.15)
project(xdmrg++-bench-asdiagonal)
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_link_libraries(${PROJECT_NAME} PRIVATE xdmrg++-deps xdmrg++-flags)
target_link_precompiled_headers(${PROJECT_NAME})

