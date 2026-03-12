include(../cmake/AddSourceTarget.cmake)
add_subdirectory(algorithms)
add_subdirectory(config)
add_subdirectory(debug)
add_subdirectory(io)
add_subdirectory(math)
add_subdirectory(qm)
add_subdirectory(tensors)
add_subdirectory(tid)
add_subdirectory(tools)
add_library(xdmrg++-main OBJECT)
target_sources(xdmrg++-main PRIVATE main.cpp)
target_link_libraries(xdmrg++-main PUBLIC xdmrg++-deps xdmrg++-flags)
target_link_libraries(xdmrg++-main INTERFACE
        xdmrg++-algorithms
        xdmrg++-config
        xdmrg++-debug
        xdmrg++-stacktrace
        xdmrg++-io
        xdmrg++-math
        xdmrg++-qm
        xdmrg++-tensors
        xdmrg++-tid
        xdmrg++-tools
        )

target_link_precompiled_headers(xdmrg++-main)
