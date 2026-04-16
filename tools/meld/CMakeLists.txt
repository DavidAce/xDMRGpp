cmake_minimum_required(VERSION 3.18)
project(meld VERSION 1.0.0 DESCRIPTION "Melds xDMRG++ batch data in HDF5 format")

include(${CMAKE_SOURCE_DIR}/cmake/environment.cmake)

# Create an executable
add_executable(meld
               source/main.cpp
               source/meld-io/find.cpp
               source/meld-io/logger.cpp
               source/meld-io/hash.cpp
               source/meld-io/h5db.cpp
               source/meld-io/h5io.cpp
               source/meld-io/h5xf.cpp
               source/meld-io/id.cpp
               source/meld-io/h5dbg.cpp
               source/general/prof.cpp
               source/general/text.cpp
               source/general/human.cpp
               source/mpi/mpi-tools.cpp
               )
target_include_directories(meld PRIVATE source)

target_link_libraries(meld PRIVATE
                      xdmrg++-debug-info
                      xdmrg++-debug-stacktrace
                      xdmrg++-tensors-mpo-h5tb # To read the hamiltonian table
                      xdmrg++-qm-lbit-h5tb  # To read the circuit table
                      xdmrg++-tid)
target_link_libraries(meld PRIVATE xdmrg++-flags)
target_include_directories(meld PRIVATE ${CMAKE_SOURCE_DIR}/source)

find_package(MPI COMPONENTS CXX REQUIRED)
find_package(CLI11 REQUIRED)
find_package(h5pp REQUIRED)
find_package(Backward REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
target_link_libraries(meld PUBLIC CLI11::CLI11)
target_link_libraries(meld PUBLIC h5pp::h5pp)
target_link_libraries(meld PUBLIC Backward::Backward)

# Print summary of CMake configuration
include(${CMAKE_SOURCE_DIR}/cmake/PrintTargetInfo.cmake)
print_project_summary(meld)


