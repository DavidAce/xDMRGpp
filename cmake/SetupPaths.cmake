cmake_minimum_required(VERSION 3.15)

# Setup build and install directories for dependencies
if(NOT DEFINED DMRG_DEPS_BUILD_DIR)
    set(DMRG_DEPS_BUILD_DIR ${CMAKE_BINARY_DIR}/pkg-build)
endif()

# Install dependencies to the same location as the main project by default
if(NOT DEFINED DMRG_DEPS_INSTALL_DIR)
    set(DMRG_DEPS_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
endif()

set(PKG_INSTALL_DIR_DEFAULT ${DMRG_DEPS_INSTALL_DIR} CACHE STRING "" FORCE )
set(PKG_BUILD_DIR_DEFAULT   ${DMRG_DEPS_BUILD_DIR}   CACHE STRING "" FORCE )

list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{CMAKE_PREFIX_PATH};${DMRG_DEPS_INSTALL_DIR}")
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} CACHE INTERNAL "Paths for find_package config lookup" FORCE)


