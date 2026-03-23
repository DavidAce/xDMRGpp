find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB XDMRG_CONFIG_ENUM_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/enums/*.h")

set(XDMRG_CONFIG_GENERATED_DIR ${CMAKE_BINARY_DIR}/source/config/generated)
set(XDMRG_CONFIG_GENERATED_CPPS)

foreach(enum_header IN LISTS XDMRG_CONFIG_ENUM_HEADERS)
    get_filename_component(enum_name ${enum_header} NAME_WE)
    list(APPEND XDMRG_CONFIG_GENERATED_CPPS ${XDMRG_CONFIG_GENERATED_DIR}/${enum_name}_enum.cpp)
endforeach()

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${XDMRG_CONFIG_GENERATED_DIR})
execute_process(COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/generate_enum_io.py
                        ${XDMRG_CONFIG_GENERATED_DIR}
                        ${XDMRG_CONFIG_ENUM_HEADERS}
                COMMAND_ERROR_IS_FATAL ANY)

add_custom_command(
    OUTPUT
        ${XDMRG_CONFIG_GENERATED_CPPS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${XDMRG_CONFIG_GENERATED_DIR}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_enum_io.py
            ${XDMRG_CONFIG_GENERATED_DIR}
            ${XDMRG_CONFIG_ENUM_HEADERS}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_enum_io.py
        ${CMAKE_CURRENT_SOURCE_DIR}/enum_utils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/enums.h
        ${XDMRG_CONFIG_ENUM_HEADERS}
    VERBATIM)

add_custom_target(xdmrg++-config-enums-generate
                  DEPENDS
                      ${XDMRG_CONFIG_GENERATED_CPPS})

add_source_target(xdmrg++-config
                  TARGET_SOURCES
                  loader.cpp
                  settings.cpp
                  parse.cpp
                  threading.cpp
                  ${XDMRG_CONFIG_GENERATED_CPPS}
                  OBJECT_LINK_LIBRARIES
                  xdmrg++-math-tenx
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-tools-common-log
                  )

add_dependencies(xdmrg++-config-o xdmrg++-config-enums-generate)
add_dependencies(xdmrg++-config xdmrg++-config-enums-generate)
target_include_directories(xdmrg++-config-o PRIVATE ${CMAKE_BINARY_DIR}/source)
target_include_directories(xdmrg++-config INTERFACE ${CMAKE_BINARY_DIR}/source)
target_include_directories(xdmrg++-flags INTERFACE ${CMAKE_BINARY_DIR}/source)
