add_source_target(xdmrg++-debug-info
        TARGET_SOURCES
#        exceptions.cxx
        info.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        )

add_source_target(xdmrg++-debug-affinity
        TARGET_SOURCES
        affinity.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        )

find_library(DW_LIBRARY NAMES dw)
if(DW_LIBRARY)
    set(BACKWARD_DW_LIBRARY ${DW_LIBRARY})
    set(BACKWARD_REDEFINE_DW BACKWARD_REDEFINE_DW=1)
endif()

add_source_target(xdmrg++-debug-stacktrace
        TARGET_SOURCES
        stacktrace.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        PRIVATE_LINK_LIBRARIES
        Backward::Backward ${BACKWARD_DW_LIBRARY}
        COMPILE_DEFINITIONS
        ${BACKWARD_REDEFINE_DW}
        )
