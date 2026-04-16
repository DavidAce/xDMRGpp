add_source_target(xdmrg++-tensors-mpo-h5tb
                  TARGET_SOURCES
                  h5tb.cpp
                  OBJECT_LINK_LIBRARIES
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-debug-info
                  xdmrg++-tools-common-log
                  )
target_include_directories(xdmrg++-tensors-mpo-h5tb SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}) # Needed by meld
