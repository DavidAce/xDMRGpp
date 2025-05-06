add_subdirectory(site)
add_subdirectory(model)
add_subdirectory(edges)
add_subdirectory(state)

add_source_target(xdmrg++-tensors-finite
                  TARGET_SOURCES
                  TensorsFinite/cx32.cpp
                  TensorsFinite/cx64.cpp
                  TensorsFinite/cx128.cpp
                  TensorsFinite/fp32.cpp
                  TensorsFinite/fp64.cpp
                  TensorsFinite/fp128.cpp


                  OBJECT_LINK_LIBRARIES
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-tid
                  xdmrg++-config
                  xdmrg++-math-tenx
                  xdmrg++-tensors-mps
                  xdmrg++-tensors-mpo
                  xdmrg++-tensors-env
                  xdmrg++-tensors-state-finite
                  xdmrg++-tensors-model-finite
                  xdmrg++-tensors-edges-finite
                  xdmrg++-tools-finite-env
                  xdmrg++-tools-finite-measure
                  xdmrg++-tools-finite-mps
                  xdmrg++-tools-finite-mpo
                  xdmrg++-tools-finite-multisite
                  xdmrg++-tools-finite-ops
                  xdmrg++-tools-common-log
                  )
add_source_target(xdmrg++-tensors-infinite
        TARGET_SOURCES
        TensorsInfinite/cx32.cpp
        TensorsInfinite/cx64.cpp
        TensorsInfinite/cx128.cpp
        TensorsInfinite/fp32.cpp
        TensorsInfinite/fp64.cpp
        TensorsInfinite/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tensors-mps
        xdmrg++-tensors-mpo
        xdmrg++-tensors-env
        xdmrg++-tensors-state-infinite
        xdmrg++-tensors-model-infinite
        xdmrg++-tensors-edges-infinite
        xdmrg++-tools-infinite-env
        xdmrg++-tools-infinite-measure
        xdmrg++-tools-infinite-mps
        xdmrg++-tools-common-log
        )

add_library(xdmrg++-tensors INTERFACE)
target_link_libraries(xdmrg++-tensors INTERFACE xdmrg++-tensors-finite xdmrg++-tensors-infinite)
