add_source_target(xdmrg++-tensors-env
                  TARGET_SOURCES
                  EnvPair.cpp
                  INST_TEMPLATE_FILES
                  EnvFactory/EnvFactory.inst.cpp.in
                  EnvBase/EnvBase.inst.cpp.in
                  EnvEne/EnvEne.inst.cpp.in
                  EnvVar/EnvVar.inst.cpp.in
                  OBJECT_LINK_LIBRARIES
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-math-hash
                  xdmrg++-tensors-mps
                  xdmrg++-tensors-mpo
                  xdmrg++-tensors-env
                  xdmrg++-tools-common-log
                  )
