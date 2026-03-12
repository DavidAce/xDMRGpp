add_source_target(xdmrg++-tensors-mps
                  INST_TEMPLATE_FILES
                  MpsSite/MpsSite.inst.cpp.in
                  OBJECT_LINK_LIBRARIES
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-tid
                  xdmrg++-debug
                  xdmrg++-math-tenx
                  xdmrg++-math-hash
                  xdmrg++-tools-common-contraction
                  xdmrg++-tools-common-log
                  )
