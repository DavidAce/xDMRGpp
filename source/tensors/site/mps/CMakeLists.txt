add_source_target(xdmrg++-tensors-mps
                  INST_TEMPLATE_FILES
                  MpsSite/MpsSite.inst.cpp.in
                  INST_SCALARS
                  ${DMRG_ENABLED_SCALARS}
                  OBJECT_LINK_LIBRARIES
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-tid
                  xdmrg++-debug-info
                  xdmrg++-math-tenx
                  xdmrg++-math-hash
                  xdmrg++-tools-common-contraction
                  xdmrg++-tools-common-log
                  )
