add_subdirectory(h5tb)
set(xdmrg_tensors_mpo_inst_scalars ${DMRG_ENABLED_SCALARS} ${DMRG_INTERNAL_HIGHPREC_SCALARS})
list(REMOVE_DUPLICATES xdmrg_tensors_mpo_inst_scalars)
add_source_target(xdmrg++-tensors-mpo
                  INST_TEMPLATE_FILES
                  MpoFactory/MpoFactory.inst.cpp.in
                  MpoSite/MpoSite.inst.cpp.in
                  IsingSelfDual/IsingSelfDual.inst.cpp.in
                  IsingMajorana/IsingMajorana.inst.cpp.in
                  IsingRandomField/IsingRandomField.inst.cpp.in
                  LBit/LBit.inst.cpp.in
                  XXZ/XXZ.inst.cpp.in
                  INST_SCALARS
                  ${xdmrg_tensors_mpo_inst_scalars}
                  INST_PAIR_TEMPLATE_FILES
                  MpoSite/MpoSite.pair.inst.cpp.in
                  INST_PAIR_DIAGONAL
                  INST_PAIR_OUTER_SCALARS
                  ${xdmrg_tensors_mpo_inst_scalars}
                  INST_PAIR_INNER_SCALARS
                  ${xdmrg_tensors_mpo_inst_scalars}
                  OBJECT_LINK_LIBRARIES
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-tensors-mpo-h5tb
                  xdmrg++-config
                  xdmrg++-debug-info
                  xdmrg++-math-hash
                  xdmrg++-math-rnd
                  xdmrg++-math-tenx
                  xdmrg++-qm-spin
                  xdmrg++-tools-common-log
                  )
