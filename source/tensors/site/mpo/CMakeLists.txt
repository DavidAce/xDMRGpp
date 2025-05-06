add_subdirectory(h5tb)
add_source_target(xdmrg++-tensors-mpo
                  TARGET_SOURCES

                  MpoFactory/fp32.cpp
                  MpoFactory/fp64.cpp
                  MpoFactory/fp128.cpp
                  MpoFactory/cx32.cpp
                  MpoFactory/cx64.cpp
                  MpoFactory/cx128.cpp

                  MpoSite/fp32.cpp
                  MpoSite/fp64.cpp
                  MpoSite/fp128.cpp
                  MpoSite/cx32.cpp
                  MpoSite/cx64.cpp
                  MpoSite/cx128.cpp

                  MpoSite/fp32/fp32.cpp
                  MpoSite/fp32/fp64.cpp
                  MpoSite/fp32/fp128.cpp
                  MpoSite/fp32/cx32.cpp
                  MpoSite/fp32/cx64.cpp
                  MpoSite/fp32/cx128.cpp
                  MpoSite/fp64/fp32.cpp
                  MpoSite/fp64/fp64.cpp
                  MpoSite/fp64/fp128.cpp
                  MpoSite/fp64/cx32.cpp
                  MpoSite/fp64/cx64.cpp
                  MpoSite/fp64/cx128.cpp
                  MpoSite/fp128/fp32.cpp
                  MpoSite/fp128/fp64.cpp
                  MpoSite/fp128/fp128.cpp
                  MpoSite/fp128/cx32.cpp
                  MpoSite/fp128/cx64.cpp
                  MpoSite/fp128/cx128.cpp
                  MpoSite/cx32/fp32.cpp
                  MpoSite/cx32/fp64.cpp
                  MpoSite/cx32/fp128.cpp
                  MpoSite/cx32/cx32.cpp
                  MpoSite/cx32/cx64.cpp
                  MpoSite/cx32/cx128.cpp
                  MpoSite/cx64/fp32.cpp
                  MpoSite/cx64/fp64.cpp
                  MpoSite/cx64/fp128.cpp
                  MpoSite/cx64/cx32.cpp
                  MpoSite/cx64/cx64.cpp
                  MpoSite/cx64/cx128.cpp
                  MpoSite/cx128/fp32.cpp
                  MpoSite/cx128/fp64.cpp
                  MpoSite/cx128/fp128.cpp
                  MpoSite/cx128/cx32.cpp
                  MpoSite/cx128/cx64.cpp
                  MpoSite/cx128/cx128.cpp


                  IsingSelfDual/fp32.cpp
                  IsingSelfDual/fp64.cpp
                  IsingSelfDual/fp128.cpp
                  IsingSelfDual/cx32.cpp
                  IsingSelfDual/cx64.cpp
                  IsingSelfDual/cx128.cpp

                  IsingMajorana/fp32.cpp
                  IsingMajorana/fp64.cpp
                  IsingMajorana/fp128.cpp
                  IsingMajorana/cx32.cpp
                  IsingMajorana/cx64.cpp
                  IsingMajorana/cx128.cpp

                  IsingRandomField/fp32.cpp
                  IsingRandomField/fp64.cpp
                  IsingRandomField/fp128.cpp
                  IsingRandomField/cx32.cpp
                  IsingRandomField/cx64.cpp
                  IsingRandomField/cx128.cpp

                  LBit/fp32.cpp
                  LBit/fp64.cpp
                  LBit/fp128.cpp
                  LBit/cx32.cpp
                  LBit/cx64.cpp
                  LBit/cx128.cpp

                  XXZ/fp32.cpp
                  XXZ/fp64.cpp
                  XXZ/fp128.cpp
                  XXZ/cx32.cpp
                  XXZ/cx64.cpp
                  XXZ/cx128.cpp

                  OBJECT_LINK_LIBRARIES
                  xdmrg++-deps xdmrg++-flags
                  INTERFACE_LINK_LIBRARIES
                  xdmrg++-tensors-mpo-h5tb
                  xdmrg++-config
                  xdmrg++-debug
                  xdmrg++-math-hash
                  xdmrg++-math-rnd
                  xdmrg++-math-tenx
                  xdmrg++-qm-spin
                  xdmrg++-tools-common-log
                  )