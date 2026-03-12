add_source_target(xdmrg++-math-svd
        TARGET_SOURCES
        svd.cpp
        INST_TEMPLATE_FILES
        inst/svd.inst.cpp.in
        inst/svd_dump.inst.cpp.in
        inst/svd_eigen.inst.cpp.in
        inst/svd_lapacke.inst.cpp.in
        inst/svd_rsvd.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        )
