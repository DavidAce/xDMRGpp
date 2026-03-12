add_source_target(xdmrg++-math-eig-eigsmpo
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        solver_base/solver_base.inst.cpp.in
        solver_gdplusk/solver_gdplusk.inst.cpp.in
        solver_lanczos/solver_lanczos.inst.cpp.in
        solver_lobpcg/solver_lobpcg.inst.cpp.in

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig-matvec
        xdmrg++-math-eig-o
        )


