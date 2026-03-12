add_source_target(xdmrg++-math-eig-spectra
        TARGET_SOURCES
        INST_TEMPLATE_FILES
        solver_spectra.inst.cpp.in
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig-matvec
        xdmrg++-math-eig-o
        )



