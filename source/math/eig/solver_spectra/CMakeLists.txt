add_source_target(xdmrg++-math-eig-spectra
        TARGET_SOURCES
        solver_spectra.cx32.cpp
        solver_spectra.cx64.cpp
        solver_spectra.cx128.cpp
        solver_spectra.fp32.cpp
        solver_spectra.fp64.cpp
        solver_spectra.fp128.cpp
        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig-matvec
        xdmrg++-math-eig-o
        )




