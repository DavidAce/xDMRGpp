add_source_target(xdmrg++-math-eig-lapacke
        TARGET_SOURCES
        solver_sgeev.cpp   # Standard problems
        solver_ssyevd.cpp  # Standard problems
        solver_ssyevx.cpp  # Standard problems
        solver_ssyevr.cpp  # Standard problems
        solver_ssygvd.cpp  # Generalized problems
        solver_ssygvx.cpp  # Generalized problems

        solver_dgeev.cpp   # Standard problems
        solver_dsyevd.cpp  # Standard problems
        solver_dsyevx.cpp  # Standard problems
        solver_dsyevr.cpp  # Standard problems
        solver_dsygvd.cpp  # Generalized problems
        solver_dsygvx.cpp  # Generalized problems

        solver_cgeev.cpp
        solver_cheevd.cpp
        solver_cheevr.cpp

        solver_zgeev.cpp
        solver_zheevd.cpp
        solver_zheevr.cpp


        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-math-eig-o
        )