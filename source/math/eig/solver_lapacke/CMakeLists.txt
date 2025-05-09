add_source_target(xdmrg++-math-eig-lapacke
        TARGET_SOURCES
        sgeev.cpp   # Standard problems
        ssyevd.cpp  # Standard problems
        ssyevx.cpp  # Standard problems
        ssyevr.cpp  # Standard problems
        ssygvd.cpp  # Generalized problems
        ssygvx.cpp  # Generalized problems

        dgeev.cpp   # Standard problems
        dsyevd.cpp  # Standard problems
        dsyevx.cpp  # Standard problems
        dsyevr.cpp  # Standard problems
        dsygvd.cpp  # Generalized problems
        dsygvx.cpp  # Generalized problems

        qgeev_eigen.cpp
        qsyevd_eigen.cpp
        qsygvd_eigen.cpp


        cgeev.cpp
        cheevd.cpp
        cheevr.cpp

        zgeev.cpp
        zheevd.cpp
        zheevr.cpp

        wgeev_eigen.cpp
        wheevd_eigen.cpp
        whegvd_eigen.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-math-eig-o
        )