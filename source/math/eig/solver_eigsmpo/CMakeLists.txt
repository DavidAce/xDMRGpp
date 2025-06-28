add_source_target(xdmrg++-math-eig-eigsmpo
        TARGET_SOURCES

        solver_base/cx32.cpp
        solver_base/cx64.cpp
        solver_base/cx128.cpp
        solver_base/fp32.cpp
        solver_base/fp64.cpp
        solver_base/fp128.cpp

        solver_gdplusk/cx32.cpp
        solver_gdplusk/cx64.cpp
        solver_gdplusk/cx128.cpp
        solver_gdplusk/fp32.cpp
        solver_gdplusk/fp64.cpp
        solver_gdplusk/fp128.cpp

        solver_lanczos/cx32.cpp
        solver_lanczos/cx64.cpp
        solver_lanczos/cx128.cpp
        solver_lanczos/fp32.cpp
        solver_lanczos/fp64.cpp
        solver_lanczos/fp128.cpp

        solver_lobpcg/cx32.cpp
        solver_lobpcg/cx64.cpp
        solver_lobpcg/cx128.cpp
        solver_lobpcg/fp32.cpp
        solver_lobpcg/fp64.cpp
        solver_lobpcg/fp128.cpp

        OBJECT_LINK_LIBRARIES
        xdmrg++-deps xdmrg++-flags
        INTERFACE_LINK_LIBRARIES
        xdmrg++-tid
        xdmrg++-math-eig-matvec
        xdmrg++-math-eig-o
        )




