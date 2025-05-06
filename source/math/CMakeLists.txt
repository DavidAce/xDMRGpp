add_subdirectory(svd)
add_subdirectory(eig)
add_subdirectory(rnd)
add_subdirectory(hash)
add_subdirectory(tenx)
add_subdirectory(fit)
add_library(xdmrg++-math INTERFACE)
target_link_libraries(xdmrg++-math INTERFACE
        xdmrg++-math-svd
        xdmrg++-math-rnd
        xdmrg++-math-eig
        xdmrg++-math-hash
        xdmrg++-math-tenx
        xdmrg++-math-fit
        )
target_link_libraries(xdmrg++-math INTERFACE xdmrg++-deps xdmrg++-flags)
