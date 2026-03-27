# Introduction

The [density matrix renormalization group](https://en.wikipedia.org/wiki/Density_matrix_renormalization_group) (DMRG) is a variational method for one-dimensional quantum systems. In `xDMRG++`, many-body states are represented as [matrix product states](https://en.wikipedia.org/wiki/Matrix_product_state) (MPS), and Hamiltonians as [matrix product operators](https://en.wikipedia.org/wiki/Matrix_product_state#Matrix_product_operator) (MPO). The code is aimed at finite and infinite lattice calculations in one dimension, with an emphasis on finite-chain excited states and dynamics in the [many-body localized](https://en.wikipedia.org/wiki/Many-body_localization) regime.

The available algorithms are:

- ***x*DMRG:** *Excited state* DMRG. Targets interior eigenstates on finite chains.
- ***f*DMRG:** *finite* DMRG. Targets extremal eigenstates, most commonly the ground state, on finite chains.
- ***i*DMRG:** *infinite* DMRG. Targets ground states of infinite translationally invariant systems.
- ***i*TEBD:** *Imaginary Time Evolving Block Decimation*. Finds ground states of infinite translationally invariant systems by imaginary-time evolution.
- ***f*LBIT:** *Finite* l-BIT. Time evolution of finite systems in terms of local integrals of motion in the [many-body localized](https://en.wikipedia.org/wiki/Many-body_localization) regime.

The pages below focus on how to configure, build, and run the code. The documentation does not attempt to reproduce every implementation detail from the source tree, but it does point to the main configuration interfaces and to the parts of the code where new models and algorithms are introduced.
