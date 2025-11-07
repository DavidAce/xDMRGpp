[![Ubuntu 22.04](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-22.04.yml/badge.svg)](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-22.04.yml)
[![Ubuntu 24.04](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-24.04.yml/badge.svg)](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-22.04.yml)
[![codecov](https://codecov.io/gh/DavidAce/DMRG/branch/master/graph/badge.svg?token=9YE72CJ522)](https://codecov.io/gh/DavidAce/DMRG)
# xDMRG++

---

# Introduction

The [density matrix renormalization group](https://en.wikipedia.org/wiki/Density_matrix_renormalization_group) (DMRG) algorithm is a
variational technique used to calculate eigenstates of 1D quantum systems. DMRG works by iteratively optimizing a trial wave function in the
form of a [Matrix Product State](https://en.wikipedia.org/wiki/Matrix_product_states) (MPS), until obtaining an
eigenstate with high precision.

xDMRG++ includes several algorithms for open boundary systems:

- ***x*DMRG:** *Excited state* DMRG. For mid-spectrum eigenstates.
- ***f*DMRG:** *finite* DMRG. For groundstates.
- ***i*DMRG:** *infinite* DMRG. For groundstates of infinite translationally invariant systems.
- ***i*TEBD:** *Imaginary Time Evolving Block Decimation*. For groundstates of infinite translationally invariant systems.

One additional algorithm is included to study the dynamics in the Many-body Localized phase:

- ***f*LBIT:** *Finite* l-BIT. Time evolution of finite systems in terms of local integrals of motion (l-bits).


## Documentation

For more information on using xDMRG++, visit

https://kth-dmrg.readthedocs.io/en/latest/

### Working Notes (in construction)

See the [working notes](https://github.com/DavidAce/Notebooks/blob/master/DMRG%2B%2B/DMRG%2B%2B.pdf) for a more
theoretical background of this implementation.


---

# Usage

## Input Configuration File

The default input configuration file `input/input.cfg` sets simulation properties, such as algorithm, system size and
precision.
`xDMRG++` takes a custom input file from command-line arguments, e.g. `./xDMRG++ -c path/to/file.cfg`.  

## Output Data File

After execution the results are stored a binary file in HDF5 format. Its location is specified in the configuration
file `input/input.cfg`. By default, the output file is `output/output.h5`, which will contain values like the final
energies, entanglement entropies, and optionally the final state in MPS form.

To view the data you can use any hdf5-viewer, such as HDFCompass or HDFViewer.

## Model Hamiltonians

These model Hamiltonians of 1D quantum systems are included:

- `ModelType::ising_sdual`: The Self-dual transverse-field Ising model.
- `ModelType::ising_tf_rf`: The Transverse-field Ising model with random on-site field.
- `ModelType::lbit`: The l-bit Hamiltonian, used to describe a many-body localized phase
  in terms of its local integrals of motion (the l-bits).

The Hamiltonians are implemented as *Matrix Product Operators* (MPO), located under `source/tensors/model`. The model
type is selected using the input configuration file in `input/input.cfg`, with the option `settings::model::model_type`.
To add another model, one currently has to implement a new MPO and derive from `class_mpo_site` just like the existing
models.


---

# Installation


## Requirements

The following software is required to build the project:

- C++20 compiler (tested with gcc-12 and up and clang-18)
- CMake version >= 3.24 to use presets.

In addition, conan version 2 is recommended for dependency installation. 
To use conan: 

```
pip install conan
conan profile detect
conan remote add conan-dmrg https://neumann.theophys.kth.se/artifactory/api/conan/conan-dmrg
```
The conan-dmrg remote is needed to download the latest version of [**h5pp**](https://github.com/DavidAce/h5pp).


## Dependencies

- Some BLAS, LAPACK and Lapacke implementation. Choose either [FlexiBLAS](https://www.mpi-magdeburg.mpg.de/projects/flexiblas) with reference Lapacke,  [Intel MKL](https://software.intel.com/en-us/mkl)
  or [OpenBLAS](https://github.com/xianyi/OpenBLAS). Set the [`BLA_VENDOR`](https://cmake.org/cmake/help/latest/module/FindBLAS.html) variable to guide CMake. 
- [**Eigen**](http://eigen.tuxfamily.org) for tensor and matrix and linear algebra (tested with version >= 3.3).
- [**TBLIS**](https://github.com/MatthewsResearchGroup/tblis) for tensor contractions.
- [**Arpack**](https://github.com/opencollab/arpack-ng) Eigenvalue solver based on Fortran. Note that this in turn
  requires LAPACK and BLAS libraries, both of which are included in OpenBLAS.
- [**Arpackpp**](https://github.com/m-reuter/eigsolver_properties) C++ frontend for Arpack.
- [**primme**](https://github.com/primme/primme) Eigenvalue solver. Complements Arpack.
- [**h5pp**](https://github.com/DavidAce/h5pp) a wrapper for HDF5. Includes [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [spdlog](https://github.com/gabime/spdlog)
  and [fmt](https://github.com/fmtlib/fmt).
- [**CLI11**](https://github.com/CLIUtils/CLI11) input argument parser
- [**Backward-cpp**](https://github.com/bombela/backward-cpp) pretty stack trace printer.


## Quick start

- `git clone git@github.com:DavidAce/xDMRGpp.git` and `cd xDMRGpp`
- Configure `cmake --preset <preset>` (see the available presets with `cmake --list-presets`)
- Build with `cmake --build --preset <preset>`
- Modify `input/default.cfg` to configure a simulation.
- Run with `./build/<preset>/xDMRG++ -c input/default.cfg`.
- Find generated data in `output/output.h5`.

Some presets, with `conan` in their name, can use the 
[CMake Dependency Provider](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html#dependency-providers)
mechanism to let CMake call conan to install all the dependencies automatically.

## Automatic Dependency Installation

The CMake flag `DMRG_PACKAGE_MANAGER` controls the automated behavior for finding or installing dependencies. It can
take one of these strings:

| Option               | Description                                                                                         |
|----------------------|-----------------------------------------------------------------------------------------------------|
| `find` **(default)** | Use CMake's `find_package`  to find dependencies. (Use this with the CMake Presets labeled `conan`) |
| `cmake`              | Use CMake to download and install dependencies during configure.                                    |
|                      |                                                                                                     |


## CMake Options

 Add options to the CMake configuration as `cmake [-D<OPTION>=<VALUE>] ...`:

| Var                           | Default | Description                                                       |
|-------------------------------|---------|-------------------------------------------------------------------|
| `DMRG_USE_QUADMATH`           | `FALSE` | Use __float128 from quadmath.h for lbit time evolution            |
| `DMRG_USE_FLOAT128`           | `FALSE` | Use std::float128_t for lbit time evolution                       |
| `DMRG_ENABLE_TBLIS`           | `FALSE` | Use TBLIS for faster tensor contractions (FP64) instead of Eigen3 |
| `DMRG_ENABLE_TESTS`           | `FALSE` | Enable unit testing with ctest                                    |
| `DMRG_BUILD_EXAMPLES`         | `FALSE` | Build examples                                                    |
| `DMRG_BUILD_TOOLS`            | `FALSE` | Build tools                                                       |
| `DMRG_ENABLE_DOCS`            | `FALSE` | Build documentation                                               |
| `DMRG_CMAKE_DEBUG`            | `FALSE` | Extra information during CMake configuration                      |
| `COMPILER_PROFILE_BUILD`      | `FALSE` | Enable -ftime-trace (inspect with ClangBuildAnalyzer)             |
| `COMPILER_ENABLE_ASAN`        | `FALSE` | Enable runtime address sanitizer -fsanitize=address               |
| `COMPILER_ENABLE_USAN`        | `FALSE` | Enable undefined behavior sanitizer -fsanitize=undefined          |
| `COMPILER_ENABLE_PCH`         | `FALSE` | Enable precompiled headers to speed up compilation                |
| `COMPILER_ENABLE_COVERAGE`    | `FALSE` | Enable test coverage                                              |
| `EIGEN_USE_THREADS`           | `TRUE`  | Use STL threads to parallelize Eigen::Tensor                      |


In addition, variables such
as [`<PackageName>_ROOT`](https://cmake.org/cmake/help/latest/variable/PackageName_ROOT.html)
and [`<PackageName>_DIR`](https://cmake.org/cmake/help/latest/command/find_package.html) can be set to guide
CMake's `find_package` calls to find dependencies.


---

# Contact Information
For questions about xDMRG++ email David Aceituno aceituno `<at>` kth.se, or create a new [issue](https://github.com/DavidAce/xDMRGpp/issues) or [discussion](https://github.com/DavidAce/xDMRGpp/discussion).