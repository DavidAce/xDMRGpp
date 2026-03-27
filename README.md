[![Ubuntu 22.04](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-22.04.yml/badge.svg)](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-22.04.yml)
[![Ubuntu 24.04](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-24.04.yml/badge.svg)](https://github.com/DavidAce/xDMRGpp/actions/workflows/ubuntu-22.04.yml)
[![codecov](https://codecov.io/gh/DavidAce/DMRG/branch/master/graph/badge.svg?token=9YE72CJ522)](https://codecov.io/gh/DavidAce/DMRG)
# xDMRG++

---

# Introduction

The [density matrix renormalization group](https://en.wikipedia.org/wiki/Density_matrix_renormalization_group) (DMRG) is a variational method for one-dimensional quantum systems. `xDMRG++` represents states as [matrix product states](https://en.wikipedia.org/wiki/Matrix_product_state) (MPS) and Hamiltonians as [matrix product operators](https://en.wikipedia.org/wiki/Matrix_product_state#Matrix_product_operator) (MPO).

xDMRG++ includes several algorithms for one-dimensional systems:

- ***x*DMRG:** *Excited state* DMRG. For interior eigenstates on finite chains.
- ***f*DMRG:** *finite* DMRG. For ground states of finite chains.
- ***i*DMRG:** *infinite* DMRG. For ground states of infinite translationally invariant systems.
- ***i*TEBD:** *Imaginary Time Evolving Block Decimation*. For ground states of infinite translationally invariant systems.
- ***f*LBIT:** *Finite* l-BIT. Time evolution of finite systems in terms of local integrals of motion in the [many-body localized](https://en.wikipedia.org/wiki/Many-body_localization) regime.


## Documentation

For more information on using xDMRG++, visit the [documentation](https://kth-dmrg.readthedocs.io/en/latest/).

### Working Notes (in construction)

See the [working notes](https://github.com/DavidAce/Notebooks/blob/master/DMRG%2B%2B/DMRG%2B%2B.pdf) for implementation details and derivations that have not yet been merged into the main documentation.


---

# Usage

## Input Configuration File

A convenient starting point is `input/default.cfg`, and additional example configurations are provided under `input/`. The configuration file sets the algorithm, model, system size, precision, storage policy, and related run parameters.
`xDMRG++` takes a custom input file from the command line, for example `./xDMRG++ --config path/to/file.cfg`.

## Output Data File

Results are stored in an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file whose path is set in the input configuration. By default this is `output/output.h5`. Depending on the storage settings, the file can contain measurements, model parameters, iteration data, and saved MPS states for later analysis or resume.

To inspect the data you can use an HDF5 viewer such as [HDF Compass](https://github.com/HDFGroup/hdf-compass) or [HDFView](https://www.hdfgroup.org/download-hdfview/).

## Model Hamiltonians

These model Hamiltonians of one-dimensional quantum systems are included:

- `ModelType::ising_sdual`: The self-dual transverse-field Ising model.
- `ModelType::ising_tf_rf`: The transverse-field Ising model with random on-site field.
- `ModelType::ising_majorana`: The Ising-Majorana model.
- `ModelType::xxz`: The XXZ spin chain.
- `ModelType::lbit`: The l-bit Hamiltonian, used to describe a [many-body localized](https://en.wikipedia.org/wiki/Many-body_localization) phase in terms of local integrals of motion.

The Hamiltonians are implemented as [matrix product operators](https://en.wikipedia.org/wiki/Matrix_product_state#Matrix_product_operator) under `source/tensors/model`. The corresponding settings are defined in `source/config/settings.h`, and example input files are provided under `input/`. The model type is selected in the input configuration file with `model::model_type`.
To add another model, one currently has to implement a new MPO and derive from `MpoSite`, following the existing models.


---

# Installation


## Requirements

The following software is required to build the project:

- C++23 compiler (tested with GCC 14 and newer, and Clang 19)
- CMake version >= 3.24 to use presets

Conan 2 is optional, but recommended if you want to use the Conan-based presets or dependency provider.
To use Conan:

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


## Quick start with CMake Presets

The recommended way to configure this project is with [CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html). A preset gives a name to a complete CMake configuration, including the build directory, build type, selected dependency provider, and toolchain-related settings.

A typical example is `release-cmake-flexiblas-native`:

```bash
git clone git@github.com:DavidAce/xDMRGpp.git
cd xDMRGpp
cmake --list-presets
cmake --preset release-cmake-flexiblas-native
cmake --build --preset release-cmake-flexiblas-native
./build/release-cmake-flexiblas-native/xDMRG++ --config input/default.cfg
```

The configure step creates the build tree under `build/release-cmake-flexiblas-native`. After the executable has been built, you can adjust `input/default.cfg` or use another file from `input/`, and the generated data will be written to the output path specified in the configuration, which is `output/output.h5` by default.

## Automatic Dependency Installation

The CMake flag `DMRG_PACKAGE_MANAGER` selects how dependencies are found or installed. It can
take one of these strings:

| Option               | Description |
|----------------------|-------------|
| `find` **(default)** | Use CMake's `find_package` to locate dependencies already available in the environment. |
| `cmake`              | Use the CMake dependency provider to download and install dependencies during configure. |
| `conan`              | Use Conan-based dependency resolution. The presets with `conan` in their name also enable the Conan dependency provider. |


## CMake Options

 Add options to the CMake configuration as `cmake [-D<OPTION>=<VALUE>] ...`:

| Var                           | Default | Description |
|-------------------------------|---------|-------------|
| `DMRG_PACKAGE_MANAGER`        | `find`  | Select the automatic dependency manager: `find`, `cmake`, `conan`. |
| `DMRG_USE_QUADMATH`           | `FALSE` | Enable `__float128` from `quadmath.h` for `fLBIT` time evolution. |
| `DMRG_USE_FLOAT128`           | `FALSE` | Enable `std::float128_t` for `fLBIT` time evolution. |
| `DMRG_ENABLE_FP32`            | `ON`    | Build explicit instantiations for `fp32`. |
| `DMRG_ENABLE_FP64`            | `ON`    | Build explicit instantiations for `fp64`. |
| `DMRG_ENABLE_FP128`           | `ON`    | Build explicit instantiations for the `fp128` aliases. |
| `DMRG_ENABLE_CX32`            | `ON`    | Build explicit instantiations for `cx32`. |
| `DMRG_ENABLE_CX64`            | `ON`    | Build explicit instantiations for `cx64`. |
| `DMRG_ENABLE_CX128`           | `ON`    | Build explicit instantiations for the `cx128` aliases. |
| `DMRG_ENABLE_TBLIS`           | `FALSE` | Use [TBLIS](https://github.com/MatthewsResearchGroup/tblis) for selected tensor contractions instead of Eigen. |
| `DMRG_ENABLE_TESTS`           | `FALSE` | Enable unit tests and CTest targets. |
| `DMRG_BUILD_EXAMPLES`         | `FALSE` | Build example programs. |
| `DMRG_BUILD_TOOLS`            | `FALSE` | Build auxiliary tools. |
| `DMRG_ENABLE_DOCS`            | `FALSE` | Build the documentation. |
| `DMRG_CMAKE_DEBUG`            | `FALSE` | Print extra information during CMake configuration. |
| `EIGEN_USE_THREADS`           | `TRUE`  | Enable threaded Eigen tensor operations. |


In addition, variables such
as [`<PackageName>_ROOT`](https://cmake.org/cmake/help/latest/variable/PackageName_ROOT.html)
and [`<PackageName>_DIR`](https://cmake.org/cmake/help/latest/command/find_package.html) can be set to guide
CMake's `find_package` calls to find dependencies.


---

# Contact Information
For questions about xDMRG++ email David Aceituno aceituno `<at>` kth.se, or create a new [issue](https://github.com/DavidAce/xDMRGpp/issues) or [discussion](https://github.com/DavidAce/xDMRGpp/discussion).
