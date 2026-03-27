# Installation

## Requirements

The project currently requires:

- A C++23 compiler. The code is tested with GCC 14 and newer, and with Clang 19.
- [CMake](https://cmake.org/) 3.24 or newer. Recent CMake is needed in order to use the shipped presets directly.

[Conan](https://conan.io/) 2 is optional, but convenient if you want CMake to install dependencies through the Conan-based presets or through `DMRG_PACKAGE_MANAGER=conan`. A minimal setup is:

```bash
python -m pip install conan
conan profile detect
conan remote add conan-dmrg https://neumann.theophys.kth.se/artifactory/api/conan/conan-dmrg
```

The `conan-dmrg` remote is needed in order to fetch the current version of [h5pp](https://github.com/DavidAce/h5pp).

## Dependencies

`xDMRG++` depends on a number of external libraries for dense and sparse linear algebra, tensor operations, file I/O, and command-line parsing.

- Some BLAS, LAPACK and Lapacke implementation. Typical choices are [FlexiBLAS](https://www.mpi-magdeburg.mpg.de/projects/flexiblas), [Intel MKL](https://software.intel.com/en-us/mkl), or [OpenBLAS](https://github.com/xianyi/OpenBLAS). The CMake variable [`BLA_VENDOR`](https://cmake.org/cmake/help/latest/module/FindBLAS.html) can be used to guide detection.
- [Eigen](https://eigen.tuxfamily.org/) for matrix algebra and tensor operations.
- [TBLIS](https://github.com/MatthewsResearchGroup/tblis) for selected tensor contractions.
- [Arpack](https://github.com/opencollab/arpack-ng) for iterative eigenvalue problems.
- [Arpackpp](https://github.com/m-reuter/eigsolver_properties) as a C++ front-end to Arpack.
- [PRIMME](https://github.com/primme/primme) for iterative eigenvalue calculations.
- [h5pp](https://github.com/DavidAce/h5pp) as the HDF5 wrapper used for simulation output.
- [CLI11](https://github.com/CLIUtils/CLI11) for command-line parsing.
- [Backward-cpp](https://github.com/bombela/backward-cpp) for stack traces.

If you use the `find` dependency mode, these libraries must already be visible to CMake through your environment, your package manager, or explicit hints such as `<PackageName>_ROOT`.

## Quick start with CMake Presets

The recommended way to configure the project is with [CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html). A preset gives a name to a complete CMake configuration, including the build directory, build type, dependency provider, and toolchain-related settings.

A typical example is `release-cmake-flexiblas-native`:

```bash
git clone git@github.com:DavidAce/xDMRGpp.git
cd xDMRGpp
cmake --list-presets
cmake --preset release-cmake-flexiblas-native
cmake --build --preset release-cmake-flexiblas-native
./build/release-cmake-flexiblas-native/xDMRG++ --config input/default.cfg
```

The configure step creates the build tree under `build/release-cmake-flexiblas-native`. After the executable has been built, you can edit `input/default.cfg` or point `--config` to another file under `input/`.

For most users this is the least error-prone way to build the project, because the preset already fixes the generator, build directory, and several toolchain choices.

## Automatic Dependency Installation

The CMake variable `DMRG_PACKAGE_MANAGER` selects how dependencies are found or installed:

| Option | Description |
| ---- | ---- |
| `find` **(default)** | Use CMake's `find_package` to locate dependencies that are already available in the environment. |
| `cmake` | Use the CMake dependency provider to download and install dependencies during configure. |
| `conan` | Use Conan-based dependency resolution. The presets with `conan` in their name also enable the Conan dependency provider. |

The `find` mode is usually the right choice when you manage dependencies with your operating system, Conda, Spack, a site installation on a cluster, or a manually maintained environment. The `cmake` and `conan` modes are useful when you want a more self-contained setup.

## CMake Options

Pass CMake options as `-D<OPTION>=<VALUE>` during configure.

| Var | Default | Description |
| ---- | ---- | ---- |
| `DMRG_PACKAGE_MANAGER` | `find` | Select the automatic dependency manager: `find`, `cmake`, `conan`. |
| `DMRG_USE_QUADMATH` | `FALSE` | Enable `__float128` from `quadmath.h` for `fLBIT` time evolution. |
| `DMRG_USE_FLOAT128` | `FALSE` | Enable `std::float128_t` for `fLBIT` time evolution. |
| `DMRG_ENABLE_FP32` | `ON` | Build explicit instantiations for `fp32`. |
| `DMRG_ENABLE_FP64` | `ON` | Build explicit instantiations for `fp64`. |
| `DMRG_ENABLE_FP128` | `ON` | Build explicit instantiations for the `fp128` aliases. |
| `DMRG_ENABLE_CX32` | `ON` | Build explicit instantiations for `cx32`. |
| `DMRG_ENABLE_CX64` | `ON` | Build explicit instantiations for `cx64`. |
| `DMRG_ENABLE_CX128` | `ON` | Build explicit instantiations for the `cx128` aliases. |
| `DMRG_ENABLE_TBLIS` | `FALSE` | Use [TBLIS](https://github.com/MatthewsResearchGroup/tblis) for selected tensor contractions instead of Eigen. |
| `DMRG_ENABLE_TESTS` | `FALSE` | Enable unit tests and CTest targets. |
| `DMRG_BUILD_EXAMPLES` | `FALSE` | Build example programs. |
| `DMRG_BUILD_TOOLS` | `FALSE` | Build auxiliary tools. |
| `DMRG_ENABLE_DOCS` | `FALSE` | Build the documentation. |
| `DMRG_CMAKE_DEBUG` | `FALSE` | Print extra information during CMake configuration. |
| `EIGEN_USE_THREADS` | `TRUE` | Enable threaded Eigen tensor operations. |

In addition, variables such as [`<PackageName>_ROOT`](https://cmake.org/cmake/help/latest/variable/PackageName_ROOT.html) and [`<PackageName>_DIR`](https://cmake.org/cmake/help/latest/command/find_package.html) can be used to help CMake locate dependencies when `DMRG_PACKAGE_MANAGER=find`.
