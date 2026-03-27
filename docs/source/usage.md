# Usage

## Input Configuration File

Simulation parameters are read from a text configuration file. A convenient starting point is `input/default.cfg`, and additional examples are provided under `input/`. These files select the algorithm, model, system size, numerical thresholds, output settings, and related run parameters.

`xDMRG++` reads a custom configuration file from the command line. A typical invocation is:

```bash
./xDMRG++ --config path/to/file.cfg
```

When using a preset build, the full command usually looks more like:

```bash
./build/release-cmake-flexiblas-native/xDMRG++ --config input/default.cfg
```

The full list of configuration namespaces and variables is documented under [Settings](settings.md).

## Output Data File

Simulation results are written to an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file. The output path is set in the input configuration, and by default it is `output/output.h5`.

The contents of the file depend on the active storage policy. A typical output file may include:

- Model parameters and run metadata.
- Iteration tables, convergence information, and measurements.
- Saved [matrix product state](https://en.wikipedia.org/wiki/Matrix_product_state) data for later analysis.
- State data needed to resume a simulation, when the relevant storage settings are enabled.

To inspect the data you can use any HDF5 viewer, such as [HDF Compass](https://github.com/HDFGroup/hdf-compass) or [HDFView](https://www.hdfgroup.org/download-hdfview/).

## Model Hamiltonians

The code currently includes the following one-dimensional model Hamiltonians:

- `ModelType::ising_sdual`: The self-dual transverse-field Ising model.
- `ModelType::ising_tf_rf`: The transverse-field Ising model with random on-site field.
- `ModelType::ising_majorana`: The Ising-Majorana model.
- `ModelType::xxz`: The XXZ spin chain.
- `ModelType::lbit`: The l-bit Hamiltonian, used to describe a [many-body localized](https://en.wikipedia.org/wiki/Many-body_localization) phase in terms of local integrals of motion.

These Hamiltonians are implemented as [matrix product operators](https://en.wikipedia.org/wiki/Matrix_product_state#Matrix_product_operator) under `source/tensors/model`. The corresponding configuration variables live in `source/config/settings.h`, and the active model is selected in the input file with `model::model_type`.

Adding a new model currently means implementing a new MPO site class and deriving it from `MpoSite`, following the structure of the existing models.
