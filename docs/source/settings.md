# Settings

Simulation settings are read from a configuration file. In ordinary use this is a `.cfg` file passed with `--config`, for example:

```bash
./xDMRG++ --config input/default.cfg
```

The corresponding variables are defined in the namespace `settings` in `source/config/settings.h`, and the sections below are generated from those source comments through Doxygen and Breathe. The page is therefore both a user reference and a direct view of the documented configuration interface in the code.

Each setting is written as

```text
<namespace>::<variable> = <value>   // Optional comment
```

For example, the declaration

```c++
namespace settings::precision {
    inline double svd_truncation_min = 1e-12;
}
```

can be overridden from a configuration file with

```text
solvers::svd::truncation_min = 1e-12   // Optional comment
```

Example configuration files are provided under `input/`. They are usually the quickest way to discover which settings matter for a particular algorithm.

## Input

```{eval-rst}
.. doxygennamespace:: settings::input
   :project: DMRG++
   :path: config/settings.h
```

## Storage and Output

The output file path, storage policy, compression settings, and resume behavior are configured under `settings::storage`.

```{eval-rst}
.. doxygennamespace:: settings::storage
   :project: DMRG++
   :path: config/settings.h
```

## Model Hamiltonian

```{eval-rst}
.. doxygennamespace:: settings::model
   :project: DMRG++
   :path: config/settings.h
```

## Algorithms

### iDMRG

```{eval-rst}
.. doxygennamespace:: settings::idmrg
   :project: DMRG++
   :path: config/settings.h
```

### fDMRG

```{eval-rst}
.. doxygennamespace:: settings::fdmrg
   :project: DMRG++
   :path: config/settings.h
```

### xDMRG

```{eval-rst}
.. doxygennamespace:: settings::xdmrg
   :project: DMRG++
   :path: config/settings.h
```

### iTEBD

```{eval-rst}
.. doxygennamespace:: settings::itebd
   :project: DMRG++
   :path: config/settings.h
```

### fLBIT

```{eval-rst}
.. doxygennamespace:: settings::flbit
   :project: DMRG++
   :path: config/settings.h
```

## State Settings

```{eval-rst}
.. doxygennamespace:: settings::state
   :project: DMRG++
   :path: config/settings.h
```

## Schedule Settings

```{eval-rst}
.. doxygennamespace:: settings::schedule
   :project: DMRG++
   :path: config/settings.h
```

## Solver Settings

```{eval-rst}
.. doxygennamespace:: settings::solvers
   :project: DMRG++
   :path: config/settings.h
```

## Precision Settings

```{eval-rst}
.. doxygennamespace:: settings::precision
   :project: DMRG++
   :path: config/settings.h
```

## Convergence Settings

```{eval-rst}
.. doxygennamespace:: settings::convergence
   :project: DMRG++
   :path: config/settings.h
```

## Post Settings

```{eval-rst}
.. doxygennamespace:: settings::post
   :project: DMRG++
   :path: config/settings.h
```

## Threading

```{eval-rst}
.. doxygennamespace:: settings::threading
   :project: DMRG++
   :path: config/settings.h
```

## Timer

```{eval-rst}
.. doxygennamespace:: settings::timer
   :project: DMRG++
   :path: config/settings.h
```

## Console

```{eval-rst}
.. doxygennamespace:: settings::console
   :project: DMRG++
   :path: config/settings.h
```
