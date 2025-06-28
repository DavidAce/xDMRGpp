#include "density_matrix_perturbation_postopt_1site.impl.h"

using Scalar = fp128;

/* clang-format off */
template BondExpansionResult<Scalar>  tools::finite::env::density_matrix_perturbation_postopt_1site(TensorsFinite<Scalar> & tensors, BondExpansionConfig bcfg);


