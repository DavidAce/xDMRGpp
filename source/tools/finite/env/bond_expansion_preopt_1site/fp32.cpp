#include "bond_expansion_preopt_1site.impl.h"

using Scalar = fp32;

/* clang-format off */
template BondExpansionResult<Scalar>  tools::finite::env::rexpand_bond_preopt_1site(StateFinite<Scalar> &, ModelFinite<Scalar> &, EdgesFinite<Scalar> &, BondExpansionConfig bcfg);


