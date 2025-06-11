#include "bond_expansion_postopt_1site.impl.h"

using Scalar = cx128;

/* clang-format off */
template BondExpansionResult<Scalar>  tools::finite::env::rexpand_bond_postopt_1site(StateFinite<Scalar> &, ModelFinite<Scalar> &, EdgesFinite<Scalar> &, BondExpansionConfig bcfg);


