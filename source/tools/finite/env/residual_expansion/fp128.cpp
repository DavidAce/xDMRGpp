#include "residual_expansion.impl.h"

using Scalar = fp128;

/* clang-format off */
template BondExpansionResult<Scalar>  tools::finite::env::rexpand_bond_postopt_1site(StateFinite<Scalar> &, ModelFinite<Scalar> &, EdgesFinite<Scalar> &, const OptMeta &);


