#include "residual_expansion.impl.h"

using Scalar = cx128;

/* clang-format off */
template BondExpansionResult<Scalar>  tools::finite::env::rexpand_bond_postopt_1site(StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &, const OptMeta &);


