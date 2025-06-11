#include "bond_expansion_preopt_nsite.impl.h"

using Scalar = cx32;

/* clang-format off */
template BondExpansionResult<Scalar> tools::finite::env::expand_bond_preopt_nsite(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                            EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);

