#include "bond_expansion_dmrg3s.impl.h"

using Scalar = fp128;

/* clang-format off */
template BondExpansionResult<Scalar> tools::finite::env::expand_bond_dmrg3s(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                            EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);

