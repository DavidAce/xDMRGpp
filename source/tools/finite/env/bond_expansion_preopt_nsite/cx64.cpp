#include "bond_expansion_preopt_nsite.impl.h"

using Scalar = cx64;

template BondExpansionResult<Scalar> tools::finite::env::expand_bond_preopt_nsite(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                            EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);

