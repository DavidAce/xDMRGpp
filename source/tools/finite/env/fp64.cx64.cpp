#include "impl.h"

using Scalar = cx64;
using T = fp64;

/* clang-format off */

template void tools::finite::env::internal::get_optimally_mixed_block<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg, BondExpansionResult<Scalar> &res);

template void tools::finite::env::internal::set_mixing_factors_to_rnorm<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg, BondExpansionResult<Scalar> &res);

template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg, BondExpansionResult<Scalar> &res);





