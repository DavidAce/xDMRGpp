#include "env.impl.h"

using Scalar = cx128;
using T = fp128;

/* clang-format off */

template void tools::finite::env::internal::get_optimally_mixed_block<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta, BondExpansionResult<Scalar> &res);

template void tools::finite::env::internal::set_mixing_factors_to_rnorm<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta, BondExpansionResult<Scalar> &res);

template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta, BondExpansionResult<Scalar> &res);





