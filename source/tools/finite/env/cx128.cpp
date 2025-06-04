#include "impl.h"

using Scalar = cx128;
using T = cx128;

/* clang-format off */

template void tools::finite::env::internal::get_optimally_mixed_block<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg, BondExpansionResult<Scalar> &res);

template void tools::finite::env::internal::set_mixing_factors_to_rnorm<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg, BondExpansionResult<Scalar> &res);

template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<T>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg, BondExpansionResult<Scalar> &res);

template BondExpansionResult<Scalar> tools::finite::env::get_mixing_factors_postopt_rnorm(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);

template BondExpansionResult<Scalar> tools::finite::env::get_mixing_factors_preopt_krylov(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);

template BondExpansionResult<Scalar>  tools::finite::env::expand_bond_postopt_1site(StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &, BondExpansionConfig bcfg);

template BondExpansionResult<Scalar>  tools::finite::env::expand_bond_preopt_nsite(StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &, BondExpansionConfig bcfg);

template void tools::finite::env::assert_edges_ene(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, const EdgesFinite<Scalar> &);

template void tools::finite::env::assert_edges_var(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, const EdgesFinite<Scalar> &);

template void tools::finite::env::assert_edges(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, const EdgesFinite<Scalar> &);

template void tools::finite::env::rebuild_edges_ene(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);

template void tools::finite::env::rebuild_edges_var(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);

template void tools::finite::env::rebuild_edges(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);



