#pragma once
#include "math/svd/config.h"
#include <optional>
#include <vector>
template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;
enum class BondExpansionPolicy;
enum class OptAlgo;
enum class OptRitz;
enum class BondExpansionPolicy;
template<typename Scalar>
struct BondExpansionResult;
namespace tools::finite::opt {
    struct OptMeta;
}
namespace tools::finite::env {

    using OptMeta = tools::finite::opt::OptMeta;
    /* clang-format off */
    namespace internal {
        template<typename T, typename Scalar> void set_mixing_factors_to_rnorm(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta, BondExpansionResult<Scalar> &res);
        template<typename T, typename Scalar> void set_mixing_factors_to_stdv_H(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta, BondExpansionResult<Scalar> &res);
        template<typename T, typename Scalar> void get_optimally_mixed_block(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta, BondExpansionResult<Scalar> &res);
    }
    template<typename Scalar>  BondExpansionResult<Scalar> get_mixing_factors_postopt_rnorm(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta);
    template<typename Scalar>  BondExpansionResult<Scalar> get_mixing_factors_preopt_krylov(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta);

    template<typename Scalar>  std::array<double, 2>         get_optimal_mixing_factor_ene(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionPolicy bep);
    template<typename Scalar>  std::array<double, 2>         get_optimal_mixing_factor_var(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionPolicy bep);
    template<typename Scalar>  double                        get_optimal_mixing_factor_ene_old(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionPolicy bep);
    template<typename Scalar>  double                        get_optimal_mixing_factor_var_old(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionPolicy bep);
    template<typename Scalar>  BondExpansionResult<Scalar>   expand_bond_ssite_preopt(StateFinite<Scalar> &state, ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, const OptMeta &opt_meta);
    template<typename Scalar>  BondExpansionResult<Scalar>   expand_bond_postopt_1site(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, const OptMeta &opt_meta);
    template<typename Scalar>  BondExpansionResult<Scalar>   expand_bond_preopt_nsite(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, const OptMeta &opt_meta);
    template<typename Scalar> void                          assert_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                          assert_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                          assert_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                          rebuild_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                          rebuild_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                          rebuild_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);

    /* clang-format on */

}