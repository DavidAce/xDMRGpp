#pragma once
#include "math/svd/config.h"
#include <optional>
#include <vector>

struct BondExpansionConfig;
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
template<typename Scalar>
struct BondExpansionResult;
namespace tools::finite::opt {
    struct OptMeta;
}
namespace tools::finite::env {

    using OptMeta = tools::finite::opt::OptMeta;
    /* clang-format off */
    template<typename Scalar>  BondExpansionResult<Scalar>   expand_bond_dmrg3s(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar>  BondExpansionResult<Scalar>   rexpand_bond_postopt_1site(StateFinite<Scalar> &state, ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar>  BondExpansionResult<Scalar>   rexpand_bond_preopt_1site(StateFinite<Scalar> &state, ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar>  BondExpansionResult<Scalar>   expand_bond_preopt_nsite(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar> void                           assert_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           assert_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           assert_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);

    /* clang-format on */

}