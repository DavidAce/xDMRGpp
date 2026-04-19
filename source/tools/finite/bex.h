#pragma once

struct BondExpansionConfig;
template<typename Scalar>
class TensorsFinite;
template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
struct BondExpansionResult;

namespace tools::finite::bex {
    /* clang-format off */
    template<typename Scalar>
    BondExpansionResult<Scalar> expand_bonds(TensorsFinite<Scalar> &tensors, BondExpansionConfig bcfg);
    template<typename Scalar> BondExpansionResult<Scalar> expand_bond_dmrg3s(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar> BondExpansionResult<Scalar> rexpand_bond_postopt_1site(StateFinite<Scalar> &state, ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar> BondExpansionResult<Scalar> rexpand_bond_preopt_1site(StateFinite<Scalar> &state, ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar> BondExpansionResult<Scalar> expand_bond_preopt_nsite(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg);
    template<typename Scalar> BondExpansionResult<Scalar> density_matrix_perturbation_preopt_1site(TensorsFinite<Scalar> &tensors, BondExpansionConfig bcfg);
    template<typename Scalar> BondExpansionResult<Scalar> density_matrix_perturbation_postopt_1site(TensorsFinite<Scalar> &tensors, BondExpansionConfig bcfg);
    /* clang-format on */
}
