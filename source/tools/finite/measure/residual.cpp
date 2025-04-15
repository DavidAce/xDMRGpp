#include "residual.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                               const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpo, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}

/* clang-format off */
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<fp64, 3> &mps, const Eigen::Tensor<fp64, 4> &mpo, const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<fp32, 3> &mps, const Eigen::Tensor<fp32, 4> &mpo, const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const Eigen::Tensor<cx64, 4> &mpo, const Eigen::Tensor<cx64, 3> &envL, const Eigen::Tensor<cx64, 3> &envR);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<cx32, 3> &mps, const Eigen::Tensor<cx32, 4> &mpo, const Eigen::Tensor<cx32, 3> &envL, const Eigen::Tensor<cx32, 3> &envR);
/* clang-format on */

template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>              &mps,
                                                                               const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                                               const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpos, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<fp32, 3> &mps, const std::vector<Eigen::Tensor<fp32, 4>> &mpos, const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<fp64, 3> &mps, const std::vector<Eigen::Tensor<fp64, 4>> &mpos, const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<cx32, 3> &mps, const std::vector<Eigen::Tensor<cx32, 4>> &mpos, const Eigen::Tensor<cx32, 3> &envL, const Eigen::Tensor<cx32, 3> &envR);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<Eigen::Tensor<cx64, 4>> &mpos, const Eigen::Tensor<cx64, 3> &envL, const Eigen::Tensor<cx64, 3> &envR);
/* clang-format on */

template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>                           &mps,
                                                                               const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                                                               const env_pair<const EnvEne &>                           &envs) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto mpo_vec = std::vector<Eigen::Tensor<Scalar, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().MPO_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.get_block_as<Scalar>(), envs.R.get_block_as<Scalar>());
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<fp32, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvEne &> &envs);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<fp64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvEne &> &envs);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<cx32, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvEne &> &envs);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvEne &> &envs);
/* clang-format on */

template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>                           &mps,
                                                                               const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                                                               const env_pair<const EnvVar &>                           &envs) {
    // Calculate the residual_norm r = |H²v - E²v|
    auto mpo_vec = std::vector<Eigen::Tensor<Scalar, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().MPO2_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.get_block_as<Scalar>(), envs.R.get_block_as<Scalar>());
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<fp32, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvVar &> &envs);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<fp64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvVar &> &envs);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<cx32, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvVar &> &envs);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvVar &> &envs);
/* clang-format on */


template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_H1(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    const auto &mps = state.get_multisite_mps<Scalar>(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_ene(sites);
    return residual_norm(mps, mpo, env);
}
template fp32 tools::finite::measure::residual_norm_H1<fp32>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp64 tools::finite::measure::residual_norm_H1<fp64>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp32 tools::finite::measure::residual_norm_H1<cx32>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp64 tools::finite::measure::residual_norm_H1<cx64>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);


template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_H1(const std::vector<size_t> &sites, const TensorsFinite &tensors) {
    return residual_norm_H1<Scalar>(sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template fp32 tools::finite::measure::residual_norm_H1<fp32>(const std::vector<size_t> &sites, const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H1<fp64>(const std::vector<size_t> &sites, const TensorsFinite &tensors);
template fp32 tools::finite::measure::residual_norm_H1<cx32>(const std::vector<size_t> &sites, const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H1<cx64>(const std::vector<size_t> &sites, const TensorsFinite &tensors);


template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_H1(const TensorsFinite &tensors) {
    return residual_norm_H1<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template fp32 tools::finite::measure::residual_norm_H1<fp32>(const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H1<fp64>(const TensorsFinite &tensors);
template fp32 tools::finite::measure::residual_norm_H1<cx32>(const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H1<cx64>(const TensorsFinite &tensors);




template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    const auto &mps = state.get_multisite_mps<Scalar>(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_var(sites);
    return residual_norm(mps, mpo, env);
}
template fp32 tools::finite::measure::residual_norm_H2<fp32>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp64 tools::finite::measure::residual_norm_H2<fp64>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp32 tools::finite::measure::residual_norm_H2<cx32>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp64 tools::finite::measure::residual_norm_H2<cx64>(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);


template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const TensorsFinite &tensors) {
    return residual_norm_H2<Scalar>(sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template fp32 tools::finite::measure::residual_norm_H2<fp32>(const std::vector<size_t> &sites, const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H2<fp64>(const std::vector<size_t> &sites, const TensorsFinite &tensors);
template fp32 tools::finite::measure::residual_norm_H2<cx32>(const std::vector<size_t> &sites, const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H2<cx64>(const std::vector<size_t> &sites, const TensorsFinite &tensors);

template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_H2(const TensorsFinite &tensors) {
    return residual_norm_H2<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template fp32 tools::finite::measure::residual_norm_H2<fp32>(const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H2<fp64>(const TensorsFinite &tensors);
template fp32 tools::finite::measure::residual_norm_H2<cx32>(const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_H2<cx64>(const TensorsFinite &tensors);


template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_full(const StateFinite &state, const ModelFinite &model,
                                                                                    const EdgesFinite &edges) {
    // Calculate the residual_norm r = |Hv - Ev|, where H is the full Hamiltonian and v is the full mps
    // Note that the full residual norm is equal to the sqrt(Var(H)) = Std(H)
    tools::log->info("Calculating residual norm with full system");
    auto        sites = num::range<size_t>(0, state.get_length());
    const auto &mps   = state.get_multisite_mps<Scalar>(sites);
    const auto &mpo   = model.get_mpo(sites);
    const auto &env   = edges.get_multisite_env_ene(sites);
    return residual_norm<Scalar>(mps, mpo, env);
}
template fp32 tools::finite::measure::residual_norm_full<fp32>(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp64 tools::finite::measure::residual_norm_full<fp64>(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp32 tools::finite::measure::residual_norm_full<cx32>(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);
template fp64 tools::finite::measure::residual_norm_full<cx64>(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges);

template<typename Scalar>
tools::finite::measure::RealType<Scalar> tools::finite::measure::residual_norm_full(const TensorsFinite &tensors) {
    return residual_norm_full<Scalar>(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template fp32 tools::finite::measure::residual_norm_full<fp32>(const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_full<fp64>(const TensorsFinite &tensors);
template fp32 tools::finite::measure::residual_norm_full<cx32>(const TensorsFinite &tensors);
template fp64 tools::finite::measure::residual_norm_full<cx64>(const TensorsFinite &tensors);