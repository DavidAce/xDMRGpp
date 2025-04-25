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

using tools::finite::measure::RealScalar;

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo,
                                                         const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpo, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}

/* clang-format off */
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<fp32, 3> &mps, const Eigen::Tensor<fp32, 4> &mpo, const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<cx32, 3> &mps, const Eigen::Tensor<cx32, 4> &mpo, const Eigen::Tensor<cx32, 3> &envL, const Eigen::Tensor<cx32, 3> &envR);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<fp64, 3> &mps, const Eigen::Tensor<fp64, 4> &mpo, const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const Eigen::Tensor<cx64, 4> &mpo, const Eigen::Tensor<cx64, 3> &envL, const Eigen::Tensor<cx64, 3> &envR);
template fp128 tools::finite::measure::residual_norm(const Eigen::Tensor<fp128, 3> &mps, const Eigen::Tensor<fp128, 4> &mpo, const Eigen::Tensor<fp128, 3> &envL, const Eigen::Tensor<fp128, 3> &envR);
template fp128 tools::finite::measure::residual_norm(const Eigen::Tensor<cx128, 3> &mps, const Eigen::Tensor<cx128, 4> &mpo, const Eigen::Tensor<cx128, 3> &envL, const Eigen::Tensor<cx128, 3> &envR);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                         const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpos, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<fp32, 3> &mps, const std::vector<Eigen::Tensor<fp32, 4>> &mpos, const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<cx32, 3> &mps, const std::vector<Eigen::Tensor<cx32, 4>> &mpos, const Eigen::Tensor<cx32, 3> &envL, const Eigen::Tensor<cx32, 3> &envR);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<fp64, 3> &mps, const std::vector<Eigen::Tensor<fp64, 4>> &mpos, const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<Eigen::Tensor<cx64, 4>> &mpos, const Eigen::Tensor<cx64, 3> &envL, const Eigen::Tensor<cx64, 3> &envR);
template fp128 tools::finite::measure::residual_norm(const Eigen::Tensor<fp128, 3> &mps, const std::vector<Eigen::Tensor<fp128, 4>> &mpos, const Eigen::Tensor<fp128, 3> &envL, const Eigen::Tensor<fp128, 3> &envR);
template fp128 tools::finite::measure::residual_norm(const Eigen::Tensor<cx128, 3> &mps, const std::vector<Eigen::Tensor<cx128, 4>> &mpos, const Eigen::Tensor<cx128, 3> &envL, const Eigen::Tensor<cx128, 3> &envR);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                         const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                         const env_pair<const EnvEne<Scalar> &>                           &envs) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto mpo_vec = std::vector<Eigen::Tensor<Scalar, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().template MPO_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.template get_block_as<Scalar>(), envs.R.template get_block_as<Scalar>());
}
/* clang-format off */
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<cx64>> > &mpo_refs, const env_pair<const EnvEne<cx64> &> &envs);
template fp128 tools::finite::measure::residual_norm(const Eigen::Tensor<cx128, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<cx128>> > &mpo_refs, const env_pair<const EnvEne<cx128> &> &envs);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                         const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                         const env_pair<const EnvVar<Scalar> &>                           &envs) {
    // Calculate the residual_norm r = |H²v - E²v|
    auto mpo_vec = std::vector<Eigen::Tensor<Scalar, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().template MPO2_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.template get_block_as<Scalar>(), envs.R.template get_block_as<Scalar>());
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<fp32, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<fp32>> > &mpo_refs, const env_pair<const EnvVar<fp32> &> &envs);
template fp32 tools::finite::measure::residual_norm(const Eigen::Tensor<cx32, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<cx32>> > &mpo_refs, const env_pair<const EnvVar<cx32> &> &envs);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<fp64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<fp64>> > &mpo_refs, const env_pair<const EnvVar<fp64> &> &envs);
template fp64 tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<cx64>> > &mpo_refs, const env_pair<const EnvVar<cx64> &> &envs);
template fp128 tools::finite::measure::residual_norm(const Eigen::Tensor<fp128, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<fp128>> > &mpo_refs, const env_pair<const EnvVar<fp128> &> &envs);
template fp128 tools::finite::measure::residual_norm(const Eigen::Tensor<cx128, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<cx128>> > &mpo_refs, const env_pair<const EnvVar<cx128> &> &envs);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H1(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                            const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    const auto &mps = state.template get_multisite_mps<Scalar>(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_ene(sites);
    return residual_norm(mps, mpo, env);
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm_H1<fp32>(const std::vector<size_t> &sites, const StateFinite<fp32> &state, const ModelFinite<fp32> &model, const EdgesFinite<fp32> &edges);
template fp32 tools::finite::measure::residual_norm_H1<cx32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges);
template fp64 tools::finite::measure::residual_norm_H1<fp64>(const std::vector<size_t> &sites, const StateFinite<fp64> &state, const ModelFinite<fp64> &model, const EdgesFinite<fp64> &edges);
template fp64 tools::finite::measure::residual_norm_H1<cx64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template fp128 tools::finite::measure::residual_norm_H1<fp128>(const std::vector<size_t> &sites, const StateFinite<fp128> &state, const ModelFinite<fp128> &model, const EdgesFinite<fp128> &edges);
template fp128 tools::finite::measure::residual_norm_H1<cx128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H1(const std::vector<size_t> &sites, const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H1<Scalar>(sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template fp32  tools::finite::measure::residual_norm_H1<fp32>(const std::vector<size_t> &sites, const TensorsFinite<fp32> &tensors);
template fp32  tools::finite::measure::residual_norm_H1<cx32>(const std::vector<size_t> &sites, const TensorsFinite<cx32> &tensors);
template fp64  tools::finite::measure::residual_norm_H1<fp64>(const std::vector<size_t> &sites, const TensorsFinite<fp64> &tensors);
template fp64  tools::finite::measure::residual_norm_H1<cx64>(const std::vector<size_t> &sites, const TensorsFinite<cx64> &tensors);
template fp128 tools::finite::measure::residual_norm_H1<fp128>(const std::vector<size_t> &sites, const TensorsFinite<fp128> &tensors);
template fp128 tools::finite::measure::residual_norm_H1<cx128>(const std::vector<size_t> &sites, const TensorsFinite<cx128> &tensors);

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H1(const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H1<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm_H1<fp32>(const TensorsFinite<fp32> &tensors);
template fp32 tools::finite::measure::residual_norm_H1<cx32>(const TensorsFinite<cx32> &tensors);
template fp64 tools::finite::measure::residual_norm_H1<fp64>(const TensorsFinite<fp64> &tensors);
template fp64 tools::finite::measure::residual_norm_H1<cx64>(const TensorsFinite<cx64> &tensors);
template fp128 tools::finite::measure::residual_norm_H1<fp128>(const TensorsFinite<fp128> &tensors);
template fp128 tools::finite::measure::residual_norm_H1<cx128>(const TensorsFinite<cx128> &tensors);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                            const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    const auto &mps = state.template get_multisite_mps<Scalar>(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_var(sites);
    return residual_norm(mps, mpo, env);
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm_H2<fp32>(const std::vector<size_t> &sites, const StateFinite<fp32> &state, const ModelFinite<fp32> &model, const EdgesFinite<fp32> &edges);
template fp32 tools::finite::measure::residual_norm_H2<cx32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges);
template fp64 tools::finite::measure::residual_norm_H2<fp64>(const std::vector<size_t> &sites, const StateFinite<fp64> &state, const ModelFinite<fp64> &model, const EdgesFinite<fp64> &edges);
template fp64 tools::finite::measure::residual_norm_H2<cx64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template fp128 tools::finite::measure::residual_norm_H2<fp128>(const std::vector<size_t> &sites, const StateFinite<fp128> &state, const ModelFinite<fp128> &model, const EdgesFinite<fp128> &edges);
template fp128 tools::finite::measure::residual_norm_H2<cx128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H2<Scalar>(sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
/* clang-format off */
template fp32  tools::finite::measure::residual_norm_H2<fp32>(const std::vector<size_t> &sites, const TensorsFinite<fp32> &tensors);
template fp32  tools::finite::measure::residual_norm_H2<cx32>(const std::vector<size_t> &sites, const TensorsFinite<cx32> &tensors);
template fp64  tools::finite::measure::residual_norm_H2<fp64>(const std::vector<size_t> &sites, const TensorsFinite<fp64> &tensors);
template fp64  tools::finite::measure::residual_norm_H2<cx64>(const std::vector<size_t> &sites, const TensorsFinite<cx64> &tensors);
template fp128 tools::finite::measure::residual_norm_H2<fp128>(const std::vector<size_t> &sites, const TensorsFinite<fp128> &tensors);
template fp128 tools::finite::measure::residual_norm_H2<cx128>(const std::vector<size_t> &sites, const TensorsFinite<cx128> &tensors);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H2<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template fp32  tools::finite::measure::residual_norm_H2<fp32>(const TensorsFinite<fp32> &tensors);
template fp32  tools::finite::measure::residual_norm_H2<cx32>(const TensorsFinite<cx32> &tensors);
template fp64  tools::finite::measure::residual_norm_H2<fp64>(const TensorsFinite<fp64> &tensors);
template fp64  tools::finite::measure::residual_norm_H2<cx64>(const TensorsFinite<cx64> &tensors);
template fp128 tools::finite::measure::residual_norm_H2<fp128>(const TensorsFinite<fp128> &tensors);
template fp128 tools::finite::measure::residual_norm_H2<cx128>(const TensorsFinite<cx128> &tensors);

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_full(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                              const EdgesFinite<Scalar> &edges) {
    // Calculate the residual_norm r = |Hv - Ev|, where H is the full Hamiltonian and v is the full mps
    // Note that the full residual norm is equal to the sqrt(Var(H)) = Std(H)
    tools::log->info("Calculating residual norm with full system");
    auto        sites = num::range<size_t>(0, state.get_length());
    const auto &mps   = state.template get_multisite_mps<Scalar>(sites);
    const auto &mpo   = model.get_mpo(sites);
    const auto &env   = edges.get_multisite_env_ene(sites);
    return residual_norm<Scalar>(mps, mpo, env);
}
/* clang-format off */
template fp32 tools::finite::measure::residual_norm_full<fp32>(const StateFinite<fp32> &state, const ModelFinite<fp32> &model, const EdgesFinite<fp32> &edges);
template fp32 tools::finite::measure::residual_norm_full<cx32>(const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges);
template fp64 tools::finite::measure::residual_norm_full<fp64>(const StateFinite<fp64> &state, const ModelFinite<fp64> &model, const EdgesFinite<fp64> &edges);
template fp64 tools::finite::measure::residual_norm_full<cx64>(const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template fp128 tools::finite::measure::residual_norm_full<fp128>(const StateFinite<fp128> &state, const ModelFinite<fp128> &model, const EdgesFinite<fp128> &edges);
template fp128 tools::finite::measure::residual_norm_full<cx128>(const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_full(const TensorsFinite<Scalar> &tensors) {
    return residual_norm_full<Scalar>(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
/* clang-format off */
template fp32  tools::finite::measure::residual_norm_full<fp32>(const TensorsFinite<fp32> &tensors);
template fp32  tools::finite::measure::residual_norm_full<cx32>(const TensorsFinite<cx32> &tensors);
template fp64  tools::finite::measure::residual_norm_full<fp64>(const TensorsFinite<fp64> &tensors);
template fp64  tools::finite::measure::residual_norm_full<cx64>(const TensorsFinite<cx64> &tensors);
template fp128 tools::finite::measure::residual_norm_full<fp128>(const TensorsFinite<fp128> &tensors);
template fp128 tools::finite::measure::residual_norm_full<cx128>(const TensorsFinite<cx128> &tensors);
/* clang-format on */