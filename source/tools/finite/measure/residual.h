#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include <vector>

template<typename Scalar>
class MpoSite;
template<typename Scalar>
class StateFinite;
template<typename Scalar>
class TensorsFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;
template<typename T>
struct env_pair;

namespace tools::finite::measure {
    template<typename T>  using RealScalar = decltype(std::real(std::declval<T>()));
    // template<typename Scalar>
    // using RealScalar = decltype(std::real(std::declval<Scalar>()));
    /* clang-format off */
    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm       (const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo, const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm       (const Eigen::Tensor<Scalar, 3> &mps, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,const Eigen::Tensor<Scalar, 3> &envL,const Eigen::Tensor<Scalar, 3> &envR);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm       (const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs, const env_pair<const EnvEne<Scalar> &> &envs);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm       (const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs, const env_pair<const EnvVar<Scalar> &> &envs);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_H1    (const std::vector<size_t> & sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> & edges);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_H1    (const std::vector<size_t> & sites, const TensorsFinite<Scalar> & tensors);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_H1    (const TensorsFinite<Scalar> & tensors);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_H2    (const std::vector<size_t> & sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> & edges);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_H2    (const std::vector<size_t> & sites, const TensorsFinite<Scalar> & tensors);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_H2    (const TensorsFinite<Scalar> & tensors);


    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_full  (const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> & edges);

    template<typename Scalar>
    [[nodiscard]] extern RealScalar<Scalar> residual_norm_full  (const TensorsFinite<Scalar> & tensors);

    /* clang-format on */
}