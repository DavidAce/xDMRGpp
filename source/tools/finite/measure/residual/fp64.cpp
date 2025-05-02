#include "../residual.impl.h"

using Scalar = fp64;

/* clang-format off */
template RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo, const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

template RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

template RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>> > &mpo_refs, const env_pair<const EnvEne<Scalar> &> &envs);

template RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>> > &mpo_refs, const env_pair<const EnvVar<Scalar> &> &envs);

template RealScalar<Scalar> tools::finite::measure::residual_norm_H1<Scalar>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template RealScalar<Scalar>  tools::finite::measure::residual_norm_H1<Scalar>(const std::vector<size_t> &sites, const TensorsFinite<Scalar> &tensors);

template RealScalar<Scalar> tools::finite::measure::residual_norm_H1<Scalar>(const TensorsFinite<Scalar> &tensors);

template RealScalar<Scalar> tools::finite::measure::residual_norm_H2<Scalar>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template RealScalar<Scalar>  tools::finite::measure::residual_norm_H2<Scalar>(const std::vector<size_t> &sites, const TensorsFinite<Scalar> &tensors);

template RealScalar<Scalar>  tools::finite::measure::residual_norm_H2<Scalar>(const TensorsFinite<Scalar> &tensors);

template RealScalar<Scalar> tools::finite::measure::residual_norm_full<Scalar>(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template RealScalar<Scalar>  tools::finite::measure::residual_norm_full<Scalar>(const TensorsFinite<Scalar> &tensors);

/* clang-format on */
