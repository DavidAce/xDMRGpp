#include "../expectation_value.impl.h"

using CalcType = fp64;
using Scalar   = cx32;
using OpType = cx64;
using MpoType = Scalar;

/* clang-format off */

template CalcType  tools::finite::measure::expectation_value<CalcType>(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<OpType>> &ops);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<MpoType, 4>> &mpos);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<MpoType, 4>> &mpos,  const Eigen::Tensor<MpoType, 1> &ledge, const Eigen::Tensor<MpoType, 1> &redge);

template Eigen::Tensor<CalcType, 1>  tools::finite::measure::expectation_values<CalcType>(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op);

template Eigen::Tensor<CalcType, 1>  tools::finite::measure::expectation_values<CalcType>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvEne<Scalar>> &envs);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvVar<Scalar>> &envs);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);

template CalcType  tools::finite::measure::expectation_value<CalcType>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);

/* clang-format on */
