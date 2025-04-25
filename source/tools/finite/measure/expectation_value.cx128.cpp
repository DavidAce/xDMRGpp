#include "expectation_value.impl.h"

using Scalar = cx128;
using Real   = fp128;

template fp32  tools::finite::measure::expectation_value<fp32>(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<cx32>> &ops);
template fp64  tools::finite::measure::expectation_value<fp64>(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<cx64>> &ops);
template fp128 tools::finite::measure::expectation_value<fp128>(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<cx128>> &ops);
template cx32  tools::finite::measure::expectation_value<cx32>(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<cx32>> &ops);
template cx64  tools::finite::measure::expectation_value<cx64>(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<cx64>> &ops);
template cx128 tools::finite::measure::expectation_value<cx128>(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<cx128>> &ops);

/* clang-format off */
template fp32  tools::finite::measure::expectation_value<fp32>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<cx32, 4>> &mpos);
template fp64  tools::finite::measure::expectation_value<fp64>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos);
template fp128 tools::finite::measure::expectation_value<fp128>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<cx128, 4>> &mpos);
template cx32  tools::finite::measure::expectation_value<cx32>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<cx32, 4>> &mpos);
template cx64  tools::finite::measure::expectation_value<cx64>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos);
template cx128 tools::finite::measure::expectation_value<cx128>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<cx128, 4>> &mpos);
/* clang-format on */

/* clang-format off */
template fp32  tools::finite::measure::expectation_value<fp32>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,  const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge);
template fp64  tools::finite::measure::expectation_value<fp64>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,  const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge);
template fp128 tools::finite::measure::expectation_value<fp128>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,  const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge);
template cx32  tools::finite::measure::expectation_value<cx32>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,  const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge);
template cx64  tools::finite::measure::expectation_value<cx64>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,  const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge);
template cx128 tools::finite::measure::expectation_value<cx128>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,  const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge);
/* clang-format on */

template Eigen::Tensor<fp32, 1>  tools::finite::measure::expectation_values<fp32>(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<fp64, 1>  tools::finite::measure::expectation_values<fp64>(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<fp128, 1> tools::finite::measure::expectation_values<fp128>(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<cx32, 1>  tools::finite::measure::expectation_values<cx32>(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<cx64, 1>  tools::finite::measure::expectation_values<cx64>(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<cx128, 1> tools::finite::measure::expectation_values<cx128>(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op);

template Eigen::Tensor<fp32, 1>  tools::finite::measure::expectation_values<fp32>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<fp64, 1>  tools::finite::measure::expectation_values<fp64>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<fp128, 1> tools::finite::measure::expectation_values<fp128>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<cx32, 1>  tools::finite::measure::expectation_values<cx32>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<cx64, 1>  tools::finite::measure::expectation_values<cx64>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<cx128, 1> tools::finite::measure::expectation_values<cx128>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op);

/* clang-format off */
template fp32  tools::finite::measure::expectation_value<fp32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvEne<Scalar>> &envs);
template fp64  tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvEne<Scalar>> &envs);
template fp128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvEne<Scalar>> &envs);
template cx32  tools::finite::measure::expectation_value<cx32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvEne<Scalar>> &envs);
template cx64  tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvEne<Scalar>> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvEne<Scalar>> &envs);

template fp32  tools::finite::measure::expectation_value<fp32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvVar<Scalar>> &envs);
template fp64  tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvVar<Scalar>> &envs);
template fp128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvVar<Scalar>> &envs);
template cx32  tools::finite::measure::expectation_value<cx32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvVar<Scalar>> &envs);
template cx64  tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvVar<Scalar>> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvVar<Scalar>> &envs);

template fp32  tools::finite::measure::expectation_value<fp32>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs);
template fp64  tools::finite::measure::expectation_value<fp64>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs);
template fp128 tools::finite::measure::expectation_value<fp128>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs);
template cx32  tools::finite::measure::expectation_value<cx32>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs);
template cx64  tools::finite::measure::expectation_value<cx64>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs);

template fp32  tools::finite::measure::expectation_value<fp32>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs);
template fp64  tools::finite::measure::expectation_value<fp64>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs);
template fp128 tools::finite::measure::expectation_value<fp128>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs);
template cx32  tools::finite::measure::expectation_value<cx32>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs);
template cx64  tools::finite::measure::expectation_value<cx64>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs);

template fp32  tools::finite::measure::expectation_value<fp32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp64  tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx32  tools::finite::measure::expectation_value<cx32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx64  tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);

template fp32  tools::finite::measure::expectation_value<fp32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp64  tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx32  tools::finite::measure::expectation_value<cx32>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx64  tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);

template fp32  tools::finite::measure::expectation_value<fp32>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp64  tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx32  tools::finite::measure::expectation_value<cx32>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx64  tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvEne<Scalar> &> &envs, std::optional<svd::config> svd_cfg);

template fp32  tools::finite::measure::expectation_value<fp32>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp64  tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template fp128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx32  tools::finite::measure::expectation_value<cx32>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx64  tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<const EnvVar<Scalar> &> &envs, std::optional<svd::config> svd_cfg);
