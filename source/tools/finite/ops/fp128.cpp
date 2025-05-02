#include "impl.h"

using Scalar = fp128;
using Real   = fp128;

/* clang-format off */
template void tools::finite::ops::apply_mpos(StateFinite<Scalar> &state, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 1> &Ledge, const Eigen::Tensor<Scalar, 1> &Redge, bool adjoint);

template void tools::finite::ops::apply_mpos(StateFinite<Scalar> &state, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 3> &Ledge, const Eigen::Tensor<Scalar, 3> &Redge, bool adjoint);

template void tools::finite::ops::apply_mpos_general(StateFinite<Scalar> &state, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const svd::config &svd_cfg);

template void tools::finite::ops::apply_mpos_general(StateFinite<Scalar> &state, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 1> &Ledge, const Eigen::Tensor<Scalar, 1> &Redge, const svd::config &svd_cfg);

template void tools::finite::ops::project_to_axis(StateFinite<Scalar> &state, const Eigen::Matrix2cd &paulimatrix, int sign, std::optional<svd::config> svd_cfg);

template std::optional<Real>  tools::finite::ops::get_spin_component_along_axis(StateFinite<Scalar> &state, std::string_view axis);

template int tools::finite::ops::project_to_nearest_axis(StateFinite<Scalar> &state, std::string_view axis, std::optional<svd::config> svd_cfg);

template Scalar  tools::finite::ops::overlap<Scalar>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2);

//template Scalar  tools::finite::ops::overlap<Real>(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2);
