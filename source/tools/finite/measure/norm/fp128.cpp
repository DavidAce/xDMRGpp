#include "../norm.impl.h"

using Scalar = fp128;

/* clang-format off */

template RealScalar<Scalar>  tools::finite::measure::norm_1site(const StateFinite<Scalar> &state);
template RealScalar<Scalar>  tools::finite::measure::norm_state(const StateFinite<Scalar> &state);

template Eigen::Tensor<Scalar, 2> tools::finite::measure::isometry_left(const StateFinite<Scalar> &state, Eigen::Index pos);
template Eigen::Tensor<Scalar, 2> tools::finite::measure::isometry_right(const StateFinite<Scalar> &state, Eigen::Index pos);