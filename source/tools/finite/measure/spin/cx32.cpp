#include "../spin.impl.h"

using Scalar = cx32;
using Real = fp32;
using Cplx = cx32;
/* clang-format off */

template std::array<RealScalar<Scalar>, 3>  tools::finite::measure::spin_components(const StateFinite<Scalar> &state);

template RealScalar<Scalar>  tools::finite::measure::spin_component<Real>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &paulimatrix);

template RealScalar<Scalar>  tools::finite::measure::spin_component<Cplx>(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &paulimatrix);

template RealScalar<Scalar>  tools::finite::measure::spin_component(const StateFinite<Scalar> &state, std::string_view axis);

template RealScalar<Scalar>  tools::finite::measure::spin_alignment(const StateFinite<Scalar> &state, std::string_view axis);

template int tools::finite::measure::spin_sign(const StateFinite<Scalar> &state, std::string_view axis);

template std::array<Eigen::Tensor<RealScalar<Scalar>, 1>, 3>  tools::finite::measure::spin_expectation_values_xyz(const StateFinite<Scalar> &state);

template std::array<RealScalar<Scalar>, 3>  tools::finite::measure::spin_expectation_value_xyz(const StateFinite<Scalar> &state);

template std::array<Eigen::Tensor<RealScalar<Scalar>, 2>, 3>  tools::finite::measure::spin_correlation_matrix_xyz(const StateFinite<Scalar> &state);

template std::array<RealScalar<Scalar>, 3>  tools::finite::measure::spin_structure_factor_xyz(const StateFinite<Scalar> &state);

/* clang-format on */
