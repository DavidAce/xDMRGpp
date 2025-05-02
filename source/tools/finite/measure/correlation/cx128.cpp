#include "../correlation.impl.h"

using Scalar = cx128;

/* clang-format off */
template Scalar   tools::finite::measure::correlation<Scalar >(const StateFinite<Scalar > &state, const Eigen::Tensor<cx64, 2> &op1, const Eigen::Tensor<cx64, 2> &op2, long pos1, long pos2);

template Eigen::Tensor<Scalar , 2>  tools::finite::measure::correlation_matrix(const StateFinite<Scalar > &state, const Eigen::Tensor<cx64, 2> &op1, const Eigen::Tensor<cx64, 2> &op2);

template RealScalar<Scalar>  tools::finite::measure::structure_factor(const StateFinite<Scalar> &state, const Eigen::Tensor<Scalar, 2> &correlation_matrix);


/* clang-format on */
