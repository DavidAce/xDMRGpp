#include "impl.h"

using Scalar = cx128;

/* clang-format off */

template Eigen::Tensor<Scalar, 3>  tools::infinite::opt::find_ground_state(const TensorsInfinite<Scalar> &tensors, std::string_view ritzstring);

template Eigen::Tensor<Scalar, 3>  tools::infinite::opt::time_evolve_state(const StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 2> &U);
