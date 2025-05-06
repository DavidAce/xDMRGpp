#include "impl.h"

using Scalar = fp64;

/* clang-format off */
template std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_mpos_with_edges(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 1> &Ledge, const Eigen::Tensor<Scalar, 1> &Redge);
