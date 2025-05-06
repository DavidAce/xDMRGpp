#include "impl.h"

using Scalar = fp64;

/* clang-format off */

template std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_svdcompressed_mpos(std::vector<Eigen::Tensor<Scalar, 4>> mpos);

template std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_deparallelized_mpos(std::vector<Eigen::Tensor<Scalar, 4>> mpos);
