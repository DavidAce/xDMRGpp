#include "impl.h"

using Scalar = cx64;

/* clang-format off */

template std::pair<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 4>> tools::finite::mpo::swap_mpo(const Eigen::Tensor<Scalar, 4> &mpoL, const Eigen::Tensor<Scalar, 4> &mpoR);

template void tools::finite::mpo::swap_sites(ModelFinite<Scalar> &model, size_t posL, size_t posR, std::vector<size_t> &sites);