#include "../TensorsFinite.impl.h"

using Scalar = fp32;

template class TensorsFinite<Scalar>;

/* clang-format off */
template Eigen::Tensor<Scalar, 2>  contract_mpo_env(const Eigen::Tensor<Scalar, 4> &mpo, const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);
