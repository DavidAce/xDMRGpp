#include "contract_env_mps_mpo.impl.h"

using T = cx64;
template void tools::common::contraction::contract_envL_mps_mpo(Eigen::Tensor<T, 3>       &res, //
                                                                const Eigen::Tensor<T, 3> &env, //
                                                                const Eigen::Tensor<T, 3> &mps, //
                                                                const Eigen::Tensor<T, 2> &mpo);
template void tools::common::contraction::contract_envR_mps_mpo(Eigen::Tensor<T, 3>       &res, //
                                                                const Eigen::Tensor<T, 3> &env, //
                                                                const Eigen::Tensor<T, 3> &mps, //
                                                                const Eigen::Tensor<T, 2> &mpo);

template void tools::common::contraction::contract_envL_mps_mpo(Eigen::Tensor<T, 3>       &res, //
                                                                const Eigen::Tensor<T, 3> &env, //
                                                                const Eigen::Tensor<T, 3> &mps, //
                                                                const Eigen::Tensor<T, 4> &mpo);
template void tools::common::contraction::contract_envR_mps_mpo(Eigen::Tensor<T, 3>       &res, //
                                                                const Eigen::Tensor<T, 3> &env, //
                                                                const Eigen::Tensor<T, 3> &mps, //
                                                                const Eigen::Tensor<T, 4> &mpo);

template void tools::common::contraction::contract_envL_mps_mpo(x2::Tensor<T, 3>          &res, //
                                                                const x2::Tensor<T, 3>    &env, //
                                                                const Eigen::Tensor<T, 3> &mps, //
                                                                const Eigen::Tensor<T, 4> &mpo);
template void tools::common::contraction::contract_envR_mps_mpo(x2::Tensor<T, 3>          &res, //
                                                                const x2::Tensor<T, 3>    &env, //
                                                                const Eigen::Tensor<T, 3> &mps, //
                                                                const Eigen::Tensor<T, 4> &mpo);