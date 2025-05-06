#pragma once
#include "../contraction.h"
#include "math/tenx.h"

using namespace tools::common::contraction;

/* clang-format off */
template<typename Scalar>
void tools::common::contraction::contract_mps_mps_partial(Scalar *       res_ptr , std::array<long,2> res_dims,
                                                          const Scalar * const mps1_ptr, std::array<long,3> mps1_dims,
                                                          const Scalar * const mps2_ptr, std::array<long,3> mps2_dims,
                                                          std::array<long,2> idx){
    auto res  = Eigen::TensorMap<Eigen::Tensor<Scalar,2>>(res_ptr,res_dims);
    auto mps1 = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(mps1_ptr, mps1_dims);
    auto mps2 = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(mps2_ptr, mps2_dims);
    auto idxs = tenx::idx(idx,idx);
    auto &threads = tenx::threads::get();
    res.device(*threads->dev) = mps1.conjugate().contract(mps2, idxs);
}
/* clang-format on */
