#include "../contraction.h"
#include "math/tenx.h"

/* clang-format off */
using namespace tools::common::contraction;

template<typename Scalar>
void  tools::common::contraction::contract_mps_bnd(      Scalar * res_ptr      , std::array<long,3> res_dims,
                                                   const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                                                   const Scalar * const bnd_ptr, std::array<long,1> bnd_dims){
    assert(mps_dims[2] == bnd_dims[0]);
    assert(mps_dims == res_dims);
    auto res = Eigen::TensorMap<Eigen::Tensor<Scalar,3>>(res_ptr,res_dims);
    auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(mps_ptr,mps_dims);
    auto bnd = Eigen::TensorMap<const Eigen::Tensor<Scalar,1>>(bnd_ptr,bnd_dims);
    tenx::asDiagonalContract(res, bnd, mps , 2);
}


template void tools::common::contraction::contract_mps_bnd(      fp32 *       res_ptr, std::array<long,3> res_dims,
                                                           const fp32 * const mps_ptr, std::array<long,3> mps_dims,
                                                           const fp32 * const bnd_ptr, std::array<long,1> bnd_dims);
template void tools::common::contraction::contract_mps_bnd(      fp64 *       res_ptr, std::array<long,3> res_dims,
                                                           const fp64 * const mps_ptr, std::array<long,3> mps_dims,
                                                           const fp64 * const bnd_ptr, std::array<long,1> bnd_dims);
template void tools::common::contraction::contract_mps_bnd(      cx32 *       res_ptr, std::array<long,3> res_dims,
                                                           const cx32 * const mps_ptr, std::array<long,3> mps_dims,
                                                           const cx32 * const bnd_ptr, std::array<long,1> bnd_dims);
template void tools::common::contraction::contract_mps_bnd(      cx64 *       res_ptr, std::array<long,3> res_dims,
                                                           const cx64 * const mps_ptr, std::array<long,3> mps_dims,
                                                           const cx64 * const bnd_ptr, std::array<long,1> bnd_dims);
/* clang-format on */
