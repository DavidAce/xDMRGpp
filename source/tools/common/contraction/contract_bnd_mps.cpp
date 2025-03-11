#include "../contraction.h"
#include "debug/exceptions.h"
#include "math/tenx.h"
#include <fmt/ranges.h>


/* clang-format off */
using namespace tools::common::contraction;

template<typename Scalar>
void  tools::common::contraction::contract_bnd_mps(      Scalar * res_ptr      , std::array<long,3> res_dims,
                                                   const Scalar * const bnd_ptr, std::array<long,1> bnd_dims,
                                                   const Scalar * const mps_ptr, std::array<long,3> mps_dims){
    auto res = Eigen::TensorMap<Eigen::Tensor<Scalar,3>>(res_ptr,res_dims);
    auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(mps_ptr,mps_dims);
    auto bnd = Eigen::TensorMap<const Eigen::Tensor<Scalar,1>>(bnd_ptr,bnd_dims);
    if(mps_dims[1] != bnd_dims[0]) throw except::runtime_error("Dimension mismatch mps {} (idx 1) and bnd {} (idx 0)", mps_dims, bnd_dims);
    if(mps_dims != res_dims) throw except::runtime_error("Dimension mismatch mps {} and res {}", mps_dims, res_dims);
    auto &threads = tenx::threads::get();
    res.device(*threads->dev) = tenx::asDiagonal(bnd).contract(mps, tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2});
}

template void tools::common::contraction::contract_bnd_mps(      fp32 *       res_ptr, std::array<long,3> res_dims,
                                                           const fp32 * const bnd_ptr, std::array<long,1> bnd_dims,
                                                           const fp32 * const mps_ptr, std::array<long,3> mps_dims);
template void tools::common::contraction::contract_bnd_mps(      fp64 *       res_ptr, std::array<long,3> res_dims,
                                                           const fp64 * const bnd_ptr, std::array<long,1> bnd_dims,
                                                           const fp64 * const mps_ptr, std::array<long,3> mps_dims);
template void tools::common::contraction::contract_bnd_mps(      cx32 *       res_ptr, std::array<long,3> res_dims,
                                                           const cx32 * const bnd_ptr, std::array<long,1> bnd_dims,
                                                           const cx32 * const mps_ptr, std::array<long,3> mps_dims);
template void tools::common::contraction::contract_bnd_mps(      cx64 *       res_ptr, std::array<long,3> res_dims,
                                                           const cx64 * const bnd_ptr, std::array<long,1> bnd_dims,
                                                           const cx64 * const mps_ptr, std::array<long,3> mps_dims);

/* clang-format on */
