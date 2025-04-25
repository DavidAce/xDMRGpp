#include "../contraction.h"
#include "debug/exceptions.h"
#include "math/tenx.h"

using namespace tools::common::contraction;

/* clang-format off */
template<typename Scalar>
Scalar tools::common::contraction::contract_mps_mps_overlap(const Scalar * const mps1_ptr, std::array<long,3> mps1_dims,
                                                            const Scalar * const mps2_ptr, std::array<long,3> mps2_dims){
    // auto t_con = tid::tic_token("contract_mps_mps_overlap", tid::level::highest);
    auto size1 = Eigen::internal::array_prod(mps1_dims);
    auto size2 = Eigen::internal::array_prod(mps2_dims);
    auto mps1 = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,1>>(mps1_ptr, size1);
    auto mps2 = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,1>>(mps2_ptr, size2);
    if(size1 != size2) throw except::runtime_error("Size mismatch mps1 {} and mps2 {}",size1, size2);
    return mps1.dot(mps2); // Calls gemv // TODO: Check that this works with the tests (used to be conjugate on mps2!)
}

template fp32   tools::common::contraction::contract_mps_mps_overlap(const fp32 * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                     const fp32 * const mps2_ptr, std::array<long,3> mps2_dims);
template fp64   tools::common::contraction::contract_mps_mps_overlap(const fp64 * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                     const fp64 * const mps2_ptr, std::array<long,3> mps2_dims);
template fp128  tools::common::contraction::contract_mps_mps_overlap(const fp128 * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                     const fp128 * const mps2_ptr, std::array<long,3> mps2_dims);
template cx32   tools::common::contraction::contract_mps_mps_overlap(const cx32 * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                     const cx32 * const mps2_ptr, std::array<long,3> mps2_dims);
template cx64   tools::common::contraction::contract_mps_mps_overlap(const cx64 * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                     const cx64 * const mps2_ptr, std::array<long,3> mps2_dims);
template cx128  tools::common::contraction::contract_mps_mps_overlap(const cx128 * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                     const cx128 * const mps2_ptr, std::array<long,3> mps2_dims);

/* clang-format on */
