#include "contract_mps_mpo_env.impl.h"

using T = cx64;
/* clang-format off */
template void tools::common::contraction::contract_mps_mpo_env(      T *       res_ptr , std::array<long,2> res_dims,
                                                               const T * const env_ptr , std::array<long,2> env_dims,
                                                               const T * const mps_ptr , std::array<long,3> mps_dims,
                                                               const T * const mpo_ptr , std::array<long,2> mpo_dims);

template void tools::common::contraction::contract_mps_mpo_env(      T *       res_ptr , std::array<long,3> res_dims,
                                                               const T * const env_ptr , std::array<long,3> env_dims,
                                                               const T * const mps_ptr , std::array<long,3> mps_dims,
                                                               const T * const mpo_ptr , std::array<long,4> mpo_dims);


/* clang-format on */
