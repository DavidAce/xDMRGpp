#include "contract_env_mps_mpo.impl.h"


using T = fp128;
template void tools::common::contraction::contract_env_mps_mpo(      T *       res_ptr , std::array<long,2> res_dims,
                                                               const T * const env_ptr , std::array<long,2> env_dims,
                                                               const T * const mps_ptr , std::array<long,3> mps_dims,
                                                               const T * const mpo_ptr , std::array<long,2> mpo_dims);


template void tools::common::contraction::contract_env_mps_mpo(      T *       res_ptr , std::array<long,3> res_dims,
                                                               const T * const env_ptr , std::array<long,3> env_dims,
                                                               const T * const mps_ptr , std::array<long,3> mps_dims,
                                                               const T * const mpo_ptr , std::array<long,4> mpo_dims);


