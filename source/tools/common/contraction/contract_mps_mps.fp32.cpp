#include "contract_mps_mps.impl.h"

using Scalar = fp32;

template void tools::common::contraction::contract_mps_mps(      Scalar * res_ptr       , std::array<long,3> res_dims,    //
                                                           const Scalar * const mpsL_ptr, std::array<long,3> mpsL_dims,   //
                                                           const Scalar * const mpsR_ptr, std::array<long,3> mpsR_dims);
/* clang-format on */
