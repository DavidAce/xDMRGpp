#include "contract_mps_mps_partial.impl.h"

using Scalar = fp128;

template void tools::common::contraction::contract_mps_mps_partial(      Scalar *       res_ptr , std::array<long,2> res_dims,
                                                                   const Scalar * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                   const Scalar * const mps2_ptr, std::array<long,3> mps2_dims,
                                                                   std::array<long,2> idx);
/* clang-format on */
