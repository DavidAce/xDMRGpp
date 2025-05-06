#include "contract_mps_mps_overlap.impl.h"

using Scalar = cx32;

template Scalar tools::common::contraction::contract_mps_mps_overlap(const Scalar * const mps1_ptr, std::array<long,3> mps1_dims,
                                                                     const Scalar * const mps2_ptr, std::array<long,3> mps2_dims);

/* clang-format on */
