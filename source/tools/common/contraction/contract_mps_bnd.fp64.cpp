#include "contract_mps_bnd.impl.h"

using Scalar = fp64;

template void tools::common::contraction::contract_mps_bnd(      Scalar *       res_ptr, std::array<long,3> res_dims, //
                                                           const Scalar * const mps_ptr, std::array<long,3> mps_dims, //
                                                           const Scalar * const bnd_ptr, std::array<long,1> bnd_dims);

