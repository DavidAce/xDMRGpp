#include "contract_bnd_mps.impl.h"

using Scalar = cx64;
template void tools::common::contraction::contract_bnd_mps(      Scalar *       res_ptr, std::array<long,3> res_dims, //
                                                           const Scalar * const bnd_ptr, std::array<long,1> bnd_dims, //
                                                           const Scalar * const mps_ptr, std::array<long,3> mps_dims);

