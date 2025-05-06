#include "matrix_inverse_vector_product.impl.h"

using Scalar = fp32;

template void tools::common::contraction::matrix_inverse_vector_product(Scalar             *res_ptr,                                 //
                                                                        const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                        const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                        const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                        const Scalar *const envR_ptr, std::array<long, 3> envR_dims, //
                                                                        InvMatVecCfg<Scalar> cfg);
