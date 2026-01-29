#include "matrix_vector_product_gemm_x2.impl.h"

using Scalar = long double;

using namespace tools::common::contraction;
template void tools::common::contraction::matrix_vector_product_gemm_x2(Scalar *res_ptr, const Scalar *const mps_ptr, std::array<long, 3> mps_dims, //
                                                                const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,                  //
                                                                const Scalar *const envL_ptr, std::array<long, 3> envL_dims,                //
                                                                const Scalar *const envR_ptr, std::array<long, 3> envR_dims);               //

