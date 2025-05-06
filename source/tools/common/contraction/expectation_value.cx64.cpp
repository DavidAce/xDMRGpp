#include "expectation_value.impl.h"

using Scalar = cx64;

template Scalar tools::common::contraction::expectation_value(const Scalar * const mps_ptr,  std::array<long,3> mps_dims,   //
                                                              const Scalar * const mpo_ptr,  std::array<long,4> mpo_dims,   //
                                                              const Scalar * const envL_ptr, std::array<long,3> envL_dims,  //
                                                              const Scalar * const envR_ptr, std::array<long,3> envR_dims); //

template Scalar tools::common::contraction::expectation_value(const Scalar * const bra_ptr,  std::array<long,3> bra_dims,   //
                                                              const Scalar * const ket_ptr,  std::array<long,3> ket_dims,   //
                                                              const Scalar * const mpo_ptr,  std::array<long,4> mpo_dims,   //
                                                              const Scalar * const envL_ptr, std::array<long,3> envL_dims,  //
                                                              const Scalar * const envR_ptr, std::array<long,3> envR_dims); //



