#include "../contraction.h"
#include "math/tenx.h"

using namespace tools::common::contraction;

/* clang-format off */
template<typename Scalar>
Scalar tools::common::contraction::expectation_value(const Scalar * const ket_ptr, std::array<long,3> ket_dims,
                                                     const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                                                     const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                                                     const Scalar * const envR_ptr, std::array<long,3> envR_dims){


    // This measures the expectation value of some multisite mps with respect to some mpo operator and corresponding environments.
    // This is usually the energy E = <psi|H|psi> or variance V = <psi|(H-E)²|psi>
    // Note that the environments must contain the correct type of mpos

    assert(ket_dims[1]  == envL_dims[0]);
    assert(ket_dims[2]  == envR_dims[0]);
    assert(ket_dims[0]  == mpo_dims[2]);
    assert(envL_dims[2] == mpo_dims[0]);
    assert(envR_dims[2] == mpo_dims[1]);

    Eigen::Tensor<Scalar, 3> Hket(ket_dims);
    matrix_vector_product(Hket.data(), ket_ptr, ket_dims, mpo_ptr, mpo_dims, envL_ptr, envL_dims, envR_ptr, envR_dims);
    return contract_mps_mps_overlap(ket_ptr, ket_dims, Hket.data(), Hket.dimensions()); // ket gets adjointed
}

/* clang-format on */

/* clang-format off */
template<typename Scalar>
Scalar tools::common::contraction::expectation_value(const Scalar * const bra_ptr, std::array<long,3> bra_dims,
                                                     const Scalar * const ket_ptr, std::array<long,3> ket_dims,
                                                     const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                                                     const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                                                     const Scalar * const envR_ptr, std::array<long,3> envR_dims){

    // This measures the expectation value of some multisite mps with respect to some mpo operator and corresponding environments.
    // This is usually the energy E = <psi|H|psi> or variance V = <psi|(H-E)²|psi>
    // Note that the environments must contain the correct type of mpos

    assert(bra_dims[1]  == envL_dims[1]);
    assert(ket_dims[1]  == envL_dims[0]);
    assert(bra_dims[2]  == envR_dims[1]);
    assert(ket_dims[2]  == envR_dims[0]);
    assert(bra_dims[0]  == mpo_dims[3]);
    assert(ket_dims[0]  == mpo_dims[2]);
    assert(envL_dims[2] == mpo_dims[0]);
    assert(envR_dims[2] == mpo_dims[1]);

    Eigen::Tensor<Scalar, 3> Hket(ket_dims);
    matrix_vector_product(Hket.data(), ket_ptr, ket_dims, mpo_ptr, mpo_dims, envL_ptr, envL_dims, envR_ptr, envR_dims);
    return contract_mps_mps_overlap(bra_ptr, bra_dims, Hket.data(), Hket.dimensions()); // bra gets adjointed

}



