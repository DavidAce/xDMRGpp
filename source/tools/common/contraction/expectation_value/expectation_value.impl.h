#include "math/tenx.h"
#include "math/x2/gemm.h"
#include "math/x2/view.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/contraction/contraction_policy.h"
#include "tools/common/contraction/expectation_value.h"
#include "tools/common/contraction/matrix_vector_product.h"
#include "tools/common/log.h"

template<typename Scalar>
Scalar tools::common::contraction::expectation_value(const Scalar *const ket_ptr, std::array<long, 3> ket_dims,   //
                                                     const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                     const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                     const Scalar *const envR_ptr, std::array<long, 3> envR_dims) {
    // This measures the expectation value of some multisite mps with respect to some mpo operator and corresponding environments.
    // This is usually the energy E = <psi|H|psi> or variance V = <psi|(H-E)²|psi>
    // Note that the environments must contain the correct type of mpos

    assert(ket_dims[1] == envL_dims[0]);
    assert(ket_dims[2] == envR_dims[0]);
    assert(ket_dims[0] == mpo_dims[2]);
    assert(envL_dims[2] == mpo_dims[0]);
    assert(envR_dims[2] == mpo_dims[1]);

    Eigen::Tensor<Scalar, 3> Hket(ket_dims);
    tools::common::contraction::matrix_vector_product(Hket.data(), ket_ptr, ket_dims, mpo_ptr, mpo_dims, envL_ptr, envL_dims, envR_ptr, envR_dims);
    Scalar expv = tools::common::contraction::contract_mps_mps_overlap(ket_ptr, ket_dims, Hket.data(), Hket.dimensions()); // ket gets adjointed
    assert(std::isfinite(std::real(expv)));
    assert(std::isfinite(std::imag(expv)));
    return expv;
}

template<typename Scalar>
Scalar tools::common::contraction::expectation_value(const Scalar *const bra_ptr, std::array<long, 3> bra_dims,   //
                                                     const Scalar *const ket_ptr, std::array<long, 3> ket_dims,   //
                                                     const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                     const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                     const Scalar *const envR_ptr, std::array<long, 3> envR_dims) {
    // This measures the expectation value of some multisite mps with respect to some mpo operator and corresponding environments.
    // This is usually the energy E = <psi|H|psi> or variance V = <psi|(H-E)²|psi>
    // Note that the environments must contain the correct type of mpos

    assert(bra_dims[1] == envL_dims[1]);
    assert(ket_dims[1] == envL_dims[0]);
    assert(bra_dims[2] == envR_dims[1]);
    assert(ket_dims[2] == envR_dims[0]);
    assert(bra_dims[0] == mpo_dims[3]);
    assert(ket_dims[0] == mpo_dims[2]);
    assert(envL_dims[2] == mpo_dims[0]);
    assert(envR_dims[2] == mpo_dims[1]);

    Eigen::Tensor<Scalar, 3> Hket(ket_dims);
    tools::common::contraction::matrix_vector_product(Hket.data(), ket_ptr, ket_dims, mpo_ptr, mpo_dims, envL_ptr, envL_dims, envR_ptr, envR_dims);
    Scalar expv = tools::common::contraction::contract_mps_mps_overlap(bra_ptr, bra_dims, Hket.data(), Hket.dimensions()); // bra gets adjointed
    assert(std::isfinite(std::real(expv)));
    assert(std::isfinite(std::imag(expv)));
    return expv;
}

template<typename Scalar>
Scalar tools::common::contraction::expectation_value(const Eigen::Tensor<Scalar, 3> &mps,  //
                                                     const Eigen::Tensor<Scalar, 4> &mpo,  //
                                                     const EnvEne<Scalar>           &envL, //
                                                     const EnvEne<Scalar>           &envR) {
    Eigen::Tensor<Scalar, 3> res(mps.dimensions());
    matrix_vector_product(res, mps, mpo, envL, envR);
    Scalar expv = contract_mps_overlap(mps, res);
    assert(std::isfinite(std::real(expv)));
    assert(std::isfinite(std::imag(expv)));
    return expv;
}

template<typename Scalar>
Scalar tools::common::contraction::expectation_value(const Eigen::Tensor<Scalar, 3> &mps,  //
                                                     const Eigen::Tensor<Scalar, 4> &mpo,  //
                                                     const EnvVar<Scalar>           &envL, //
                                                     const EnvVar<Scalar>           &envR) {
    Eigen::Tensor<Scalar, 3> res(mps.dimensions());
    tools::common::contraction::matrix_vector_product(res, mps, mpo, envL, envR);
    Scalar expv = contract_mps_overlap(mps, res);
    assert(std::isfinite(std::real(expv)));
    assert(std::isfinite(std::imag(expv)));
    return expv;
}
