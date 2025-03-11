#include "../contraction.h"
#include "math/tenx.h"

using namespace tools::common::contraction;

/* clang-format off */
template<typename Scalar>
Scalar tools::common::contraction::expectation_value(const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                                                     const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                                                     const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                                                     const Scalar * const envR_ptr, std::array<long,3> envR_dims){


    // This measures the expectation value of some multisite mps with respect to some mpo operator and corresponding environments.
    // This is usually the energy E = <psi|H|psi> or variance V = <psi|(H-E)²|psi>
    // Note that the environments must contain the correct type of mpos
    auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(mps_ptr,mps_dims);
    auto mpo = Eigen::TensorMap<const Eigen::Tensor<Scalar,4>>(mpo_ptr,mpo_dims);
    auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envL_ptr,envL_dims);
    auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envR_ptr,envR_dims);

    assert(mps.dimension(1)  == envL.dimension(0));
    assert(mps.dimension(2)  == envR.dimension(0));
    assert(mps.dimension(0)  == mpo.dimension(2));
    assert(envL.dimension(2) == mpo.dimension(0));
    assert(envR.dimension(2) == mpo.dimension(1));

    Eigen::Tensor<Scalar, 0> expval;
    auto &threads = tenx::threads::get();
    expval.device(*threads->dev) =
        envL
            .contract(mps,             tenx::idx({0}, {1}))
            .contract(mpo,             tenx::idx({2, 1}, {2, 0}))
            .contract(mps.conjugate(), tenx::idx({3, 0}, {0, 1}))
            .contract(envR,            tenx::idx({0, 2, 1}, {0, 1, 2}));

    Scalar result = expval(0);
    return result;
}

template fp32 tools::common::contraction::expectation_value(const fp32 * const mps_ptr,  std::array<long,3> mps_dims,
                                                            const fp32 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const fp32 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const fp32 * const envR_ptr, std::array<long,3> envR_dims);
template fp64 tools::common::contraction::expectation_value(const fp64 * const mps_ptr,  std::array<long,3> mps_dims,
                                                            const fp64 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const fp64 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const fp64 * const envR_ptr, std::array<long,3> envR_dims);
template cx32 tools::common::contraction::expectation_value(const cx32 * const mps_ptr,  std::array<long,3> mps_dims,
                                                            const cx32 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const cx32 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const cx32 * const envR_ptr, std::array<long,3> envR_dims);
template cx64 tools::common::contraction::expectation_value(const cx64 * const mps_ptr,  std::array<long,3> mps_dims,
                                                            const cx64 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const cx64 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const cx64 * const envR_ptr, std::array<long,3> envR_dims);
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
    auto bra = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(bra_ptr,bra_dims);
    auto ket = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(ket_ptr,ket_dims);
    auto mpo = Eigen::TensorMap<const Eigen::Tensor<Scalar,4>>(mpo_ptr,mpo_dims);
    auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envL_ptr,envL_dims);
    auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envR_ptr,envR_dims);

    assert(bra.dimension(1)  == envL.dimension(1));
    assert(ket.dimension(1)  == envL.dimension(0));
    assert(bra.dimension(2)  == envR.dimension(1));
    assert(ket.dimension(2)  == envR.dimension(0));
    assert(bra.dimension(0)  == mpo.dimension(3));
    assert(ket.dimension(0)  == mpo.dimension(2));
    assert(envL.dimension(2) == mpo.dimension(0));
    assert(envR.dimension(2) == mpo.dimension(1));

    Eigen::Tensor<Scalar, 0> expval;
    auto &threads = tenx::threads::get();
    expval.device(*threads->dev) =
        envL
            .contract(ket,             tenx::idx({0}, {1}))
            .contract(mpo,             tenx::idx({2, 1}, {2, 0}))
            .contract(bra.conjugate(), tenx::idx({3, 0}, {0, 1}))
            .contract(envR,            tenx::idx({0, 2, 1}, {0, 1, 2}));

    Scalar result = expval(0);
    return result;
}
template fp32 tools::common::contraction::expectation_value(const fp32 * const bra_ptr,  std::array<long,3> bra_dims,
                                                            const fp32 * const ket_ptr,  std::array<long,3> ket_dims,
                                                            const fp32 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const fp32 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const fp32 * const envR_ptr, std::array<long,3> envR_dims);
template fp64 tools::common::contraction::expectation_value(const fp64 * const bra_ptr,  std::array<long,3> bra_dims,
                                                            const fp64 * const ket_ptr,  std::array<long,3> ket_dims,
                                                            const fp64 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const fp64 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const fp64 * const envR_ptr, std::array<long,3> envR_dims);
template cx32 tools::common::contraction::expectation_value(const cx32 * const bra_ptr,  std::array<long,3> bra_dims,
                                                            const cx32 * const ket_ptr,  std::array<long,3> ket_dims,
                                                            const cx32 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const cx32 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const cx32 * const envR_ptr, std::array<long,3> envR_dims);
template cx64 tools::common::contraction::expectation_value(const cx64 * const bra_ptr,  std::array<long,3> bra_dims,
                                                            const cx64 * const ket_ptr,  std::array<long,3> ket_dims,
                                                            const cx64 * const mpo_ptr,  std::array<long,4> mpo_dims,
                                                            const cx64 * const envL_ptr, std::array<long,3> envL_dims,
                                                            const cx64 * const envR_ptr, std::array<long,3> envR_dims);


