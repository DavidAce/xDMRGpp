#pragma once
#include "math/tenx.h"

template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;

namespace tools::common::contraction {
    template<typename T> using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    template<typename T> using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename T>
    using TensorWrite = Eigen::TensorBase<T, Eigen::WriteAccessors>;
    template<typename T>
    using TensorRead = Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>;

    template<typename Scalar>
    Scalar expectation_value(const Scalar *const ket_ptr, std::array<long, 3> ket_dims,   //
                             const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                             const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                             const Scalar *const envR_ptr, std::array<long, 3> envR_dims);

    template<typename Scalar>
    Scalar expectation_value(const Scalar *const bra_ptr, std::array<long, 3> bra_dims,   //
                             const Scalar *const ket_ptr, std::array<long, 3> ket_dims,   //
                             const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                             const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                             const Scalar *const envR_ptr, std::array<long, 3> envR_dims);

    template<typename mps_type, typename mpo_type, typename env_type>
    auto expectation_value(const TensorRead<mps_type> &mps,  //
                           const TensorRead<mpo_type> &mpo,  //
                           const TensorRead<env_type> &envL, //
                           const TensorRead<env_type> &envR) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of expectation_value");
        static_assert(mpo_type::NumIndices == 4 and "Wrong mpo tensor rank != 4 passed to calculation of expectation_value");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of expectation_value");
        auto mps_eval  = tenx::asEval(mps);
        auto mpo_eval  = tenx::asEval(mpo);
        auto envL_eval = tenx::asEval(envL);
        auto envR_eval = tenx::asEval(envR);

        return expectation_value(mps_eval.data(), mps_eval.dimensions(), mpo_eval.data(), mpo_eval.dimensions(), envL_eval.data(), envL_eval.dimensions(),
                                 envR_eval.data(), envR_eval.dimensions());
    }

    template<typename bra_type, typename ket_type, typename mpo_type, typename env_type>
    auto expectation_value(const TensorRead<bra_type> &bra,  //
                           const TensorRead<ket_type> &ket,  //
                           const TensorRead<mpo_type> &mpo,  //
                           const TensorRead<env_type> &envL, //
                           const TensorRead<env_type> &envR) {
        static_assert(bra_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of expectation_value");
        static_assert(ket_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of expectation_value");
        static_assert(mpo_type::NumIndices == 4 and "Wrong mpo tensor rank != 4 passed to calculation of expectation_value");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of expectation_value");
        auto bra_eval  = tenx::asEval(bra);
        auto ket_eval  = tenx::asEval(ket);
        auto mpo_eval  = tenx::asEval(mpo);
        auto envL_eval = tenx::asEval(envL);
        auto envR_eval = tenx::asEval(envR);

        return expectation_value(bra_eval.data(), bra_eval.dimensions(),   //
                                 ket_eval.data(), ket_eval.dimensions(),   //
                                 mpo_eval.data(), mpo_eval.dimensions(),   //
                                 envL_eval.data(), envL_eval.dimensions(), //
                                 envR_eval.data(), envR_eval.dimensions());
    }

    template<typename Scalar>
    Scalar expectation_value(const Eigen::Tensor<Scalar, 3> &mps,  //
                             const Eigen::Tensor<Scalar, 4> &mpo,  //
                             const EnvEne<Scalar>           &envL, //
                             const EnvEne<Scalar>           &envR);
    template<typename Scalar>
    Scalar expectation_value(const Eigen::Tensor<Scalar, 3> &mps,  //
                             const Eigen::Tensor<Scalar, 4> &mpo,  //
                             const EnvVar<Scalar>           &envL, //
                             const EnvVar<Scalar>           &envR);

}