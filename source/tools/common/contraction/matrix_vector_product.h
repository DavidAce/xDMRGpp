#pragma once
#include "contraction_policy.h"
#include "math/tenx.h"
#include "math/x2/gemm.h"
#include "math/x2/view.h"
#include "tid/tid.h"
#include "tools/common/log.h"

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
    void matrix_vector_product(Scalar             *res_ptr,                                 //
                               const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                               const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                               const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                               const Scalar *const envR_ptr, std::array<long, 3> envR_dims);

    template<typename Scalar, typename mpo_type>
    void matrix_vector_product(Scalar             *res_ptr,                                 //
                               const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                               const std::vector<mpo_type> &mpos_shf,                       //
                               const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                               const Scalar *const envR_ptr, std::array<long, 3> envR_dims);

    template<typename Scalar>
    void matrix_vector_product(Eigen::Tensor<Scalar, 3>       &res,  //
                               const Eigen::Tensor<Scalar, 3> &mps,  //
                               const Eigen::Tensor<Scalar, 4> &mpo,  //
                               const x2::Tensor<Scalar, 3>    &envL, //
                               const x2::Tensor<Scalar, 3>    &envR);

    template<typename Scalar>
    void matrix_vector_product(Eigen::Tensor<Scalar, 3>                    &res,     //
                               const Eigen::Tensor<Scalar, 3>              &mps,     //
                               const std::vector<Eigen::Tensor<Scalar, 4>> &mpo_shf, //
                               const x2::Tensor<Scalar, 3>                 &envL,    //
                               const x2::Tensor<Scalar, 3>                 &envR);

    template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
    void matrix_vector_product(TensorWrite<res_type>      &res,  //
                               const TensorRead<mps_type> &mps,  //
                               const TensorRead<mpo_type> &mpo,  //
                               const TensorRead<env_type> &envL, //
                               const TensorRead<env_type> &envR) {
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mpo_type::NumIndices == 4 and "Wrong mpo tensor rank != 4 passed to calculation of matrix_vector_product");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of matrix_vector_product");
        auto &res_ref   = static_cast<res_type &>(res);
        auto  mps_eval  = tenx::asEval(mps);
        auto  mpo_eval  = tenx::asEval(mpo);
        auto  envL_eval = tenx::asEval(envL);
        auto  envR_eval = tenx::asEval(envR);
        matrix_vector_product(res_ref.data(),                           //
                              mps_eval.data(), mps_eval.dimensions(),   //
                              mpo_eval.data(), mpo_eval.dimensions(),   //
                              envL_eval.data(), envL_eval.dimensions(), //
                              envR_eval.data(), envR_eval.dimensions());
    }

    template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
    void matrix_vector_product(TensorWrite<res_type>       &res,      //
                               const TensorRead<mps_type>  &mps,      //
                               const std::vector<mpo_type> &mpos_shf, //
                               const TensorRead<env_type>  &envL,     //
                               const TensorRead<env_type>  &envR) {
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of matrix_vector_product");
        auto &res_ref  = static_cast<res_type &>(res);
        auto  mps_eval = tenx::asEval(mps);
        // auto  mpo_eval = tenx::asEval(mpo);
        auto envL_eval = tenx::asEval(envL);
        auto envR_eval = tenx::asEval(envR);
        matrix_vector_product(res_ref.data(),         //
                              mps_eval.data(),        //
                              mps_eval.dimensions(),  //
                              mpos_shf,               //
                              envL_eval.data(),       //
                              envL_eval.dimensions(), //
                              envR_eval.data(),       //
                              envR_eval.dimensions());
    }

    template<typename Scalar>
    void matrix_vector_product(Eigen::Tensor<Scalar, 3>       &res,  //
                               const Eigen::Tensor<Scalar, 3> &mps,  //
                               const Eigen::Tensor<Scalar, 4> &mpo,  //
                               const x2::Tensor<Scalar, 3>    &envL, //
                               const x2::Tensor<Scalar, 3>    &envR);

    template<typename Scalar>
    requires(sfinae::is_any_v<typename Eigen::NumTraits<Scalar>::Real, fp32, fp64, fp128>)
    void matrix_vector_product(Eigen::Tensor<Scalar, 3>       &res,  //
                               const Eigen::Tensor<Scalar, 3> &mps,  //
                               const Eigen::Tensor<Scalar, 4> &mpo,  //
                               const EnvEne<Scalar>           &envL, //
                               const EnvEne<Scalar>           &envR);
    template<typename Scalar>
    requires(sfinae::is_any_v<typename Eigen::NumTraits<Scalar>::Real, fp32, fp64, fp128>)
    void matrix_vector_product(Eigen::Tensor<Scalar, 3>       &res,  //
                               const Eigen::Tensor<Scalar, 3> &mps,  //
                               const Eigen::Tensor<Scalar, 4> &mpo,  //
                               const EnvVar<Scalar>           &envL, //
                               const EnvVar<Scalar>           &envR);

    template<typename mps_type, typename mpo_type, typename env_type>
    mps_type matrix_vector_product(const mps_type &mps,  //
                                   const mpo_type &mpo,  //
                                   const env_type &envL, //
                                   const env_type &envR) {
        using Scalar = typename mps_type::Scalar;
        Eigen::Tensor<Scalar, 3> result(mps.dimensions());
        matrix_vector_product(result, mps, mpo, envL, envR);
        return result;
    }

}