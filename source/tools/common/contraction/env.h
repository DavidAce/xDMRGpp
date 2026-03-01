#pragma once
#include <Eigen/Core>
// Eigen goes first
#include "math/tenx/eval.h"
#include "math/tenx/threads.h"

template<typename Scalar> class MatrixLikeOperator;
template<typename Scalar> struct IterativeLinearSolverConfig;

template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;

namespace x2 {
    template<typename Scalar, int rank>
    class Tensor;
}

namespace tools::common::contraction {
    template<typename T> using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    template<typename T> using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename T>
    using TensorWrite = Eigen::TensorBase<T, Eigen::WriteAccessors>;
    template<typename T>
    using TensorRead = Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>;

    template<typename Scalar>
    void contract_envL_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                               const Eigen::Tensor<Scalar, 3> &env, //
                               const Eigen::Tensor<Scalar, 3> &mps, //
                               const Eigen::Tensor<Scalar, 2> &mpo);
    template<typename Scalar>
    void contract_envR_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                               const Eigen::Tensor<Scalar, 3> &env, //
                               const Eigen::Tensor<Scalar, 3> &mps, //
                               const Eigen::Tensor<Scalar, 2> &mpo);

    template<typename Scalar>
    void contract_envL_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                               const Eigen::Tensor<Scalar, 3> &env, //
                               const Eigen::Tensor<Scalar, 3> &mps, //
                               const Eigen::Tensor<Scalar, 4> &mpo);
    template<typename Scalar>
    void contract_envR_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                               const Eigen::Tensor<Scalar, 3> &env, //
                               const Eigen::Tensor<Scalar, 3> &mps, //
                               const Eigen::Tensor<Scalar, 4> &mpo);

    template<typename Scalar>
    void contract_envL_mps_mpo(x2::Tensor<Scalar, 3>          &res, //
                               const x2::Tensor<Scalar, 3>    &env, //
                               const Eigen::Tensor<Scalar, 3> &mps, //
                               const Eigen::Tensor<Scalar, 4> &mpo);
    template<typename Scalar>
    void contract_envR_mps_mpo(x2::Tensor<Scalar, 3>          &res, //
                               const x2::Tensor<Scalar, 3>    &env, //
                               const Eigen::Tensor<Scalar, 3> &mps, //
                               const Eigen::Tensor<Scalar, 4> &mpo);

    // template<typename res_type, typename env_type, typename mps_type, typename mpo_type>
    // void contract_envL_mps_mpo(TensorWrite<res_type> &res, const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo)
    // {
    //     static_assert((res_type::NumIndices == 2 or res_type::NumIndices == 3) and "Wrong res tensor rank != 2 or 3");
    //     static_assert((env_type::NumIndices == 2 or env_type::NumIndices == 3) and "Wrong env tensor rank != 2 or 3");
    //     static_assert((mpo_type::NumIndices == 2 or mpo_type::NumIndices == 4) and "Wrong mpo tensor rank != 2 or 4");
    //     static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
    //     /* clang-format off */
    //     auto &res_ref = static_cast<res_type &>(res);
    //     auto env_eval = tenx::asEval(env);
    //     auto mps_eval = tenx::asEval(mps);
    //     auto mpo_eval = tenx::asEval(mpo);
    //     if constexpr(env_type::NumIndices == 2){
    //         res_ref.resize(mps_eval.dimension(2), mps_eval.dimension(2));
    //         contract_envL_mps_mpo(res_ref.data(), res_ref.dimensions(),
    //                              env_eval.data(), env_eval.dimensions(),
    //                              mps_eval.data(), mps_eval.dimensions(),
    //                              mpo_eval.data(), mpo_eval.dimensions());
    //     }
    //     else {
    //         res_ref.resize(mps_eval.dimension(2), mps_eval.dimension(2), mpo_eval.dimension(1));
    //         contract_envL_mps_mpo(res_ref.data(), res_ref.dimensions(),
    //                              env_eval.data(), env_eval.dimensions(),
    //                              mps_eval.data(), mps_eval.dimensions(),
    //                              mpo_eval.data(), mpo_eval.dimensions());
    //     }
    /* clang-format on */
    // }
    // template<typename res_type, typename env_type, typename mps_type, typename mpo_type>
    // void contract_envR_mps_mpo(TensorWrite<res_type> &res, const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo)
    // {
    //     static_assert((res_type::NumIndices == 2 or res_type::NumIndices == 3) and "Wrong res tensor rank != 2 or 3");
    //     static_assert((env_type::NumIndices == 2 or env_type::NumIndices == 3) and "Wrong env tensor rank != 2 or 3");
    //     static_assert((mpo_type::NumIndices == 2 or mpo_type::NumIndices == 4) and "Wrong mpo tensor rank != 2 or 4");
    //     static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
    //     /* clang-format off */
    //     auto &res_ref = static_cast<res_type &>(res);
    //     auto env_eval = tenx::asEval(env);
    //     auto mps_eval = tenx::asEval(mps);
    //     auto mpo_eval = tenx::asEval(mpo);
    //
    //
    //     if constexpr(env_type::NumIndices == 2){
    //         res_ref.resize(mps_eval.dimension(1), mps_eval.dimension(1));
    //         contract_envR_mps_mpo(res_ref.data(), res_ref.dimensions(),
    //                              env_eval.data(), env_eval.dimensions(),
    //                              mps_eval.data(), mps_eval.dimensions(),
    //                              mpo_eval.data(), mpo_eval.dimensions());
    //     }
    //     else {
    //         res_ref.resize(mps_eval.dimension(1), mps_eval.dimension(1), mpo_eval.dimension(0));
    //         contract_envR_mps_mpo(res_ref.data(), res_ref.dimensions(),
    //                              env_eval.data(), env_eval.dimensions(),
    //                              mps_eval.data(), mps_eval.dimensions(),
    //                              mpo_eval.data(), mpo_eval.dimensions());
    //     }
    /* clang-format on */
    // }

    // template<typename env_type, typename mps_type, typename mpo_type>
    // [[nodiscard]] auto contract_envL_mps_mpo(const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo) {
    //     Eigen::Tensor<typename env_type::Scalar, env_type::NumIndices> res;
    //     contract_envL_mps_mpo(res, env, mps, mpo);
    //     return res;
    // }
    // template<typename env_type, typename mps_type, typename mpo_type>
    // [[nodiscard]] auto contract_envR_mps_mpo(const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo) {
    //     Eigen::Tensor<typename env_type::Scalar, env_type::NumIndices> res;
    //     contract_envR_mps_mpo(res, env, mps, mpo);
    //     return res;
    // }
}