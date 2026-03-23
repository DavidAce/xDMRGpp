#pragma once

#include <Eigen/Core>
// Eigen goes first
#include "math/tenx/eval.h"
#include <array>

template<typename Scalar> class MatrixLikeOperator;
template<typename Scalar> struct IterativeLinearSolverConfig;

namespace tools::common::contraction {
    template<typename T> using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    template<typename T> using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename T>
    using TensorWrite = Eigen::TensorBase<T, Eigen::WriteAccessors>;
    template<typename T>
    using TensorRead = Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>;

    /* clang-format off */



    template<typename Scalar>
    VectorType<Scalar> matrix_inverse_vector_product(MatrixLikeOperator<Scalar> &MatrixOp,     //
                                                     const Scalar *rhs_ptr,                    //
                                                     const IterativeLinearSolverConfig<Scalar> &cfg);


    template<typename Scalar>
    void contract_mps_bnd(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const bnd_ptr,
                          std::array<long, 1> bnd_dims);

    template<typename Scalar>
    void contract_bnd_mps(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const bnd_ptr, std::array<long, 1> bnd_dims, const Scalar *const mps_ptr,
                          std::array<long, 3> mps_dims);

    template<typename Scalar>
    void contract_mps_mps(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const mpsL_ptr, std::array<long, 3> mpsL_dims,
                          const Scalar *const mpsR_ptr, std::array<long, 3> mpsR_dims);



    template<typename res_type, typename mps_type, typename bnd_type>
    void contract_mps_bnd(TensorWrite<res_type> &res, const TensorRead<mps_type> &mps, const TensorRead<bnd_type> &bnd) {
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(bnd_type::NumIndices == 1 and "Wrong bnd tensor rank != 1");
        auto &res_ref  = static_cast<res_type &>(res);
        auto  mps_eval = tenx::asEval(mps);
        auto  bnd_eval = tenx::asEval(bnd);

        res_ref.resize(mps_eval.dimensions());
        contract_mps_bnd(res_ref.data(), res_ref.dimensions(), mps_eval.data(), mps_eval.dimensions(), bnd_eval.data(), bnd_eval.dimensions());
    }

    template<typename res_type, typename bnd_type, typename mps_type>
    void contract_bnd_mps(TensorWrite<res_type> &res, const TensorRead<bnd_type> &bnd, const TensorRead<mps_type> &mps) {
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(bnd_type::NumIndices == 1 and "Wrong bnd tensor rank != 1");
        auto &res_ref  = static_cast<res_type &>(res);
        auto  mps_eval = tenx::asEval(mps);
        auto  bnd_eval = tenx::asEval(bnd);
        res_ref.resize(mps_eval.dimensions());
        contract_bnd_mps(res_ref.data(), res_ref.dimensions(), bnd_eval.data(), bnd_eval.dimensions(), mps_eval.data(), mps_eval.dimensions());
    }

    template<typename mps_type, typename bnd_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 3> contract_mps_bnd(const TensorRead<mps_type> &mps, const TensorRead<bnd_type> &bnd) {
        Eigen::Tensor<typename mps_type::Scalar, 3> res;
        contract_mps_bnd(res, mps, bnd);
        return res;
    }

    template<typename bnd_type, typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 3> contract_bnd_mps(const TensorRead<bnd_type> &bnd, const TensorRead<mps_type> &mps) {
        Eigen::Tensor<typename mps_type::Scalar, 3> res;
        contract_bnd_mps(res, bnd, mps);
        return res;
    }

    template<typename mps_type>
    void contract_mps_mps(TensorWrite<mps_type> &res, const TensorRead<mps_type> &mpsL, const TensorRead<mps_type> &mpsR) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        auto               &res_ref   = static_cast<mps_type &>(res);
        auto                mpsL_eval = tenx::asEval(mpsL);
        auto                mpsR_eval = tenx::asEval(mpsR);
        long                d0        = mpsL_eval.dimension(0) * mpsR_eval.dimension(0);
        long                d1        = mpsL_eval.dimension(1);
        long                d2        = mpsR_eval.dimension(2);
        auto res_dims  = std::array<long, 3> {d0, d1, d2};
        res_ref.resize(res_dims);
        contract_mps_mps(res_ref.data(), res_ref.dimensions(), mpsL_eval.data(), mpsL_eval.dimensions(), mpsR_eval.data(), mpsR_eval.dimensions());
    }

    template<typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 3> contract_mps_mps(const TensorRead<mps_type> &mpsL, const TensorRead<mps_type> &mpsR) {
        Eigen::Tensor<typename mps_type::Scalar, 3> res;
        contract_mps_mps(res, mpsL, mpsR);
        return res;
    }

    template<typename Scalar>
    Scalar contract_mps_mps_overlap(const Scalar *const mps1_ptr, std::array<long, 3> mps1_dims, const Scalar *const mps2_ptr, std::array<long, 3> mps2_dims);

    template<typename mps_type1, typename mps_type2>
    auto contract_mps_overlap(const TensorRead<mps_type1> &mps1, const TensorRead<mps_type2> &mps2) {
        static_assert(mps_type1::NumIndices == 3 and "Wrong mps1 tensor rank != 3");
        static_assert(mps_type2::NumIndices == 3 and "Wrong mps2 tensor rank != 3");
        auto mps1_eval = tenx::asEval(mps1);
        auto mps2_eval = tenx::asEval(mps2);
        return contract_mps_mps_overlap(mps1_eval.data(), mps1_eval.dimensions(), mps2_eval.data(), mps2_eval.dimensions());
    }

    template<typename mps_type>
    auto contract_mps_norm(const TensorRead<mps_type> &mps) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        auto mps_eval = tenx::asEval(mps);
        return contract_mps_mps_overlap(mps_eval.data(), mps_eval.dimensions(), mps_eval.data(), mps_eval.dimensions());
    }

    template<typename Scalar>
    void contract_mps_mps_partial(Scalar *res_ptr, std::array<long, 2> res_dims,               //
                                  const Scalar *const mps1_ptr, std::array<long, 3> mps1_dims, //
                                  const Scalar *const mps2_ptr, std::array<long, 3> mps2_dims, //
                                  std::array<long, 2> idx);

    template<std::array<long, 2> idx, typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 2> contract_mps_mps_partial(const TensorRead<mps_type> &mps1, const TensorRead<mps_type> &mps2) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(idx == std::array{0l, 1l} or idx == std::array{0l, 2l} or idx == std::array{1l, 2l});
        auto                mps1_eval = tenx::asEval(mps1);
        auto                mps2_eval = tenx::asEval(mps2);
        std::array<long, 2> dims      = {};
        if constexpr(idx == std::array{0l, 1l})
            dims = {mps1_eval.dimension(2), mps2_eval.dimension(2)};
        else if constexpr(idx == std::array{0l, 2l})
            dims = {mps1_eval.dimension(1), mps2_eval.dimension(1)};
        else if constexpr(idx == std::array{1l, 2l})
            dims = {mps1_eval.dimension(0), mps2_eval.dimension(0)};

        Eigen::Tensor<typename mps_type::Scalar, 2> res(dims);
        contract_mps_mps_partial(res.data(), res.dimensions(), mps1_eval.data(), mps1_eval.dimensions(), mps2_eval.data(), mps2_eval.dimensions(), idx);
        return res;
    }

    template<std::array<long, 2> idx, typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 2> contract_mps_partial(const TensorRead<mps_type> &mps) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(idx == std::array{0l, 1l} or idx == std::array{0l, 2l} or idx == std::array{1l, 2l});
        auto                mps_eval = tenx::asEval(mps);
        std::array<long, 2> dims     = {};
        if constexpr(idx == std::array{0l, 1l})
            dims = {mps_eval.dimension(2), mps_eval.dimension(2)};
        else if constexpr(idx == std::array{0l, 2l})
            dims = {mps_eval.dimension(1), mps_eval.dimension(1)};
        else if constexpr(idx == std::array{1l, 2l})
            dims = {mps_eval.dimension(0), mps_eval.dimension(0)};
        Eigen::Tensor<typename mps_type::Scalar, 2> res(dims);
        contract_mps_mps_partial(res.data(), res.dimensions(), mps_eval.data(), mps_eval.dimensions(), mps_eval.data(), mps_eval.dimensions(), idx);
        return res;
    }


}
