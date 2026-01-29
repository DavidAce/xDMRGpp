#pragma once
#include "../contraction.h"
#include "internal/gemm_x2.h"
#include "internal/util.h"
#include "math/float.h"
#include "math/linalg/tensor/to_string.h"
#include "math/tenx.h"
#include <cassert>
namespace settings {
    static constexpr bool debug_contraction = false;
}

#include "tools/common/log.h"

template<typename Scalar>
void tools::common::contraction::matrix_vector_product_gemm_x2(Scalar             *res_ptr,                                 //
                                                                  const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                  const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                  const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                  const Scalar *const envR_ptr, std::array<long, 3> envR_dims  //
) {
    // This applies the mpo's with corresponding environments to local multisite mps
    // This is usually the operation H|psi>  or H²|psi>
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    using namespace tools::common::contraction::internal;

    auto contract_with_gemm_x2 = [&](auto &res, const auto &mps, const auto &mpo, const auto &envL,
                                        const auto &envR) -> tools::common::contraction::internal::Info<Scalar> {
        Info<Scalar> info;
        info.contract_left = mps.dimension(1) >= mps.dimension(2);

        Eigen::Tensor<Scalar, 4> mpo_shf  = mpo.shuffle(std::array{0, 3, 2, 1});
        Eigen::Tensor<Scalar, 3> envL_shf = envL.shuffle(std::array{0, 2, 1});

        Eigen::Index md = mps.dimension(0);
        Eigen::Index mL = mps.dimension(1);
        Eigen::Index mR = mps.dimension(2);
        Eigen::Index wL = mpo.dimension(0);
        Eigen::Index wR = mpo.dimension(1);
        Eigen::Index wd = mpo.dimension(3);

        auto mps_x2      = TensorX2<Scalar, 3>(mps);
        auto mpo_shf_x2  = TensorX2<Scalar, 4>(mpo_shf);
        auto envL_shf_x2 = TensorX2<Scalar, 3>(envL_shf);
        auto envR_x2     = TensorX2<Scalar, 3>(envR);
        auto res_shf_x2  = TensorX2<Scalar, 3>(wd, mR, mL);

        thread_local TensorX2<Scalar, 4> T1;
        thread_local TensorX2<Scalar, 4> T2;

        T1.resize(std::array{md, mL, mR, wR});
        T2.resize(std::array{wL, wd, mL, mR});

        // Map the DD tensors to DD matrices
        auto mps_mat_x2  = ConstMatrixX2Map<Scalar>(mps_x2.hi.data(), mps_x2.lo.data(), md * mL, mR);
        auto envR_mat_x2 = ConstMatrixX2Map<Scalar>(envR_x2.hi.data(), envR_x2.lo.data(), mR, mR * wR);

        {
            auto T1_mat_x2 = MatrixX2Map<Scalar>(T1.hi.data(), T1.lo.data(), md * mL, mR * wR);
            gemm_x2(T1_mat_x2, mps_mat_x2, envR_mat_x2);
        }

        {
            auto T2_mat_x2      = MatrixX2Map<Scalar>(T2.hi.data(), T2.lo.data(), wL * wd, mL * mR);
            auto mpo_shf_mat_x2 = ConstMatrixX2Map<Scalar>(mpo_shf_x2.hi.data(), mpo_shf_x2.lo.data(), wL * wd, md * wR);
            T1.shuffle(std::array{0, 3, 1, 2});
            auto T1_mat_x2 = ConstMatrixX2Map<Scalar>(T1.hi.data(), T1.lo.data(), md * wR, mL * mR);

            gemm_x2(T2_mat_x2, mpo_shf_mat_x2, T1_mat_x2);
        }

        {
            auto res_shf_mat_x2 = MatrixX2Map<Scalar>(res_shf_x2.hi.data(), res_shf_x2.lo.data(), wd * mR, mL);
            T2.shuffle(std::array{1, 3, 2, 0});
            auto T2_mat_x2   = ConstMatrixX2Map<Scalar>(T2.hi.data(), T2.lo.data(), wd * mR, mL * wL);
            auto envL_mat_x2 = ConstMatrixX2Map<Scalar>(envL_shf_x2.hi.data(), envL_shf_x2.lo.data(), mL * wL, mL);
            gemm_x2(res_shf_mat_x2, T2_mat_x2, envL_mat_x2);
        }
        // final permutation back to tensor layout
        res_shf_x2.shuffle(std::array{0, 2, 1});
        res = res_shf_x2.to_TensorType();

        info.mps_norm           = mps_x2.norm();
        info.mpo_norm           = mpo_shf_x2.norm();
        info.envL_norm          = envL_shf_x2.norm();
        info.envR_norm          = envR_x2.norm();
        info.ST1                = T1.norm();
        info.ST2                = T2.norm();
        info.ST3                = res_shf_x2.norm();
        auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.envL_norm, info.envR_norm, info.ST1, info.ST2});
        info.cancelation_factor = Smax / info.ST3;
        return info;
    };

    assert(mps_dims[1] == envL_dims[0]);
    assert(mps_dims[2] == envR_dims[0]);
    assert(mps_dims[0] == mpo_dims[2]);
    assert(envL_dims[2] == mpo_dims[0]);
    assert(envR_dims[2] == mpo_dims[1]);
    auto res  = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, mps_dims);
    auto mps  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto mpo  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
    auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envL_ptr, envL_dims);
    auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envR_ptr, envR_dims);

    auto mps_norm  = internal::get_norm(mps_ptr, mps_dims);
    auto mpo_norm  = internal::get_norm(mpo_ptr, mpo_dims);
    auto envL_norm = internal::get_norm(envL_ptr, envL_dims);
    auto envR_norm = internal::get_norm(envR_ptr, envR_dims);

    internal::Info<Scalar>       info;
    [[maybe_unused]] std::string msg;
    info = contract_with_gemm_x2(res, mps, mpo, envL, envR);

    using namespace internal;
    if constexpr(settings::debug_contraction)
        if(!msg.empty())
            tools::log->info("res {:.4e} mps {:.4e} envL {:.4e} envR {:.4e} mpo {:.4e} ST1 {:.4e} ST2 {:.4e} ST3 {:.4e} cf: {:.4e} {}",
                             fp(get_norm(res_ptr, mps_dims)), fp(mps_norm), fp(envL_norm), fp(envR_norm), fp(mpo_norm), fp(info.ST1), fp(info.ST2),
                             fp(info.ST3), fp(info.cancelation_factor), msg);
}
