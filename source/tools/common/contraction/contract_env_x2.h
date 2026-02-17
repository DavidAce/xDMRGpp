#pragma once
#include "../contraction.h"
#include "contraction_policy.h"
#include "general/sfinae.h"
#include "math/tenx.h"
#include "math/x2/gemm.h"
#include "tools/common/log.h"

namespace settings {
    inline constexpr bool debug_contract_env_x2 = false;
}
namespace tools::common::contraction::internal::env_x2 {
    template<typename Scalar>
    struct Info {
        using RealScalar              = decltype(std::real(std::declval<Scalar>()));
        RealScalar mps_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar mpo_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar env_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar res_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST1                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST2                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar cancelation_factor = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar highprec_threshold = RealScalar{10} / std::sqrt(std::numeric_limits<RealScalar>::epsilon());
        template<typename T>
        Info<Scalar> &operator=(const Info<T> &info_) {
            this->mps_norm           = static_cast<RealScalar>(info_.mps_norm);
            this->mpo_norm           = static_cast<RealScalar>(info_.mpo_norm);
            this->env_norm           = static_cast<RealScalar>(info_.env_norm);
            this->res_norm           = static_cast<RealScalar>(info_.res_norm);
            this->ST1                = static_cast<RealScalar>(info_.ST1);
            this->ST2                = static_cast<RealScalar>(info_.ST2);
            this->cancelation_factor = static_cast<RealScalar>(info_.cancelation_factor);
            return *this;
        }
    };

    auto get_size(const auto &dims) -> Eigen::Index {
        Eigen::Index size = 1;
        for(Eigen::Index i = 0; i < static_cast<Eigen::Index>(dims.size()); ++i) size *= dims[i];
        return size;
    }
    template<typename Scalar>
    auto get_norm(const Scalar *const ptr, const auto &dims) -> decltype(std::real(std::declval<Scalar>())) {
        return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(ptr, get_size(dims)).norm();
    }
    template<typename Scalar>
    Info<Scalar> contract_envL_with_gemm_x2(x2::Tensor<Scalar, 3> &res_x2, const x2::Tensor<Scalar, 3> &env_x2, const Eigen::Tensor<Scalar, 3> &mps,
                                            const Eigen::Tensor<Scalar, 4> &mpo) {
        assert(res_x2.dimension(0) == mps.dimension(2));
        assert(res_x2.dimension(1) == mps.dimension(2));
        assert(res_x2.dimension(2) == mpo.dimension(1));

        Eigen::Tensor<Scalar, 4> mpo_shf = mpo.shuffle(std::array{0, 2, 1, 3});
        Eigen::Tensor<Scalar, 3> mps_shf = mps.shuffle(std::array{2, 0, 1});

        Eigen::Index md = mps.dimension(0);
        Eigen::Index mL = mps.dimension(1);
        Eigen::Index mR = mps.dimension(2);
        Eigen::Index wL = mpo.dimension(0);
        Eigen::Index wR = mpo.dimension(1);
        Eigen::Index wd = mpo.dimension(3);

        thread_local x2::Tensor<Scalar, 4> T1;
        thread_local x2::Tensor<Scalar, 4> T2;

        T1.resize(std::array{mR, md, mL, wL});
        T2.resize(std::array{mR, mL, wR, wd});

        // Map the DD tensors to DD matrices

        {
            auto mps_shf_x2     = x2::Tensor<Scalar, 3>(mps_shf);
            auto mps_shf_mat_x2 = x2::ConstMatrixMap<Scalar>(mps_shf_x2.hi_data(), mps_shf_x2.lo_data(), md * mR, mL);

            auto env_mat_x2 = x2::ConstMatrixMap<Scalar>(env_x2.hi_data(), env_x2.lo_data(), mL, mL * wL);
            auto T1_mat_x2  = x2::MatrixMap<Scalar>(T1.hi_data(), T1.lo_data(), mR * md, mL * wL);
            gemm_x2(T1_mat_x2, mps_shf_mat_x2, env_mat_x2);
        }

        {
            T1.shuffle(std::array{0, 2, 3, 1});
            auto T1_mat_x2      = x2::ConstMatrixMap<Scalar>(T1.hi_data(), T1.lo_data(), mR * mL, wL * md);
            auto mpo_shf_x2     = x2::Tensor<Scalar, 4>(mpo_shf);
            auto mpo_shf_mat_x2 = x2::ConstMatrixMap<Scalar>(mpo_shf_x2.hi_data(), mpo_shf_x2.lo_data(), wL * md, wR * wd);
            auto T2_mat_x2      = x2::MatrixMap<Scalar>(T2.hi_data(), T2.lo_data(), mR * mL, wR * wd);

            gemm_x2(T2_mat_x2, T1_mat_x2, mpo_shf_mat_x2);
        }

        x2::Tensor<Scalar, 3> res_shf_x2(mR, mR, wR);

        {
            auto mps_shf_adj_x2     = x2::Tensor<Scalar, 3>(Eigen::Tensor<Scalar, 3>(mps_shf.conjugate()));
            auto mps_shf_adj_mat_x2 = x2::ConstMatrixMap<Scalar>(mps_shf_adj_x2.hi_data(), mps_shf_adj_x2.lo_data(), mR, wd * mL);
            auto res_shf_mat_x2     = x2::MatrixMap<Scalar>(res_shf_x2.hi_data(), res_shf_x2.lo_data(), mR, mR * wR);
            T2.shuffle(std::array{3, 1, 0, 2});
            auto T2_mat_x2 = x2::ConstMatrixMap<Scalar>(T2.hi_data(), T2.lo_data(), wd * mL, mR * wR);
            x2::gemm_x2(res_shf_mat_x2, mps_shf_adj_mat_x2, T2_mat_x2);
        }
        // final permutation back to tensor layout
        res_shf_x2.shuffle(std::array{1, 0, 2});
        res_x2 = res_shf_x2;

        Info<Scalar> info;
        if constexpr(settings::debug_contract_env_x2) {
            info.mps_norm = get_norm(mps.data(), mps.dimensions());
            info.mpo_norm = get_norm(mpo.data(), mpo.dimensions());
            info.env_norm = env_x2.norm();
            info.res_norm = res_x2.norm();

            info.ST1                = T1.norm();
            info.ST2                = T2.norm();
            auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.env_norm, info.ST1, info.ST2});
            info.cancelation_factor = Smax / info.res_norm;
            tools::log->debug("envL_x2 norms: mps {:.4e} mpo {:.4e} envL {:.4e} res {:.4e} ST1 {:.4e} ST2 {:.4e} cf: {:.4e}", fp(info.mps_norm),
                              fp(info.mpo_norm), fp(info.env_norm), fp(info.res_norm), fp(info.ST1), fp(info.ST2), fp(info.cancelation_factor));
        }
        return info;
    }

    template<typename Scalar>
    Info<Scalar> contract_envR_with_gemm_x2(x2::Tensor<Scalar, 3> &res_x2, const x2::Tensor<Scalar, 3> &env_x2, const Eigen::Tensor<Scalar, 3> &mps,
                                            const Eigen::Tensor<Scalar, 4> &mpo) {
        assert(res_x2.dimension(0) == mps.dimension(1));
        assert(res_x2.dimension(1) == mps.dimension(1));
        assert(res_x2.dimension(2) == mpo.dimension(0));

        Eigen::Tensor<Scalar, 4> mpo_shf = mpo.shuffle(std::array{0, 3, 2, 1});
        Eigen::Tensor<Scalar, 3> mps_shf = mps.conjugate().shuffle(std::array{1, 0, 2});

        Eigen::Index md = mps.dimension(0);
        Eigen::Index mL = mps.dimension(1);
        Eigen::Index mR = mps.dimension(2);
        Eigen::Index wL = mpo.dimension(0);
        Eigen::Index wR = mpo.dimension(1);
        Eigen::Index wd = mpo.dimension(3);

        thread_local x2::Tensor<Scalar, 4> T1;
        thread_local x2::Tensor<Scalar, 4> T2;

        T1.resize(std::array{md, mL, mR, wR});
        T2.resize(std::array{wL, wd, mL, mR});

        // Map the X2 tensors to X2 matrices

        {
            auto mps_x2     = x2::Tensor<Scalar, 3>(mps);
            auto T1_mat_x2  = x2::MatrixMap<Scalar>(T1.hi_data(), T1.lo_data(), md * mL, mR * wR);
            auto mps_mat_x2 = x2::ConstMatrixMap<Scalar>(mps_x2.hi_data(), mps_x2.lo_data(), md * mL, mR);
            auto env_mat_x2 = x2::ConstMatrixMap<Scalar>(env_x2.hi_data(), env_x2.lo_data(), mR, mR * wR);
            gemm_x2(T1_mat_x2, mps_mat_x2, env_mat_x2);
        }

        {
            auto mpo_shf_x2 = x2::Tensor<Scalar, 4>(mpo_shf);

            auto T2_mat_x2      = x2::MatrixMap<Scalar>(T2.hi_data(), T2.lo_data(), wL * wd, mL * mR);
            auto mpo_shf_mat_x2 = x2::ConstMatrixMap<Scalar>(mpo_shf_x2.hi_data(), mpo_shf_x2.lo_data(), wL * wd, md * wR);
            T1.shuffle(std::array{0, 3, 1, 2});
            auto T1_mat_x2 = x2::ConstMatrixMap<Scalar>(T1.hi_data(), T1.lo_data(), md * wR, mL * mR);
            gemm_x2(T2_mat_x2, mpo_shf_mat_x2, T1_mat_x2);
        }

        // Last contraction
        x2::Tensor<Scalar, 3> res_shf_x2(mL, mL, wL);
        {
            auto mps_shf_x2     = x2::Tensor<Scalar, 3>(mps_shf);
            auto res_shf_mat_x2 = x2::MatrixMap<Scalar>(res_shf_x2.hi_data(), res_shf_x2.lo_data(), mL, mL * wL);
            T2.shuffle(std::array{1, 3, 2, 0});
            auto T2_mat_x2      = x2::ConstMatrixMap<Scalar>(T2.hi_data(), T2.lo_data(), wd * mR, mL * wL);
            auto mps_shf_mat_x2 = x2::ConstMatrixMap<Scalar>(mps_shf_x2.hi_data(), mps_shf_x2.lo_data(), mL, md * mR);
            gemm_x2(res_shf_mat_x2, mps_shf_mat_x2, T2_mat_x2);
        }
        // final permutation back to tensor layout
        res_shf_x2.shuffle(std::array{1, 0, 2});
        res_x2 = res_shf_x2;

        Info<Scalar> info;
        if constexpr(settings::debug_contract_env_x2) {
            info.mps_norm = get_norm(mps.data(), mps.dimensions());
            info.mpo_norm = get_norm(mpo.data(), mpo.dimensions());
            info.env_norm = env_x2.norm();
            info.res_norm = res_x2.norm();

            info.ST1                = T1.norm();
            info.ST2                = T2.norm();
            auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.env_norm, info.ST1, info.ST2});
            info.cancelation_factor = Smax / info.res_norm;
            tools::log->info("envR_x2 norms: mps {:.4e} mpo {:.4e} envR {:.4e} res {:.4e} ST1 {:.4e} ST2 {:.4e} cf: {:.4e}", fp(info.mps_norm),
                             fp(info.mpo_norm), fp(info.env_norm), fp(info.res_norm), fp(info.ST1), fp(info.ST2), fp(info.cancelation_factor));
        }
        return info;
    }

}

namespace tools::common::contraction {
    template<typename Scalar>
    void contract_envL_mps_mpo(x2::Tensor<Scalar, 3> &res, const x2::Tensor<Scalar, 3> &env, //
                               const Eigen::Tensor<Scalar, 3> &mps,                          //
                               const Eigen::Tensor<Scalar, 4> &mpo) {
        auto envinfo = internal::get_info_env();
        if(envinfo.backend == ContractionBackend::X2) {
            res.resize(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
            internal::env_x2::contract_envL_with_gemm_x2<Scalar>(res, env, mps, mpo);
        } else {
            Eigen::Tensor<Scalar, 3> block(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
            tools::common::contraction::contract_envL_mps_mpo(block, env.to_TensorType(), mps, mpo);
            res = block;
            res.renorm();
        }
    }

    template<typename Scalar>
    void contract_envR_mps_mpo(x2::Tensor<Scalar, 3> &res, const x2::Tensor<Scalar, 3> &env, //
                               const Eigen::Tensor<Scalar, 3> &mps,                          //
                               const Eigen::Tensor<Scalar, 4> &mpo) {
        auto envinfo = internal::get_info_env();
        if(envinfo.backend == ContractionBackend::X2) {
            res.resize(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
            internal::env_x2::contract_envR_with_gemm_x2<Scalar>(res, env, mps, mpo);
        } else {
            Eigen::Tensor<Scalar, 3> block(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
            tools::common::contraction::contract_envR_mps_mpo(block, env.to_TensorType(), mps, mpo);
            res = block;
            res.renorm();
        }
    }
}