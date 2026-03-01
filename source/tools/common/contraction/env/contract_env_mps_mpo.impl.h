#pragma once
#include "../contraction_policy.h"
#include "../env.h"
#include "../internal/info.h"
#include "../internal/util.h"
#include "general/sfinae.h"
#include "math/tenx.h"
#include "math/x2/gemm.h"
#include "tools/common/log.h"
namespace settings {
    inline constexpr bool debug_contract_env    = false;
    inline constexpr bool debug_contract_env_x2 = false;

}

namespace tools::common::contraction::internal {
    template<typename Scalar>
    StatsEnv<Scalar> contract_env_with_eigen(auto &res, const auto &env, const auto &mps, const auto &mpo, const char side) {
        assert(side == 'L' or side == 'R');

        auto                    &threads = tenx::threads::get();
        Eigen::Tensor<Scalar, 4> T1;
        Eigen::Tensor<Scalar, 4> T2;

        auto idx_T1 = side == 'L' ? tenx::idx({1}, {1}) : tenx::idx({1}, {2});
        auto idx_T2 = side == 'L' ? tenx::idx({1, 2}, {0, 3}) : tenx::idx({1, 2}, {1, 3});
        auto idx_T3 = side == 'L' ? tenx::idx({0, 1}, {3, 0}) : tenx::idx({0, 2}, {3, 0});

        if(side == 'L') {
            assert(res.dimension(0) == mps.dimension(2));
            assert(res.dimension(1) == mps.dimension(2));
            assert(res.dimension(2) == mpo.dimension(1));
            T1.resize(env.dimension(0), env.dimension(2), mps.dimension(0), mps.dimension(2));
            T2.resize(env.dimension(0), mps.dimension(2), mpo.dimension(1), mpo.dimension(2));
        } else {
            assert(res.dimension(0) == mps.dimension(1));
            assert(res.dimension(1) == mps.dimension(1));
            assert(res.dimension(2) == mpo.dimension(0));
            T1.resize(env.dimension(0), env.dimension(2), mps.dimension(0), mps.dimension(1));
            T2.resize(env.dimension(0), mps.dimension(1), mpo.dimension(0), mpo.dimension(2));
        }

        T1.device(*threads->dev)  = env.contract(mps.conjugate(), idx_T1);
        T2.device(*threads->dev)  = T1.contract(mpo, idx_T2);
        res.device(*threads->dev) = mps.contract(T2, idx_T3);

        auto info = StatsEnv<Scalar>();
        if constexpr(settings::debug_contract_env) {
            info.mps_norm           = get_norm(mps.data(), mps.dimensions());
            info.mpo_norm           = get_norm(mpo.data(), mpo.dimensions());
            info.env_norm           = get_norm(env.data(), env.dimensions());
            info.res_norm           = get_norm(res.data(), res.dimensions());
            info.ST1                = get_norm(T1.data(), T1.dimensions());
            info.ST2                = get_norm(T2.data(), T2.dimensions());
            auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.env_norm, info.ST1, info.ST2});
            info.cancelation_factor = Smax / info.res_norm;
            tools::log->info("Contracted env{} with eigen: cf: {:.3e} | type {}", side, fp(info.cancelation_factor), sfinae::type_name<Scalar>());
        }
        return info;
    }

    template<typename Scalar>
    StatsEnv<Scalar> contract_envL_with_gemm_x2(auto &res, const auto &env, const auto &mps, const auto &mpo) {
        assert(res.dimension(0) == mps.dimension(2));
        assert(res.dimension(1) == mps.dimension(2));
        assert(res.dimension(2) == mpo.dimension(1));

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

        {
            auto mps_mat    = x2::as_const_matrix(mps, {2, 0}, {1});
            auto env_mat_x2 = x2::as_const_matrix(env, {0}, {1, 2});
            auto T1_mat_x2  = x2::as_matrix_x2<Scalar, 4>(T1, {0, 1}, {2, 3});
            x2::gemm_x2(T1_mat_x2, mps_mat, env_mat_x2);
        }

        {
            auto mpo_mat   = x2::as_const_matrix(mpo, {0, 2}, {1, 3});
            auto T1_mat_x2 = x2::as_const_matrix_x2<Scalar, 4>(T1, {0, 2}, {3, 1});
            auto T2_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T2, {0, 1}, {2, 3});
            x2::gemm_x2(T2_mat_x2, T1_mat_x2, mpo_mat);
        }

        {
            auto mps_adj     = Eigen::Tensor<Scalar, 3>(mps.conjugate());
            auto mps_adj_mat = x2::as_const_matrix(mps_adj, {2}, {0, 1});
            auto T2_mat_x2   = x2::as_const_matrix_x2<Scalar, 4>(T2, {3, 1}, {0, 2});
            auto res_x2      = x2::Tensor<Scalar, 3>(res.dimensions());
            auto res_mat_x2  = x2::as_matrix_x2<Scalar, 3>(res_x2, {1}, {0, 2});
            x2::gemm_x2(res_mat_x2, mps_adj_mat, T2_mat_x2);
            res = res_x2.to_EigenTensor();
        }

        StatsEnv<Scalar> info;
        if constexpr(settings::debug_contract_env) {
            info.mps_norm = get_norm(mps.data(), mps.dimensions());
            info.mpo_norm = get_norm(mpo.data(), mpo.dimensions());
            info.env_norm = get_norm(env.data(), env.dimensions());
            info.res_norm = get_norm(res.data(), res.dimensions());

            info.ST1                = T1.norm();
            info.ST2                = T2.norm();
            auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.env_norm, info.ST1, info.ST2});
            info.cancelation_factor = Smax / info.res_norm;
            // if constexpr(settings::debug_contract_env)
            tools::log->info("norms: mps {:.4e} mpo {:.4e} envL {:.4e} res {:.4e} ST1 {:.4e} ST2 {:.4e} cf: {:.4e}", fp(info.mps_norm), fp(info.mpo_norm),
                             fp(info.env_norm), fp(info.res_norm), fp(info.ST1), fp(info.ST2), fp(info.cancelation_factor));
        }
        return info;
    }

    template<typename Scalar>
    StatsEnv<Scalar> contract_envR_with_gemm_x2(auto &res, const auto &env, const auto &mps, const auto &mpo) {
        assert(res.dimension(0) == mps.dimension(1));
        assert(res.dimension(1) == mps.dimension(1));
        assert(res.dimension(2) == mpo.dimension(0));

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

        {
            auto mps_mat    = x2::as_const_matrix(mps, {0, 1}, {2});
            auto env_mat_x2 = x2::as_const_matrix(env, {0}, {1, 2});
            auto T1_mat_x2  = x2::as_matrix_x2<Scalar, 4>(T1, {0, 1}, {2, 3});
            gemm_x2(T1_mat_x2, mps_mat, env_mat_x2);
        }

        {
            auto mpo_mat   = x2::as_const_matrix(mpo, {0, 3}, {1, 2});
            auto T1_mat_x2 = x2::as_const_matrix_x2<Scalar, 4>(T1, {3, 0}, {1, 2});
            auto T2_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T2, {0, 1}, {2, 3});
            gemm_x2(T2_mat_x2, mpo_mat, T1_mat_x2);
        }

        {
            auto mps_adj     = Eigen::Tensor<Scalar, 3>(mps.conjugate());
            auto mps_adj_mat = x2::as_const_matrix(mps_adj, {1}, {0, 2});
            auto T2_mat_x2   = x2::as_const_matrix_x2<Scalar, 4>(T2, {1, 3}, {2, 0});
            auto res_x2      = x2::Tensor<Scalar, 3>(res.dimensions());
            auto res_mat_x2  = x2::as_matrix_x2<Scalar, 3>(res_x2, {1}, {0, 2});
            gemm_x2(res_mat_x2, mps_adj_mat, T2_mat_x2);
            res = res_x2.to_EigenTensor();
        }
        StatsEnv<Scalar> info;
        if constexpr(settings::debug_contract_env) {
            info.mps_norm = get_norm(mps.data(), mps.dimensions());
            info.mpo_norm = get_norm(mpo.data(), mpo.dimensions());
            info.env_norm = get_norm(env.data(), env.dimensions());
            info.res_norm = get_norm(res.data(), res.dimensions());

            info.ST1                = T1.norm();
            info.ST2                = T2.norm();
            auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.env_norm, info.ST1, info.ST2});
            info.cancelation_factor = Smax / info.res_norm;
            tools::log->info("norms: mps {:.4e} mpo {:.4e} envR {:.4e} res {:.4e} ST1 {:.4e} ST2 {:.4e} cf: {:.4e}", fp(info.mps_norm), fp(info.mpo_norm),
                             fp(info.env_norm), fp(info.res_norm), fp(info.ST1), fp(info.ST2), fp(info.cancelation_factor));
        }
        return info;
    }

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
        tools::log->trace("contract_envL_with_gemm_x2");

        // Eigen::Tensor<Scalar, 4> mpo_shf = mpo.shuffle(std::array{0, 2, 1, 3});
        // Eigen::Tensor<Scalar, 3> mps_shf = mps.shuffle(std::array{2, 0, 1});

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

        {
            auto mps_mat    = x2::as_const_matrix(mps, {2, 0}, {1});
            auto env_mat_x2 = x2::as_const_matrix_x2<Scalar, 3>(env_x2, {0}, {1, 2});
            auto T1_mat_x2  = x2::as_matrix_x2<Scalar, 4>(T1, {0, 1}, {2, 3});
            x2::gemm_x2(T1_mat_x2, mps_mat, env_mat_x2);
        }

        {
            auto mpo_mat   = x2::as_const_matrix(mpo, {0, 2}, {1, 3});
            auto T1_mat_x2 = x2::as_const_matrix_x2<Scalar, 4>(T1, {0, 2}, {3, 1});
            auto T2_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T2, {0, 1}, {2, 3});
            x2::gemm_x2(T2_mat_x2, T1_mat_x2, mpo_mat);
        }

        {
            auto mps_adj     = Eigen::Tensor<Scalar, 3>(mps.conjugate());
            auto mps_adj_mat = x2::as_const_matrix(mps_adj, {2}, {0, 1});
            auto T2_mat_x2   = x2::as_const_matrix_x2<Scalar, 4>(T2, {3, 1}, {0, 2});
            auto res_mat_x2  = x2::as_matrix_x2<Scalar, 3>(res_x2, {1}, {0, 2});
            x2::gemm_x2(res_mat_x2, mps_adj_mat, T2_mat_x2);
        }

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
        tools::log->trace("contract_envR_with_gemm_x2");

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

        {
            auto mps_mat    = x2::as_const_matrix(mps, {0, 1}, {2});
            auto env_mat_x2 = x2::as_const_matrix_x2<Scalar, 3>(env_x2, {0}, {1, 2});
            auto T1_mat_x2  = x2::as_matrix_x2<Scalar, 4>(T1, {0, 1}, {2, 3});
            gemm_x2(T1_mat_x2, mps_mat, env_mat_x2);
        }

        {
            auto mpo_mat   = x2::as_const_matrix(mpo, {0, 3}, {1, 2});
            auto T1_mat_x2 = x2::as_const_matrix_x2<Scalar, 4>(T1, {3, 0}, {1, 2});
            auto T2_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T2, {0, 1}, {2, 3});
            gemm_x2(T2_mat_x2, mpo_mat, T1_mat_x2);
        }

        {
            auto mps_adj     = Eigen::Tensor<Scalar, 3>(mps.conjugate());
            auto mps_adj_mat = x2::as_const_matrix(mps_adj, {1}, {0, 2});
            auto T2_mat_x2   = x2::as_const_matrix_x2<Scalar, 4>(T2, {1, 3}, {2, 0});
            auto res_mat_x2  = x2::as_matrix_x2<Scalar, 3>(res_x2, {1}, {0, 2});
            gemm_x2(res_mat_x2, mps_adj_mat, T2_mat_x2);
        }

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

template<typename Scalar>
void tools::common::contraction::contract_envL_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                                                       const Eigen::Tensor<Scalar, 3> &env, //
                                                       const Eigen::Tensor<Scalar, 3> &mps, //
                                                       const Eigen::Tensor<Scalar, 2> &mpo) {
    auto mpo4 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo.data(), std::array<long, 4>{1, 1, mpo.dimension(0), mpo.dimension(1)});
    res.resize(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
    internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo4, 'L');
}
template<typename Scalar>
void tools::common::contraction::contract_envR_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                                                       const Eigen::Tensor<Scalar, 3> &env, //
                                                       const Eigen::Tensor<Scalar, 3> &mps, //
                                                       const Eigen::Tensor<Scalar, 2> &mpo) {
    auto mpo4 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo.data(), std::array<long, 4>{1, 1, mpo.dimension(0), mpo.dimension(1)});
    res.resize(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
    internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo4, 'R');
}

template<typename Scalar>
void tools::common::contraction::contract_envL_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                                                       const Eigen::Tensor<Scalar, 3> &env, //
                                                       const Eigen::Tensor<Scalar, 3> &mps, //
                                                       const Eigen::Tensor<Scalar, 4> &mpo) {
    res.resize(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
    internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'L');
}
template<typename Scalar>
void tools::common::contraction::contract_envR_mps_mpo(Eigen::Tensor<Scalar, 3>       &res, //
                                                       const Eigen::Tensor<Scalar, 3> &env, //
                                                       const Eigen::Tensor<Scalar, 3> &mps, //
                                                       const Eigen::Tensor<Scalar, 4> &mpo) {
    res.resize(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
    internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'R');
}

template<typename Scalar>
void tools::common::contraction::contract_envL_mps_mpo(x2::Tensor<Scalar, 3>          &res, //
                                                       const x2::Tensor<Scalar, 3>    &env, //
                                                       const Eigen::Tensor<Scalar, 3> &mps, //
                                                       const Eigen::Tensor<Scalar, 4> &mpo) {
    using Real   = Eigen::NumTraits<Scalar>::Real;
    auto envinfo = internal::get_info_env();
    if(envinfo.backend == ContractionBackend::X2) {
        res.resize(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
        internal::env_x2::contract_envL_with_gemm_x2<Scalar>(res, env, mps, mpo);
        if constexpr(settings::debug_contract_env_x2) {
            Eigen::Tensor<Scalar, 3> block(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
            internal::contract_env_with_eigen<Scalar>(block, env.to_EigenTensor(), mps, mpo, 'L');
            auto resv   = tenx::VectorMap(res.to_EigenTensor());
            auto blockv = tenx::VectorMap(block);
            auto err    = (resv - blockv).norm();
            auto rel    = err / resv.norm();
            if(rel > Real{1e-14f}) tools::log->warn("contract_envL_mps_mpo x2 err: {:.4e} rel: {:.4e}", fp(err), fp(rel));
        }
    } else {
        Eigen::Tensor<Scalar, 3> block(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
        internal::contract_env_with_eigen<Scalar>(block, env.to_EigenTensor(), mps, mpo, 'L');
        res = block;
        if constexpr(settings::debug_contract_env_x2) {
            x2::Tensor<Scalar, 3> blkx2(res.dimensions());
            internal::env_x2::contract_envL_with_gemm_x2<Scalar>(blkx2, env, mps, mpo);

            auto resv = tenx::VectorMap(res.to_EigenTensor());
            auto blkv = tenx::VectorMap(blkx2.to_EigenTensor());
            auto err  = (resv - blkv).norm();
            auto rel  = err / resv.norm();
            if(rel > Real{1e-14f}) tools::log->warn("contract_envL_mps_mpo eigen err: {:.4e} rel: {:.4e}", fp(err), fp(rel));
        }
    }
}

template<typename Scalar>
void tools::common::contraction::contract_envR_mps_mpo(x2::Tensor<Scalar, 3>          &res, //
                                                       const x2::Tensor<Scalar, 3>    &env, //
                                                       const Eigen::Tensor<Scalar, 3> &mps, //
                                                       const Eigen::Tensor<Scalar, 4> &mpo) {
    using Real   = Eigen::NumTraits<Scalar>::Real;
    auto envinfo = internal::get_info_env();
    if(envinfo.backend == ContractionBackend::X2) {
        res.resize(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
        internal::env_x2::contract_envR_with_gemm_x2<Scalar>(res, env, mps, mpo);
        if constexpr(settings::debug_contract_env_x2) {
            Eigen::Tensor<Scalar, 3> block(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
            internal::contract_env_with_eigen<Scalar>(block, env.to_EigenTensor(), mps, mpo, 'R');
            auto resv   = tenx::VectorMap(res.to_EigenTensor());
            auto blockv = tenx::VectorMap(block);
            auto err    = (resv - blockv).norm();
            auto rel    = err / resv.norm();
            if(rel > Real{1e-14f}) tools::log->warn("contract_envR_mps_mpo x2 err: {:.4e} rel: {:.4e}", fp(err), fp(rel));
        }
    } else {
        Eigen::Tensor<Scalar, 3> block(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
        internal::contract_env_with_eigen<Scalar>(block, env.to_EigenTensor(), mps, mpo, 'R');
        res = block;

        if constexpr(settings::debug_contract_env_x2) {
            x2::Tensor<Scalar, 3> blkx2(res.dimensions());
            internal::env_x2::contract_envR_with_gemm_x2<Scalar>(blkx2, env, mps, mpo);

            auto resv = tenx::VectorMap(res.to_EigenTensor());
            auto blkv = tenx::VectorMap(blkx2.to_EigenTensor());
            auto err  = (resv - blkv).norm();
            auto rel  = err / resv.norm();
            if(rel > Real{1e-14f}) tools::log->warn("contract_envR_mps_mpo eigen err: {:.4e} rel: {:.4e}", fp(err), fp(rel));
        }
    }
}

// template<typename Scalar>
// void tools::common::contraction::contract_envL_mps_mpo(Scalar *res_ptr, std::array<long, 2> res_dims,             //
//                                                        const Scalar *const env_ptr, std::array<long, 2> env_dims, //
//                                                        const Scalar *const mps_ptr, std::array<long, 3> mps_dims, //
//                                                        const Scalar *const mpo_ptr, std::array<long, 2> mpo_dims) {
//     auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(res_ptr, res_dims);
//     auto env = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(env_ptr, env_dims);
//     auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
//     auto mpo = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, std::array<long, 4>{1, 1, mpo_dims[0], mpo_dims[1]});
//     internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'L');
// }

// template<typename Scalar>
// void tools::common::contraction::contract_envL_mps_mpo(Scalar *res_ptr, std::array<long, 3> res_dims,             //
//                                                        const Scalar *const env_ptr, std::array<long, 3> env_dims, //
//                                                        const Scalar *const mps_ptr, std::array<long, 3> mps_dims, //
//                                                        const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims) {
//     auto                  res     = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, res_dims);
//     auto                  env     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(env_ptr, env_dims);
//     auto                  mps     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
//     auto                  mpo     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
//     auto                  envinfo = internal::get_info_env();
//     [[maybe_unused]] auto stat    = internal::StatsEnv<Scalar>();
//     if(envinfo.backend == ContractionBackend::X2) {
//         stat = internal::contract_envL_with_gemm_x2<Scalar>(res, env, mps, mpo);
//     } else if(envinfo.backend == ContractionBackend::FP80) {
//         using ScalarL                     = std::conditional_t<std::is_floating_point_v<Scalar>, long double, std::complex<long double>>;
//         Eigen::Tensor<ScalarL, 3> res_fp  = Eigen::Tensor<ScalarL, 3>(res.dimensions());
//         Eigen::Tensor<ScalarL, 3> mps_fp  = mps.template cast<ScalarL>();
//         Eigen::Tensor<ScalarL, 4> mpo_fp  = mpo.template cast<ScalarL>();
//         Eigen::Tensor<ScalarL, 3> env_fp  = env.template cast<ScalarL>();
//         [[maybe_unused]] auto     stat_fp = internal::contract_env_with_eigen<ScalarL>(res_fp, env_fp, mps_fp, mpo_fp, 'L');
//         res                               = res_fp.template cast<Scalar>();
//     } else {
//         stat = internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'L');
//     }
// }

// template<typename Scalar>
// void tools::common::contraction::contract_envR_mps_mpo(Scalar *res_ptr, std::array<long, 2> res_dims,             //
//                                                        const Scalar *const env_ptr, std::array<long, 2> env_dims, //
//                                                        const Scalar *const mps_ptr, std::array<long, 3> mps_dims, //
//                                                        const Scalar *const mpo_ptr, std::array<long, 2> mpo_dims) {
//     auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(res_ptr, res_dims);
//     auto env = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(env_ptr, env_dims);
//     auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
//     auto mpo = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, std::array<long, 4>{1, 1, mpo_dims[0], mpo_dims[1]});
//     internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'R');
// }

// template<typename Scalar>
// void tools::common::contraction::contract_envR_mps_mpo(Scalar *res_ptr, std::array<long, 3> res_dims,             //
//                                                        const Scalar *const env_ptr, std::array<long, 3> env_dims, //
//                                                        const Scalar *const mps_ptr, std::array<long, 3> mps_dims, //
//                                                        const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims) {
//     auto                  res     = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, res_dims);
//     auto                  env     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(env_ptr, env_dims);
//     auto                  mps     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
//     auto                  mpo     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
//     auto                  envinfo = internal::get_info_env();
//     [[maybe_unused]] auto stat    = internal::StatsEnv<Scalar>();
//     if(envinfo.backend == ContractionBackend::X2) {
//         stat = internal::contract_envR_with_gemm_x2<Scalar>(res, env, mps, mpo);
//     } else if(envinfo.backend == ContractionBackend::FP80) {
//         using ScalarL                     = std::conditional_t<std::is_floating_point_v<Scalar>, long double, std::complex<long double>>;
//         Eigen::Tensor<ScalarL, 3> res_fp  = Eigen::Tensor<ScalarL, 3>(res.dimensions());
//         Eigen::Tensor<ScalarL, 3> mps_fp  = mps.template cast<ScalarL>();
//         Eigen::Tensor<ScalarL, 4> mpo_fp  = mpo.template cast<ScalarL>();
//         Eigen::Tensor<ScalarL, 3> env_fp  = env.template cast<ScalarL>();
//         [[maybe_unused]] auto     info_fp = internal::contract_env_with_eigen<ScalarL>(res_fp, env_fp, mps_fp, mpo_fp, 'R');
//         res                               = res_fp.template cast<Scalar>();
//     } else {
//         stat = internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'R');
//     }
// }
