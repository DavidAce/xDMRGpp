#pragma once
#include "../contraction.h"
#include "contraction_policy.h"
#include "general/sfinae.h"
#include "internal/info.h"
#include "internal/util.h"
#include "math/tenx.h"
#include "math/x2/gemm.h"
#include "tools/common/log.h"
namespace settings {
    inline constexpr bool debug_contract_env = false;
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

        // Map the DD tensors to DD matrices

        {
            auto mps_shf_x2     = x2::Tensor<Scalar, 3>(mps_shf);
            auto mps_shf_mat_x2 = x2::ConstMatrixMap<Scalar>(mps_shf_x2.hi_data(), mps_shf_x2.lo_data(), md * mR, mL);

            auto env_x2     = x2::Tensor<Scalar, 3>(env);
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

        auto res_shf_x2 = x2::Tensor<Scalar, 3>(mR, mR, wR);

        {
            auto mps_shf_adj_x2     = x2::Tensor<Scalar, 3>(Eigen::Tensor<Scalar, 3>(mps_shf.conjugate()));
            auto mps_shf_adj_mat_x2 = x2::ConstMatrixMap<Scalar>(mps_shf_adj_x2.hi_data(), mps_shf_adj_x2.lo_data(), mR, wd * mL);
            auto res_shf_mat_x2     = x2::MatrixMap<Scalar>(res_shf_x2.hi_data(), res_shf_x2.lo_data(), mR, mR * wR);
            T2.shuffle(std::array{3, 1, 0, 2});
            auto T2_mat_x2 = x2::ConstMatrixMap<Scalar>(T2.hi_data(), T2.lo_data(), wd * mL, mR * wR);
            gemm_x2(res_shf_mat_x2, mps_shf_adj_mat_x2, T2_mat_x2);
        }
        // final permutation back to tensor layout
        res_shf_x2.shuffle(std::array{1, 0, 2});
        res = res_shf_x2.to_TensorType();

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

        // Map the DD tensors to DD matrices

        {
            auto mps_x2     = x2::Tensor<Scalar, 3>(mps);
            auto env_x2     = x2::Tensor<Scalar, 3>(env);
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
        auto res_shf_x2 = x2::Tensor<Scalar, 3>(mL, mL, wL);
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
        res = res_shf_x2.to_TensorType();

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

template<typename Scalar>
void tools::common::contraction::contract_envL_mps_mpo(Scalar *res_ptr, std::array<long, 2> res_dims, const Scalar *const env_ptr, std::array<long, 2> env_dims,
                                                       const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr,
                                                       std::array<long, 2> mpo_dims) {
    auto  res                 = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(res_ptr, res_dims);
    auto  env                 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(env_ptr, env_dims);
    auto  mps                 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto  mpo                 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(mpo_ptr, mpo_dims);
    auto &threads             = tenx::threads::get();
    res.device(*threads->dev) = env.contract(mps, tenx::idx({0}, {1})).contract(mpo, tenx::idx({1}, {0})).contract(mps.conjugate(), tenx::idx({0, 2}, {1, 0}));
}

template<typename Scalar>
void tools::common::contraction::contract_envL_mps_mpo(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const env_ptr, std::array<long, 3> env_dims,
                                                       const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr,
                                                       std::array<long, 4> mpo_dims) {
    auto                  res     = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, res_dims);
    auto                  env     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(env_ptr, env_dims);
    auto                  mps     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto                  mpo     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
    auto                  envinfo = internal::get_info_env();
    [[maybe_unused]] auto stat    = internal::StatsEnv<Scalar>();
    if(envinfo.backend == ContractionBackend::X2) {
        stat = internal::contract_envL_with_gemm_x2<Scalar>(res, env, mps, mpo);
    } else if(envinfo.backend == ContractionBackend::FP80) {
        using ScalarL                     = std::conditional_t<std::is_floating_point_v<Scalar>, long double, std::complex<long double>>;
        Eigen::Tensor<ScalarL, 3> res_fp  = Eigen::Tensor<ScalarL, 3>(res.dimensions());
        Eigen::Tensor<ScalarL, 3> mps_fp  = mps.template cast<ScalarL>();
        Eigen::Tensor<ScalarL, 4> mpo_fp  = mpo.template cast<ScalarL>();
        Eigen::Tensor<ScalarL, 3> env_fp  = env.template cast<ScalarL>();
        [[maybe_unused]] auto     stat_fp = internal::contract_env_with_eigen<ScalarL>(res_fp, env_fp, mps_fp, mpo_fp, 'L');
        res                               = res_fp.template cast<Scalar>();
    } else {
        stat = internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'L');
    }
}

template<typename Scalar>
void tools::common::contraction::contract_envR_mps_mpo(Scalar *res_ptr, std::array<long, 2> res_dims, const Scalar *const env_ptr, std::array<long, 2> env_dims,
                                                       const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr,
                                                       std::array<long, 2> mpo_dims) {
    auto  res                 = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(res_ptr, res_dims);
    auto  env                 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(env_ptr, env_dims);
    auto  mps                 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto  mpo                 = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(mpo_ptr, mpo_dims);
    auto &threads             = tenx::threads::get();
    res.device(*threads->dev) = env.contract(mps, tenx::idx({0}, {1})).contract(mpo, tenx::idx({1}, {0})).contract(mps.conjugate(), tenx::idx({0, 2}, {1, 0}));
}

template<typename Scalar>
void tools::common::contraction::contract_envR_mps_mpo(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const env_ptr, std::array<long, 3> env_dims,
                                                       const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr,
                                                       std::array<long, 4> mpo_dims) {
    auto                  res     = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, res_dims);
    auto                  env     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(env_ptr, env_dims);
    auto                  mps     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto                  mpo     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
    auto                  envinfo = internal::get_info_env();
    [[maybe_unused]] auto stat    = internal::StatsEnv<Scalar>();
    if(envinfo.backend == ContractionBackend::X2) {
        stat = internal::contract_envR_with_gemm_x2<Scalar>(res, env, mps, mpo);
    } else if(envinfo.backend == ContractionBackend::FP80) {
        using ScalarL                     = std::conditional_t<std::is_floating_point_v<Scalar>, long double, std::complex<long double>>;
        Eigen::Tensor<ScalarL, 3> res_fp  = Eigen::Tensor<ScalarL, 3>(res.dimensions());
        Eigen::Tensor<ScalarL, 3> mps_fp  = mps.template cast<ScalarL>();
        Eigen::Tensor<ScalarL, 4> mpo_fp  = mpo.template cast<ScalarL>();
        Eigen::Tensor<ScalarL, 3> env_fp  = env.template cast<ScalarL>();
        [[maybe_unused]] auto     info_fp = internal::contract_env_with_eigen<ScalarL>(res_fp, env_fp, mps_fp, mpo_fp, 'R');
        res                               = res_fp.template cast<Scalar>();
    } else {
        stat = internal::contract_env_with_eigen<Scalar>(res, env, mps, mpo, 'R');
    }
}
