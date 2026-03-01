#pragma once
#include "../contraction_policy.h"
#include "../contraction_tblis.h"
#include "../matrix_vector_product.h"
#include "math/tenx.h"
#include "math/x2/gemm.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tid/tid.h"
#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis.h>
    #include <tblis_config.h>
#endif

#include "tools/common/log.h"

namespace settings {
    inline constexpr bool debug_contraction = false;
}

namespace tools::common::contraction::internal {

    template<typename Scalar>
    struct StatsMv {
        using RealScalar              = decltype(std::real(std::declval<Scalar>()));
        bool       contract_left      = true;
        RealScalar mps_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar mpo_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar envL_norm          = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar envR_norm          = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST1                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST2                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST3                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar cancelation_factor = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar highprec_threshold = RealScalar{10} / std::sqrt(std::numeric_limits<RealScalar>::epsilon());
        template<typename T>
        StatsMv<Scalar> &operator=(const StatsMv<T> &stats_) {
            this->contract_left      = stats_.contract_left;
            this->mps_norm           = static_cast<RealScalar>(stats_.mps_norm);
            this->mpo_norm           = static_cast<RealScalar>(stats_.mpo_norm);
            this->envL_norm          = static_cast<RealScalar>(stats_.envL_norm);
            this->envR_norm          = static_cast<RealScalar>(stats_.envR_norm);
            this->ST1                = static_cast<RealScalar>(stats_.ST1);
            this->ST2                = static_cast<RealScalar>(stats_.ST2);
            this->ST3                = static_cast<RealScalar>(stats_.ST3);
            this->cancelation_factor = static_cast<RealScalar>(stats_.cancelation_factor);
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
    StatsMv<Scalar> contract_with_gemm_x2(Eigen::Tensor<Scalar, 3>       &res,  //
                                          const Eigen::Tensor<Scalar, 3> &mps,  //
                                          const Eigen::Tensor<Scalar, 4> &mpo,  //
                                          const x2::Tensor<Scalar, 3>    &envL, //
                                          const x2::Tensor<Scalar, 3>    &envR) {
        StatsMv<Scalar> info;
        info.contract_left = false;

        // Eigen::Tensor<Scalar, 4> mpo_shf  = mpo.shuffle(std::array{0, 3, 2, 1});
        // Eigen::Tensor<Scalar, 3> envL_shf = envL.shuffle(std::array{0, 2, 1});

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
            auto mps_mat   = x2::as_const_matrix(mps, {0, 1}, {2});
            auto envR_mat  = x2::as_const_matrix_x2<Scalar, 3>(envR, {0}, {1, 2});
            auto T1_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T1, {0, 1}, {2, 3});
            x2::gemm_x2(T1_mat_x2, mps_mat, envR_mat);
        }

        {
            auto mpo_mat   = x2::as_const_matrix(mpo, {0, 3}, {1, 2});
            auto T1_mat_x2 = x2::as_const_matrix_x2<Scalar, 4>(T1, {3, 0}, {1, 2});
            auto T2_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T2, {0, 1}, {2, 3});
            x2::gemm_x2(T2_mat_x2, mpo_mat, T1_mat_x2);
        }
        auto res_x2 = x2::Tensor<Scalar, 3>(wd, mL, mR);

        {
            auto envL_mat   = x2::as_const_matrix_x2<Scalar, 3>(envL, {0, 2}, {1});
            auto T2_mat_x2  = x2::as_const_matrix_x2<Scalar, 4>(T2, {1, 3}, {2, 0});
            auto res_mat_x2 = x2::as_matrix_x2<Scalar, 3>(res_x2, {0, 2}, {1});
            x2::gemm_x2(res_mat_x2, T2_mat_x2, envL_mat);
        }

        res = res_x2.to_EigenTensor();

        info.mps_norm           = get_norm(mps.data(), mps.dimensions());
        info.mpo_norm           = get_norm(mpo.data(), mpo.dimensions());
        info.envL_norm          = envL.norm();
        info.envR_norm          = envR.norm();
        info.ST1                = T1.norm();
        info.ST2                = T2.norm();
        info.ST3                = get_norm(res.data(), res.dimensions());
        auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.envL_norm, info.envR_norm, info.ST1, info.ST2});
        info.cancelation_factor = Smax / info.ST3;
        if constexpr(settings::debug_contraction)
            tools::log->info("norms: mps {:.4e} mpo {:.4e} envL {:.4e} envR {:.4e} ST1 {:.4e} ST2 {:.4e} ST3 {:.4e} cf: {:.4e}", fp(info.mps_norm),
                             fp(info.mpo_norm), fp(info.envL_norm), fp(info.envR_norm), fp(info.ST1), fp(info.ST2), fp(info.ST3), fp(info.cancelation_factor));
        return info;
    }

    template<typename Scalar>
    StatsMv<Scalar> contract_with_eigen(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
        assert(mps.dimension(1) == envL.dimension(0));
        assert(mps.dimension(2) == envR.dimension(0));
        assert(mps.dimension(0) == mpo.dimension(2));
        assert(envL.dimension(2) == mpo.dimension(0));
        assert(envR.dimension(2) == mpo.dimension(1));
        StatsMv<Scalar> info;
        info.contract_left = mps.dimension(1) >= mps.dimension(2);

        auto                                 &threads = tenx::threads::get();
        thread_local Eigen::Tensor<Scalar, 4> T1;
        thread_local Eigen::Tensor<Scalar, 4> T2;

        if(info.contract_left) {
            T1.resize(mps.dimension(0), mps.dimension(2), envL.dimension(1), envL.dimension(2));
            T2.resize(mps.dimension(2), envL.dimension(1), mpo.dimension(1), mpo.dimension(3));

            T1.device(*threads->dev)  = mps.contract(envL, tenx::idx({1}, {0}));
            T2.device(*threads->dev)  = T1.contract(mpo, tenx::idx({3, 0}, {0, 2}));
            res.device(*threads->dev) = T2.contract(envR, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 0, 2});

        } else {
            T1.resize(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
            T2.resize(mps.dimension(1), envR.dimension(1), mpo.dimension(0), mpo.dimension(3));

            T1.device(*threads->dev)  = mps.contract(envR, tenx::idx({2}, {0}));
            T2.device(*threads->dev)  = T1.contract(mpo, tenx::idx({3, 0}, {1, 2}));
            res.device(*threads->dev) = T2.contract(envL, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 2, 0});
        }
        info.mps_norm           = get_norm(mps.data(), mps.dimensions());
        info.mpo_norm           = get_norm(mpo.data(), mpo.dimensions());
        info.envL_norm          = get_norm(envL.data(), envL.dimensions());
        info.envR_norm          = get_norm(envR.data(), envR.dimensions());
        info.ST1                = get_norm(T1.data(), T1.dimensions());
        info.ST2                = get_norm(T2.data(), T2.dimensions());
        info.ST3                = get_norm(res.data(), res.dimensions());
        auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.envL_norm, info.envR_norm, info.ST1, info.ST2});
        info.cancelation_factor = Smax / info.ST3;
        tools::log->info("Contracted with eigen: cf: {:.3e} | type {}", fp(info.cancelation_factor), sfinae::type_name<Scalar>());
        return info;
    }

    template<typename Scalar>
    StatsMv<Scalar> contract_with_tblis(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
        static_assert(settings::tblis_enabled);
        if constexpr(settings::tblis_enabled) {
            assert(mps.dimension(1) == envL.dimension(0));
            assert(mps.dimension(2) == envR.dimension(0));
            assert(mps.dimension(0) == mpo.dimension(2));
            assert(envL.dimension(2) == mpo.dimension(0));
            assert(envR.dimension(2) == mpo.dimension(1));
            static const tblis::tblis_config *tblis_cfg = nullptr; // tblis::config(get_tblis_arch().data());
            if constexpr(settings::tblis_use_openmp) { tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads())); }
            auto contract_tblis_wrapper = [](const auto &A, const auto &B, auto &C, std::string_view la, std::string_view lb, std::string_view lc,
                                             const tblis::tblis_config *cfg) {
                contract_tblis(A.data(), A.dimensions(), B.data(), B.dimensions(), C.data(), C.dimensions(), la, lb, lc, cfg);
            };

            StatsMv<Scalar> info;
            info.contract_left = mps.dimension(1) >= mps.dimension(2);

            thread_local Eigen::Tensor<Scalar, 4> T1;
            thread_local Eigen::Tensor<Scalar, 4> T2;
            if(mps.dimension(1) >= mps.dimension(2)) {
                T1.resize(mps.dimension(0), mps.dimension(2), envL.dimension(1), envL.dimension(2));
                T2.resize(mpo.dimension(1), mpo.dimension(3), mps.dimension(2), envL.dimension(1));
                contract_tblis_wrapper(mps, envL, T1, "afb", "fcd", "abcd", tblis_cfg);
                contract_tblis_wrapper(mpo, T1, T2, "qhri", "rgjq", "higj", tblis_cfg);
                contract_tblis_wrapper(T2, envR, res, "higj", "gkh", "ijk", tblis_cfg);
            } else {
                T1.resize(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
                T2.resize(mps.dimension(1), envR.dimension(1), mpo.dimension(0), mpo.dimension(3));
                contract_tblis_wrapper(mps, envR, T1, "abf", "fcd", "abcd", tblis_cfg);
                contract_tblis_wrapper(T1, mpo, T2, "qijk", "rkql", "ijrl", tblis_cfg);
                contract_tblis_wrapper(T2, envL, res, "qkri", "qjr", "ijk", tblis_cfg);
            }

            info.mps_norm           = get_norm(mps.data(), mps.dimensions());
            info.mpo_norm           = get_norm(mpo.data(), mpo.dimensions());
            info.envL_norm          = get_norm(envL.data(), envL.dimensions());
            info.envR_norm          = get_norm(envR.data(), envR.dimensions());
            info.ST1                = get_norm(T1.data(), T1.dimensions());
            info.ST2                = get_norm(T2.data(), T2.dimensions());
            info.ST3                = get_norm(res.data(), res.dimensions());
            auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.envL_norm, info.envR_norm, info.ST1, info.ST2});
            info.cancelation_factor = Smax / info.ST3;
            return info;
        };
    }

    template<typename Scalar>
    StatsMv<Scalar> contract_with_gemm_x2(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
        StatsMv<Scalar> info;
        info.contract_left = false;

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
            auto mps_mat   = x2::as_const_matrix(mps, {0, 1}, {2});
            auto envR_mat  = x2::as_const_matrix(envR, {0}, {1, 2});
            auto T1_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T1, {0, 1}, {2, 3});
            x2::gemm_x2(T1_mat_x2, mps_mat, envR_mat);
        }

        {
            auto mpo_mat   = x2::as_const_matrix(mpo, {0, 3}, {1, 2});
            auto T1_mat_x2 = x2::as_const_matrix_x2<Scalar, 4>(T1, {3, 0}, {1, 2});
            auto T2_mat_x2 = x2::as_matrix_x2<Scalar, 4>(T2, {0, 1}, {2, 3});
            x2::gemm_x2(T2_mat_x2, mpo_mat, T1_mat_x2);
        }
        auto res_x2 = x2::Tensor<Scalar, 3>(wd, mL, mR);

        {
            auto envL_mat   = x2::as_const_matrix(envL, {0, 2}, {1});
            auto T2_mat_x2  = x2::as_const_matrix_x2<Scalar, 4>(T2, {1, 3}, {2, 0});
            auto res_mat_x2 = x2::as_matrix_x2<Scalar, 3>(res_x2, {0, 2}, {1});
            x2::gemm_x2(res_mat_x2, T2_mat_x2, envL_mat);
        }

        res = res_x2.to_EigenTensor();

        info.mps_norm           = get_norm(mps.data(), mps.dimensions());
        info.mpo_norm           = get_norm(mpo.data(), mpo.dimensions());
        info.envL_norm          = get_norm(envL.data(), envL.dimensions());
        info.envR_norm          = get_norm(envR.data(), envR.dimensions());
        info.ST1                = T1.norm();
        info.ST2                = T2.norm();
        info.ST3                = get_norm(res.data(), res.dimensions());
        auto Smax               = std::max({info.mps_norm, info.mpo_norm, info.envL_norm, info.envR_norm, info.ST1, info.ST2});
        info.cancelation_factor = Smax / info.ST3;
        if constexpr(settings::debug_contraction)
            tools::log->info("norms: mps {:.4e} mpo {:.4e} envL {:.4e} envR {:.4e} ST1 {:.4e} ST2 {:.4e} ST3 {:.4e} cf: {:.4e}", fp(info.mps_norm),
                             fp(info.mpo_norm), fp(info.envL_norm), fp(info.envR_norm), fp(info.ST1), fp(info.ST2), fp(info.ST3), fp(info.cancelation_factor));
        return info;
    }
}

template<typename Scalar>
void tools::common::contraction::matrix_vector_product(Scalar             *res_ptr,                                 //
                                                       const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                       const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                       const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                       const Scalar *const envR_ptr, std::array<long, 3> envR_dims) {
    // This applies the mpo's with corresponding environments to local multisite mps
    // This is usually the operation H|psi>  or H²|psi>

    assert(mps_dims[1] == envL_dims[0]);
    assert(mps_dims[2] == envR_dims[0]);
    assert(mps_dims[0] == mpo_dims[2]);
    assert(envL_dims[2] == mpo_dims[0]);
    assert(envR_dims[2] == mpo_dims[1]);
    assert(mpo_dims[3] == mps_dims[0]);
    using RealScalar         = decltype(std::real(std::declval<Scalar>()));
    using VectorType         = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    constexpr bool use_tblis = settings::tblis_enabled and (std::is_same_v<RealScalar, fp32> or std::is_same_v<RealScalar, fp64>);

    auto res  = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, mps_dims);
    auto mps  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto mpo  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
    auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envL_ptr, envL_dims);
    auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envR_ptr, envR_dims);

    auto mpsv = Eigen::Map<const VectorType>(mps.data(), mps.size());
    auto resv = Eigen::Map<VectorType>(res.data(), res.size());

    internal::StatsMv<Scalar>    info;
    [[maybe_unused]] std::string msg;

    const internal::InfoH1Mv h1info         = internal::get_info_h1mv();
    const internal::InfoH2Mv h2info         = internal::get_info_h2mv();
    ContractionBackend       backend_active = ContractionBackend::AUTO;

    if(mpo_dims == h1info.H1_local_dims)
        backend_active = h1info.backend;
    else if(mpo_dims == h2info.H2_local_dims)
        backend_active = h2info.backend;

    switch(backend_active) {
        case ContractionBackend::X2: {
            info = internal::contract_with_gemm_x2<Scalar>(res, mps, mpo, envL, envR);
            // info = internal::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR);

            // TODO: compare with eigen
            break;
        }
        case ContractionBackend::EIGEN: info = internal::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR); break;
        case ContractionBackend::TBLIS: {
            if constexpr(use_tblis) {
                info = internal::contract_with_tblis<Scalar>(res, mps, mpo, envL, envR);
                break;
            } else {
                tools::log->debug("matrix_vector_product: Detected ContractionBackend::TBLIS, but use_tblis==false. Switching to Eigen.");
                info = internal::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR);
                break;
            }
        }
        case ContractionBackend::FP80: {
            using ScalarL = std::conditional_t<std::is_floating_point_v<Scalar>, long double, std::complex<long double>>;
            Eigen::Tensor<ScalarL, 3> res_fp(mps.dimensions());
            Eigen::Tensor<ScalarL, 3> mps_fp  = mps.template cast<ScalarL>();
            Eigen::Tensor<ScalarL, 4> mpo_fp  = mpo.template cast<ScalarL>();
            Eigen::Tensor<ScalarL, 3> envL_fp = envL.template cast<ScalarL>();
            Eigen::Tensor<ScalarL, 3> envR_fp = envR.template cast<ScalarL>();
            info                              = internal::contract_with_eigen<ScalarL>(res_fp, mps_fp, mpo_fp, envL_fp, envR_fp);
            res                               = res_fp.template cast<Scalar>();
            break;
        }
        case ContractionBackend::AUTO: {
            auto get_op_norm = [&]() -> RealScalar {
                using namespace internal;
                if(mpo_dims == h1info.H1_local_dims and std::isfinite(h1info.H1_local_norm)) return static_cast<RealScalar>(h1info.H1_local_norm);
                if(mpo_dims == h2info.H2_local_dims and std::isfinite(h2info.H2_local_norm)) return static_cast<RealScalar>(h2info.H2_local_norm);
                if constexpr(settings::debug_contraction) { tools::log->debug("matrix_vector_product: no registered norms for local H1 or H2: using envs"); }
                auto env_max_norm = std::max({get_norm(envL.data(), envL.dimensions()), get_norm(envR.data(), envR.dimensions())});
                return env_max_norm;
            };

            Eigen::Index md = mps_dims[0];
            Eigen::Index mL = mps_dims[1];
            Eigen::Index mR = mps_dims[2];
            Eigen::Index wL = mpo_dims[0];
            Eigen::Index wR = mpo_dims[1];

            const Eigen::Index cplx_factor = Eigen::NumTraits<Scalar>::IsComplex == 0 ? 1 : 4;
            const RealScalar   k_eff =
                static_cast<RealScalar>(cplx_factor * std::max({md * wL, md * wR, mL * wL, mR * wR})); // inner dimension of the dot products
            const RealScalar eps   = std::numeric_limits<RealScalar>::epsilon();
            const RealScalar gamma = (k_eff * eps) / (RealScalar(1) - k_eff * eps); // Slightly larger than eps
            const RealScalar q0Tol = RealScalar{1} / gamma;
            const RealScalar q2Tol = RealScalar{10} / gamma;
            const RealScalar q4Tol = RealScalar{1e-9f};

            const RealScalar xnorm2 = mpsv.squaredNorm();
            const RealScalar xnorm  = std::sqrt(xnorm2);
            const RealScalar opnorm = get_op_norm();
            const RealScalar q0     = opnorm * xnorm2;
            const bool       use_x2 = !std::isfinite(q0) or q0 > q0Tol;
            if(xnorm2 == RealScalar{0}) {
                resv.setZero();
                return;
            }
            if(use_x2) {
                info                    = internal::contract_with_gemm_x2<Scalar>(res, mps, mpo, envL, envR);
                const Scalar     xAx    = mpsv.dot(resv);
                const RealScalar denom  = std::abs(xAx);
                const RealScalar Axnorm = resv.norm();
                const RealScalar q2     = opnorm * xnorm2 / std::max(denom, eps);
                const RealScalar q4     = gamma * opnorm * xnorm / std::max(Axnorm, eps); // E.g. q4 > 1e-2: 2 digits lost

                if constexpr(settings::debug_contraction)
                    tools::log->debug("Switched matvec to x2:  opnorm={:.4e} xnorm={:.4e} q0={:.4e} q2={:.4e} q4={:.4e} q0Tol {:.4e} q2Tol {:.4e}", fp(opnorm),
                                      fp(xnorm), fp(q0), fp(q2), fp(q4), fp(q0Tol), fp(q2Tol));
            } else {
                if constexpr(use_tblis) {
                    info = internal::contract_with_tblis<Scalar>(res, mps, mpo, envL, envR);
                } else {
                    info = internal::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR);
                }

                // Decide redo: If |A|/(xAx/xx) is too large, there is catastrophic cancellation in the matvec.
                // Then we better switch to ContractionBackend::X2 (more precise) if the backend is AUTO.
                const Scalar     xAx    = mpsv.dot(resv);
                const RealScalar denom  = std::abs(xAx);
                const RealScalar Axnorm = resv.norm();
                const RealScalar q2     = opnorm * xnorm2 / std::max(denom, eps);
                const RealScalar q4     = gamma * opnorm * xnorm / std::max(Axnorm, eps);

                const bool q2_redo = !std::isfinite(q2) or q2 > q2Tol;
                const bool q4_redo = !std::isfinite(q4) or q4 > q4Tol;
                // tools::log->debug("matrix_vector_product:  opnorm={:.4e} xAx={:.4e} xnorm={:.4e} "
                // "q0={:.4e} q2={:.4e} q4={:.4e} q2Tol={:.4e} q4Tol={:.4e}",
                // fp(opnorm), fp(std::real(xAx)), fp(xnorm), fp(q0), fp(q2), fp(q4), fp(q2Tol), fp(q4Tol));
                if(q2_redo or q4_redo) {
                    VectorType resv_old = resv;
                    info                = internal::contract_with_gemm_x2<Scalar>(res, mps, mpo, envL, envR);

                    const Scalar     xAx_x2     = mpsv.dot(resv);
                    const RealScalar Axnorm_x2  = resv.norm();
                    const RealScalar xAx_diff   = std::abs(xAx_x2 - xAx);
                    const RealScalar Ax_diff    = (resv - resv_old).norm();
                    const RealScalar Ax_reldiff = (resv - resv_old).norm() / Axnorm_x2;
                    const RealScalar f          = Ax_reldiff / q4;
                    if constexpr(settings::debug_contraction)
                        tools::log->debug("matrix_vector_product: Redo matvec in x2:  opnorm={:.4e} xnorm={:.4e} Axnorm {:.4e} xAx={:.4e} -> {:.4e} "
                                          "q0={:.4e} q2={:.4e} q4={:.4e} q0Tol={:.4e} q2Tol={:.4e} q4Tol={:.4e} |xAx-xAx|={:.16e} "
                                          "|Ax-Ax_x2|={:.4e} |Ax-Ax_x2|/|Ax|={:.16e} f={:.4e}",
                                          fp(opnorm), fp(xnorm), fp(Axnorm), fp(std::real(xAx)), fp(std::real(xAx_x2)), fp(q0), fp(q2), fp(q4), fp(q0Tol),
                                          fp(q2Tol), fp(q4Tol), fp(xAx_diff), fp(Ax_diff), fp(Ax_reldiff), fp(f));
                }
            }
            break;
        }
        default: throw std::runtime_error("matrix_vector_product: Unknown ContractionBackend");
    }

    if constexpr(settings::debug_contraction)
        if(!msg.empty())
            tools::log->info("matrix_vector_product: ST1 {:.4e} ST2 {:.4e} ST3 {:.4e} cf: {:.4e} {}", fp(info.ST1), fp(info.ST2), fp(info.ST3),
                             fp(info.cancelation_factor), msg);
}

template<typename Scalar>
void tools::common::contraction::matrix_vector_product(Eigen::Tensor<Scalar, 3> &res, const Eigen::Tensor<Scalar, 3> &mps, //
                                                       const Eigen::Tensor<Scalar, 4> &mpo,                                //
                                                       const x2::Tensor<Scalar, 3>    &envL,                               //
                                                       const x2::Tensor<Scalar, 3>    &envR) {
    // This applies the mpo's with corresponding environments to local multisite mps
    // This is usually the operation H|psi>  or H²|psi>

    assert(mps.dimension(1) == envL.dimension(0));
    assert(mps.dimension(2) == envR.dimension(0));
    assert(mps.dimension(0) == mpo.dimension(2));
    assert(envL.dimension(2) == mpo.dimension(0));
    assert(envR.dimension(2) == mpo.dimension(1));
    assert(mpo.dimension(3) == mps.dimension(0));

    const internal::InfoH1Mv h1info         = internal::get_info_h1mv();
    const internal::InfoH2Mv h2info         = internal::get_info_h2mv();
    ContractionBackend       backend_active = ContractionBackend::AUTO;

    if(mpo.dimensions() == h1info.H1_local_dims)
        backend_active = h1info.backend;
    else if(mpo.dimensions() == h2info.H2_local_dims)
        backend_active = h2info.backend;

    if(backend_active == ContractionBackend::X2) {
        internal::StatsMv<Scalar>    info;
        [[maybe_unused]] std::string msg;
        info = internal::contract_with_gemm_x2<Scalar>(res, mps, mpo, envL, envR);
        if constexpr(settings::debug_contraction)
            if(!msg.empty())
                tools::log->info("matrix_vector_product: ST1 {:.4e} ST2 {:.4e} ST3 {:.4e} cf: {:.4e} {}", fp(info.ST1), fp(info.ST2), fp(info.ST3),
                                 fp(info.cancelation_factor), msg);
    } else {
        matrix_vector_product(res, mps, mpo, envL.to_EigenTensor(), envR.to_EigenTensor());
    }
}

template<typename Scalar>
requires(sfinae::is_any_v<typename Eigen::NumTraits<Scalar>::Real, fp32, fp64, fp128>)
void tools::common::contraction::matrix_vector_product(Eigen::Tensor<Scalar, 3>       &res,  //
                                                       const Eigen::Tensor<Scalar, 3> &mps,  //
                                                       const Eigen::Tensor<Scalar, 4> &mpo,  //
                                                       const EnvEne<Scalar>           &envL, //
                                                       const EnvEne<Scalar>           &envR) {
    const internal::InfoH1Mv h1info         = internal::get_info_h1mv();
    ContractionBackend       backend_active = ContractionBackend::AUTO;

    if(mpo.dimensions() == h1info.H1_local_dims) backend_active = h1info.backend;
    if(backend_active == ContractionBackend::X2) {
        matrix_vector_product(res, mps, mpo, envL.template get_blkx2_as<Scalar>(), envR.template get_blkx2_as<Scalar>());
    } else {
        matrix_vector_product(res, mps, mpo, envL.template get_block_as<Scalar>(), envR.template get_block_as<Scalar>());
    }
}

template<typename Scalar>
requires(sfinae::is_any_v<typename Eigen::NumTraits<Scalar>::Real, fp32, fp64, fp128>)
void tools::common::contraction::matrix_vector_product(Eigen::Tensor<Scalar, 3>       &res,  //
                                                       const Eigen::Tensor<Scalar, 3> &mps,  //
                                                       const Eigen::Tensor<Scalar, 4> &mpo,  //
                                                       const EnvVar<Scalar>           &envL, //
                                                       const EnvVar<Scalar>           &envR) {
    const internal::InfoH2Mv h2info         = internal::get_info_h2mv();
    ContractionBackend       backend_active = ContractionBackend::AUTO;

    if(mpo.dimensions() == h2info.H2_local_dims) backend_active = h2info.backend;
    if(backend_active == ContractionBackend::X2) {
        matrix_vector_product(res, mps, mpo, envL.template get_blkx2_as<Scalar>(), envR.template get_blkx2_as<Scalar>());
    } else {
        matrix_vector_product(res, mps, mpo, envL.template get_block_as<Scalar>(), envR.template get_block_as<Scalar>());
    }
}

template<typename Scalar, typename mpo_type>
void tools::common::contraction::matrix_vector_product(Scalar             *res_ptr,                                 //
                                                       const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                       const std::vector<mpo_type> &mpos_shf,                       //
                                                       const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                       const Scalar *const envR_ptr, std::array<long, 3> envR_dims) {
    // Make sure the mpos are pre-shuffled. If not, shuffle and call this function again
    bool is_shuffled = mpos_shf.front().dimension(2) == envL_dims[2] and mpos_shf.back().dimension(3) == envR_dims[2];
    if(not is_shuffled) {
        // mpos_shf are not actually shuffled. Let's shuffle.
        std::vector<Eigen::Tensor<Scalar, 4>> mpos_really_shuffled;
        for(const auto &mpo : mpos_shf) { mpos_really_shuffled.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1})); }
        return matrix_vector_product(res_ptr, mps_ptr, mps_dims, mpos_really_shuffled, envL_ptr, envL_dims, envR_ptr, envR_dims);
    }

    auto &threads = tenx::threads::get();
    auto  mps_out = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, mps_dims);
    auto  mps_in  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto  envL    = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envL_ptr, envL_dims);
    auto  envR    = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envR_ptr, envR_dims);

    assert(mps_in.dimension(1) == envL.dimension(0));
    assert(mps_in.dimension(2) == envR.dimension(0));

    auto L = mpos_shf.size();

    auto mpodimprod = [&](size_t fr, size_t to) -> long {
        long prod = 1;
        if(fr == -1ul) fr = 0;
        if(to == 0 or to == -1ul) return prod;
        for(size_t idx = fr; idx < to; ++idx) {
            if(idx >= mpos_shf.size()) break;
            prod *= mpos_shf[idx].dimension(1);
        }
        return prod;
    };

// At best, the number of operations for contracting left-to-right and right-to-left are equal.
// Since the site indices are contracted left to right, we do not need any shuffles in this direction.

// Contract left to right
#if defined(DMRG_ENABLE_TBLIS)
    static const tblis::tblis_config *tblis_cfg = nullptr; // tblis::tblis_get_config(get_tblis_arch().data());
    #if defined(TCI_USE_OPENMP_THREADS)
    tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads()));
    #endif
#endif
    auto d0       = mpodimprod(0, 1); // Split 0 --> 0,1
    auto d1       = mpodimprod(1, L); // Split 0 --> 0,1
    auto d2       = mps_in.dimension(2);
    auto d3       = envL.dimension(1);
    auto d4       = envL.dimension(2);
    auto d5       = 1l; // A new dummy index
    auto new_shp6 = tenx::array6{d0, d1, d2, d3, d4, d5};
    auto mps_tmp1 = Eigen::Tensor<Scalar, 6>();
    auto mps_tmp2 = Eigen::Tensor<Scalar, 6>();
    mps_tmp1.resize(tenx::array6{d0, d1, d2, d3, d5, d4});
#if defined(DMRG_ENABLE_TBLIS)
    if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>) {
        auto mps_tmp1_map4 = Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>(mps_tmp1.data(), std::array{d0 * d1, d2, d3, d4 * d5});
        // contract_tblis(mps_in, envL, mps_tmp1_map4, "afb", "fcd", "abcd", tblis_cfg);
        contract_tblis(mps_in.data(), mps_in.dimensions(),               //
                       envL.data(), envL.dimensions(),                   //
                       mps_tmp1_map4.data(), mps_tmp1_map4.dimensions(), //
                       "afb", "fcd", "abcd", tblis_cfg);
    } else
#endif
    {
        mps_tmp1.device(*threads->dev) = mps_in.contract(envL, tenx::idx({1}, {0})).reshape(new_shp6).shuffle(tenx::array6{0, 1, 2, 3, 5, 4});
    }
    for(size_t idx = 0; idx < L; ++idx) {
        const auto &mpo = mpos_shf[idx];
        // Set up the dimensions for the reshape after the contraction
        d0 = mpodimprod(idx + 1, idx + 2); // if idx == k, this has the mpo at idx == k+1
        d1 = mpodimprod(idx + 2, L);       // if idx == 0,  this has the mpos at idx == k+2...L-1
        d2 = mps_tmp1.dimension(2);
        d3 = mps_tmp1.dimension(3);
        d4 = mpodimprod(0, idx + 1); // if idx == 0, this has the mpos at idx == 0...k (i.e. including the one from the current iteration)
        d5 = mpo.dimension(3);       // The virtual bond of the current mpo
#if defined(DMRG_ENABLE_TBLIS)
        if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>) {
            auto md  = mps_tmp1.dimensions();
            new_shp6 = tenx::array6{d0, d1, d2, d3, d4, d5};
            mps_tmp2.resize(new_shp6);
            auto map_shp6     = tenx::array6{md[1], md[2], md[3], md[4], mpo.dimension(1), mpo.dimension(3)};
            auto mps_tmp2_map = Eigen::TensorMap<Eigen::Tensor<Scalar, 6>>(mps_tmp2.data(), map_shp6);
            // contract_tblis(mps_tmp1, mpo, mps_tmp2_map, "qbcder", "qfrg", "bcdefg", tblis_cfg);
            contract_tblis(mps_tmp1.data(), mps_tmp1.dimensions(),         //
                           mpo.data(), mpo.dimensions(),                   //
                           mps_tmp2_map.data(), mps_tmp2_map.dimensions(), //
                           "qbcder", "qfrg", "bcdefg", tblis_cfg);
            mps_tmp1 = std::move(mps_tmp2);
        } else
#endif
        {
            new_shp6 = tenx::array6{d0, d1, d2, d3, d4, d5};
            mps_tmp2.resize(new_shp6);
            mps_tmp2.device(*threads->dev) = mps_tmp1.contract(mpo, tenx::idx({0, 5}, {0, 2})).reshape(new_shp6);
            mps_tmp1                       = std::move(mps_tmp2);
        }
    }
    d0 = mps_tmp1.dimension(0) * mps_tmp1.dimension(1) * mps_tmp1.dimension(2); // idx 0 and 1 should have dim == 1
    d1 = mps_tmp1.dimension(3);
    d2 = mps_tmp1.dimension(4);
    d3 = mps_tmp1.dimension(5);
#if defined(DMRG_ENABLE_TBLIS)
    if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>) {
        auto mps_tmp1_map4 = Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>(mps_tmp1.data(), std::array{d0, d1, d2, d3});
        // contract_tblis(mps_tmp1_map4, envR, mps_out, "qjir", "qkr", "ijk", tblis_cfg);
        contract_tblis(mps_tmp1_map4.data(), mps_tmp1_map4.dimensions(), envR.data(), envR.dimensions(), mps_out.data(), mps_out.dimensions(), "qjir", "qkr",
                       "ijk", tblis_cfg);
    } else
#endif
    {
        mps_out.device(*threads->dev) = mps_tmp1.reshape(tenx::array4{d0, d1, d2, d3}).contract(envR, tenx::idx({0, 3}, {0, 2})).shuffle(tenx::array3{1, 0, 2});
    }
}

template<typename Scalar>
void tools::common::contraction::matrix_vector_product(Eigen::Tensor<Scalar, 3>                    &res,     //
                                                       const Eigen::Tensor<Scalar, 3>              &mps,     //
                                                       const std::vector<Eigen::Tensor<Scalar, 4>> &mpo_shf, //
                                                       const x2::Tensor<Scalar, 3>                 &envL,    //
                                                       const x2::Tensor<Scalar, 3>                 &envR) {
    auto envLt = envL.to_EigenTensor();
    auto envRt = envR.to_EigenTensor();
    matrix_vector_product(res.data(), mps.data(), mps.dimensions(), mpo_shf, envLt.data(), envLt.dimensions(), envRt.data(), envRt.dimensions());
}