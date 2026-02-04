#pragma once
#include "../contraction.h"
#include "internal/gemm_x2.h"
#include "math/tenx.h"
#include "matvec_policy.h"
#include "tid/tid.h"
#if defined(DMRG_ENABLE_TBLIS)
    // #include <tblis/util/configs.h>
    // #include <tblis/util/thread.h>
    // #include <tci/tci_config.h>
    // #if defined(TCI_USE_OPENMP_THREADS)
    // #include <omp.h>
    // #endif
    #include <tblis.h>
    #include <tblis_config.h>
#endif

using namespace tools::common::contraction;

namespace settings {
#if defined(DMRG_ENABLE_TBLIS)
    static constexpr bool tblis_enabled = true;
#else
    static constexpr bool tblis_enabled = false;
#endif

#if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
    static constexpr bool tblis_use_openmp = true;
#else
    static constexpr bool tblis_use_openmp = false;
#endif

    static constexpr bool debug_contraction = false;
}

#include "tools/common/log.h"

namespace tools::common::contraction::internal {
    template<typename Scalar>
    struct Info {
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
        Info<Scalar> &operator=(const Info<T> &info_) {
            this->contract_left      = info_.contract_left;
            this->mps_norm           = static_cast<RealScalar>(info_.mps_norm);
            this->mpo_norm           = static_cast<RealScalar>(info_.mpo_norm);
            this->envL_norm          = static_cast<RealScalar>(info_.envL_norm);
            this->envR_norm          = static_cast<RealScalar>(info_.envR_norm);
            this->ST1                = static_cast<RealScalar>(info_.ST1);
            this->ST2                = static_cast<RealScalar>(info_.ST2);
            this->ST3                = static_cast<RealScalar>(info_.ST3);
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
    Info<Scalar> contract_with_eigen(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
        assert(mps.dimension(1) == envL.dimension(0));
        assert(mps.dimension(2) == envR.dimension(0));
        assert(mps.dimension(0) == mpo.dimension(2));
        assert(envL.dimension(2) == mpo.dimension(0));
        assert(envR.dimension(2) == mpo.dimension(1));
        Info<Scalar> info;
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
    Info<Scalar> contract_with_tblis(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
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
                contraction::contract_tblis(A.data(), A.dimensions(), B.data(), B.dimensions(), C.data(), C.dimensions(), la, lb, lc, cfg);
            };

            Info<Scalar> info;
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
    Info<Scalar> contract_with_gemm_x2(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
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
        // if constexpr(settings::debug_contraction)
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
                                                       const Scalar *const envR_ptr, std::array<long, 3> envR_dims  //
) {
    // This applies the mpo's with corresponding environments to local multisite mps
    // This is usually the operation H|psi>  or H²|psi>

    assert(mps_dims[1] == envL_dims[0]);
    assert(mps_dims[2] == envR_dims[0]);
    assert(mps_dims[0] == mpo_dims[2]);
    assert(envL_dims[2] == mpo_dims[0]);
    assert(envR_dims[2] == mpo_dims[1]);

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

    internal::Info<Scalar>       info;
    [[maybe_unused]] std::string msg;

    const internal::MatVecOptions opts           = internal::matvec_options_active();
    const MatVecBackend           backend_active = opts.backend;

    switch(backend_active) {
        case MatVecBackend::X2: info = internal::contract_with_gemm_x2<Scalar>(res, mps, mpo, envL, envR); break;
        case MatVecBackend::EIGEN: info = internal::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR); break;
        case MatVecBackend::TBLIS: {
            if constexpr(use_tblis) {
                info = internal::contract_with_tblis<Scalar>(res, mps, mpo, envL, envR);
                break;
            } else {
                tools::log->debug("matrix_vector_product: Detected MatVecBackend::TBLIS, but use_tblis==false. Switching to Eigen.");
                info = internal::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR);
                break;
            }
        }
        case MatVecBackend::AUTO: {
            auto get_op_norm = [&]() -> RealScalar {
                using namespace internal;
                if(mpo_dims == opts.H1_dims) return static_cast<RealScalar>(opts.H1_norm);
                if(mpo_dims == opts.H2_dims) return static_cast<RealScalar>(opts.H2_norm);
                return std::max({get_norm(envL.data(), envL.dimensions()), get_norm(envR.data(), envR.dimensions())});
            };

            Eigen::Index md = mps_dims[0];
            Eigen::Index mL = mps_dims[1];
            Eigen::Index mR = mps_dims[2];
            Eigen::Index wL = mpo_dims[0];
            Eigen::Index wR = mpo_dims[1];

            const Eigen::Index cplx_factor = Eigen::NumTraits<Scalar>::IsComplex == 0 ? 1 : 4;
            const RealScalar   k_eff =
                static_cast<RealScalar>(cplx_factor * std::max({md * wL, md * wR, mL * wL, mR * wR})); // inner dimension of the dot products
            const RealScalar eps          = std::numeric_limits<RealScalar>::epsilon();
            const RealScalar gamma        = (k_eff * eps) / (RealScalar(1) - k_eff * eps);
            const RealScalar x2_redoTol   = RealScalar{10} / gamma;
            const RealScalar x2_switchTol = RealScalar{1} / gamma;
            const RealScalar xnorm2       = mpsv.squaredNorm();
            const RealScalar opnorm       = get_op_norm();
            const RealScalar crit_switch  = opnorm * xnorm2;
            const bool       use_x2       = !std::isfinite(crit_switch) or crit_switch > x2_switchTol;
            if(xnorm2 == RealScalar{0}) {
                resv.setZero();
                return;
            }
            if(use_x2) {
                tools::log->debug("Switching matvec to x2:  opnorm={:.4e} xnorm2={:.4e} criterion: {:.4e}  switchTol {:.4e}", fp(opnorm), fp(xnorm2),
                                  fp(opnorm * xnorm2), fp(x2_switchTol));
                info = internal::contract_with_gemm_x2<Scalar>(res, mps, mpo, envL, envR);
            } else {
                if constexpr(use_tblis) {
                    info = internal::contract_with_tblis<Scalar>(res, mps, mpo, envL, envR);
                } else {
                    info = internal::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR);
                }

                // Decide redo: If |A|/(xAx/xx) is too large, there is catastrophic cancellation in the matvec.
                // Then we better switch to MatVecBackend::X2 (more precise) if the backend is AUTO.
                const Scalar     xAx       = mpsv.dot(resv);
                const RealScalar denom     = std::abs(xAx);
                const RealScalar crit_redo = opnorm * xnorm2 / std::max(denom, eps);
                const bool       do_redo   = !std::isfinite(crit_redo) or crit_redo > x2_redoTol;

                if(do_redo) {
                    VectorType resv_old = resv;
                    info                = internal::contract_with_gemm_x2<Scalar>(res, mps, mpo, envL, envR);

                    const Scalar     xAx_x2   = mpsv.dot(resv);
                    const RealScalar xAx_diff = std::abs(xAx_x2 - xAx);
                    const RealScalar Ax_diff  = (resv - resv_old).norm();
                    tools::log->debug("matrix_vector_product: Redo matvec in x2:  opnorm={:.4e} xAx={:.4e} xnorm2={:.4e} criterion={:.4e}  redoTol={:.4e} "
                                      "|xAx-xAx|={:.16e} | |Ax-Ax|={:.16e}",
                                      fp(opnorm), fp(std::real(xAx)), fp(xnorm2), fp(crit_redo), fp(x2_redoTol), fp(xAx_diff), fp(Ax_diff));
                }
            }
            break;
        }
        default: throw std::runtime_error("matrix_vector_product: Unknown MatVecBackend");
    }

    if constexpr(settings::debug_contraction)
        if(!msg.empty())
            tools::log->info("matrix_vector_product: ST1 {:.4e} ST2 {:.4e} ST3 {:.4e} cf: {:.4e} {}", fp(info.ST1), fp(info.ST2), fp(info.ST3),
                             fp(info.cancelation_factor), msg);
}

template<typename Scalar, typename mpo_type>
void tools::common::contraction::matrix_vector_product(Scalar *res_ptr, const Scalar *const mps_ptr, std::array<long, 3> mps_dims,
                                                       const std::vector<mpo_type> &mpos_shf, const Scalar *const envL_ptr, std::array<long, 3> envL_dims,
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
