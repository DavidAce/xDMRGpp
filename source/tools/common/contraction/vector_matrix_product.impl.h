#pragma once
#include "../contraction.h"
#include "math/tenx.h"
#include "ScaledTensor.h"
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

namespace tools::common::contraction::internal_vmp {
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
            T1.resize(mps.dimension(0), mps.dimension(2), envL.dimension(0), envL.dimension(2));
            T2.resize(mps.dimension(2), envL.dimension(0), mpo.dimension(1), mpo.dimension(2));
            T1.device(*threads->dev)  = mps.contract(envL, tenx::idx({1}, {1}));
            T2.device(*threads->dev)  = T1.contract(mpo, tenx::idx({3, 0}, {0, 3}));
            res.device(*threads->dev) = T2.contract(envR, tenx::idx({0, 2}, {1, 2})).shuffle(tenx::array3{1, 0, 2});

        } else {
            T1.resize(mps.dimension(0), mps.dimension(1), envR.dimension(0), envR.dimension(2));
            T2.resize(mps.dimension(1), envR.dimension(0), mpo.dimension(0), mpo.dimension(2));
            T1.device(*threads->dev)  = mps.contract(envR, tenx::idx({2}, {1}));
            T2.device(*threads->dev)  = T1.contract(mpo, tenx::idx({3, 0}, {1, 3}));
            res.device(*threads->dev) = T2.contract(envL, tenx::idx({0, 2}, {1, 2})).shuffle(tenx::array3{1, 2, 0});
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
            assert(mps.dimension(1) == envL.dimension(1));
            assert(mps.dimension(2) == envR.dimension(1));
            assert(mps.dimension(0) == mpo.dimension(3));
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
                T1.resize(mps.dimension(0), mps.dimension(2), envL.dimension(0), envL.dimension(2));
                T2.resize(mpo.dimension(1), mpo.dimension(2), mps.dimension(2), envL.dimension(0));
                contract_tblis_wrapper(mps, envL, T1, "afb", "cfd", "abcd", tblis_cfg);
                contract_tblis_wrapper(mpo, T1, T2, "qhir", "rgjq", "higj", tblis_cfg);
                contract_tblis_wrapper(T2, envR, res, "higj", "kgh", "ijk", tblis_cfg);
            } else {
                T1.resize(mps.dimension(0), mps.dimension(1), envR.dimension(0), envR.dimension(2));
                T2.resize(mps.dimension(1), envR.dimension(0), mpo.dimension(0), mpo.dimension(2));
                contract_tblis_wrapper(mps, envR, T1, "abf", "cfd", "abcd", tblis_cfg);
                contract_tblis_wrapper(T1, mpo, T2, "qijk", "rklq", "ijrl", tblis_cfg);
                contract_tblis_wrapper(T2, envL, res, "qkri", "jqr", "ijk", tblis_cfg);
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
    Info<Scalar> contract_with_longsum(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        auto gemm_highprecision_fp80 = [](const Eigen::Ref<const MatrixType> &A_in, const Eigen::Ref<const MatrixType> &B_in, Eigen::Index BK) -> MatrixType {
            // Multiply in FP64, accumulate in long double, return FP64.
            // - If BK == 1: do scalar FMAs (double multiply) into long double accumulator.
            // - If BK  > 1: do GEMM in double for each k-block, then add that block result into long double accumulator.
            //
            // Requirements: A.cols() == B.rows().

            const Eigen::Index m = A_in.rows();
            const Eigen::Index k = A_in.cols();
            const Eigen::Index n = B_in.cols();

            assert(B_in.rows() == A_in.cols());
            assert(BK >= 1);
            using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;

            // long double accumulator (no upcast of A/B storage; only the running sum is long double)
            Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic> acc(m, n);
            acc.setZero();

            // Special case: BK == 1 uses scalar updates (double multiply, long double add)
            if(BK == 1) {
                // Access as plain matrices (still views, no copy)
                const auto &A   = A_in.derived();
                const auto &B   = B_in.derived();
                auto        Bkk = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(n);
                for(Eigen::Index kk = 0; kk < k; ++kk) {
                    const auto a_col = A.col(kk); // length m
                    Bkk              = B.row(kk);
                    for(Eigen::Index j = 0; j < n; ++j) { acc.col(j).noalias() += (a_col * Bkk(j)).template cast<ScalarL>(); }
                }
                return acc.template cast<Scalar>();
            }

            // General case: BK > 1
            // Reusable FP64 buffer for each block contribution.
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> P(m, n);
            P.setZero();

            for(Eigen::Index kk = 0; kk < k; kk += BK) {
                const Eigen::Index kb = std::min<Eigen::Index>(BK, k - kk);

                P.noalias() = A_in.middleCols(kk, kb) * B_in.middleRows(kk, kb); // FP64 block GEMM

                acc.noalias() += P.template cast<ScalarL>(); // Accumulate block result in long double
            }

            // Downcast final result back to FP64
            return acc.template cast<Scalar>();
        };
        Info<Scalar> info;
        info.contract_left = mps.dimension(1) >= mps.dimension(2);

        Eigen::Tensor<Scalar, 4> mpo_shf  = mpo.shuffle(std::array{0, 3, 2, 1});
        Eigen::Tensor<Scalar, 3> envL_shf = envL.shuffle(std::array{0, 2, 1});

        Eigen::Index md       = mps.dimension(0);
        Eigen::Index mL       = mps.dimension(1);
        Eigen::Index mR       = mps.dimension(2);
        Eigen::Index wL       = mpo.dimension(0);
        Eigen::Index wR       = mpo.dimension(1);
        Eigen::Index wd       = mpo.dimension(3);
        auto         envR_mat = Eigen::Map<const MatrixType>(envR.data(), mR, mR * wR);
        auto         envL_mat = Eigen::Map<const MatrixType>(envL_shf.data(), mL * wL, mL);
        auto         res_shf  = Eigen::Tensor<Scalar, 3>(wd, mR, mL);
        auto         res_mat  = Eigen::Map<MatrixType>(res_shf.data(), wd * mR, mL);

        thread_local Eigen::Tensor<Scalar, 4> T1;
        thread_local Eigen::Tensor<Scalar, 4> T2;
        T1.resize(md, mL, mR, wR);
        T2.resize(wL, wd, mL, mR);

        auto mps_mat = Eigen::Map<const MatrixType>(mps.data(), md * mL, mR);

        {
            auto T1_mat = Eigen::Map<MatrixType>(T1.data(), md * mL, mR * wR);
            T1_mat      = gemm_highprecision_fp80(mps_mat, envR_mat, 1);
        }

        {
            T1           = Eigen::Tensor<Scalar, 4>(T1.shuffle(std::array{0, 3, 1, 2}));
            auto T1_mat  = Eigen::Map<const MatrixType>(T1.data(), md * wR, mL * mR);
            auto T2_mat  = Eigen::Map<MatrixType>(T2.data(), wL * wd, mL * mR);
            auto mpo_mat = Eigen::Map<const MatrixType>(mpo_shf.data(), wL * wd, md * wR);
            T2_mat       = gemm_highprecision_fp80(mpo_mat, T1_mat, 1);
        }

        {
            T2          = Eigen::Tensor<Scalar, 4>(T2.shuffle(std::array{1, 3, 2, 0}));
            auto T2_mat = Eigen::Map<const MatrixType>(T2.data(), wd * mR, mL * wL);
            res_mat     = gemm_highprecision_fp80(T2_mat, envL_mat, 1);
            res         = res_shf.shuffle(std::array{0, 2, 1});
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
    }

    template<typename Scalar>
    Info<Scalar> contract_with_longprod(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        auto gemm_highprecision_fp80 = [](const Eigen::Ref<const MatrixType> &A_in, const Eigen::Ref<const MatrixType> &B_in,
                                          [[maybe_unused]] Eigen::Index BK) -> MatrixType {
            // Multiply in FP64, accumulate in long double, return FP64.
            // - If BK == 1: do scalar FMAs (double multiply) into long double accumulator.
            // - If BK  > 1: do GEMM in double for each k-block, then add that block result into long double accumulator.
            //
            // Requirements: A.cols() == B.rows().

            const Eigen::Index m = A_in.rows();
            const Eigen::Index k = A_in.cols();
            const Eigen::Index n = B_in.cols();

            assert(B_in.rows() == A_in.cols());
            assert(BK >= 1);
            using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;

            const auto &A = A_in.derived();
            const auto &B = B_in.derived();

            // high precision accumulator and compensation
            Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic> acc;
            Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic> comp;
            // Resize and initialize to zero
            acc.setZero(m, n);
            comp.setZero(m, n);

#pragma omp parallel
            {
                Eigen::Matrix<ScalarL, Eigen::Dynamic, 1> aL(m), y(m), tmp(m), prod(m);
                for(Eigen::Index kk = 0; kk < k; ++kk) {
                    aL.noalias()     = A.col(kk).template cast<ScalarL>(); // length m
                    const auto b_row = B.row(kk);
#pragma omp for schedule(static)
                    for(Eigen::Index j = 0; j < n; ++j) {
                        const ScalarL bL      = static_cast<ScalarL>(b_row(j));
                        prod.noalias()        = aL * bL;
                        y.noalias()           = prod - comp.col(j);
                        tmp.noalias()         = acc.col(j) + y;
                        comp.col(j).noalias() = (tmp - acc.col(j)) - y;
                        acc.col(j).noalias()  = tmp;
                    }
                }
            }
            return acc.template cast<Scalar>();
        };
        Info<Scalar> info;
        info.contract_left = mps.dimension(1) >= mps.dimension(2);

        Eigen::Tensor<Scalar, 4> mpo_shf  = mpo.shuffle(std::array{0, 3, 2, 1});
        Eigen::Tensor<Scalar, 3> envL_shf = envL.shuffle(std::array{0, 2, 1});

        Eigen::Index md       = mps.dimension(0);
        Eigen::Index mL       = mps.dimension(1);
        Eigen::Index mR       = mps.dimension(2);
        Eigen::Index wL       = mpo.dimension(0);
        Eigen::Index wR       = mpo.dimension(1);
        Eigen::Index wd       = mpo.dimension(3);
        auto         envR_mat = Eigen::Map<const MatrixType>(envR.data(), mR, mR * wR);
        auto         envL_mat = Eigen::Map<const MatrixType>(envL_shf.data(), mL * wL, mL);
        auto         res_shf  = Eigen::Tensor<Scalar, 3>(wd, mR, mL);
        auto         res_mat  = Eigen::Map<MatrixType>(res_shf.data(), wd * mR, mL);

        thread_local Eigen::Tensor<Scalar, 4> T1;
        thread_local Eigen::Tensor<Scalar, 4> T2;
        T1.resize(md, mL, mR, wR);
        T2.resize(wL, wd, mL, mR);

        auto mps_mat = Eigen::Map<const MatrixType>(mps.data(), md * mL, mR);

        {
            auto T1_mat = Eigen::Map<MatrixType>(T1.data(), md * mL, mR * wR);
            T1_mat      = gemm_highprecision_fp80(mps_mat, envR_mat, 1);
        }

        {
            T1           = Eigen::Tensor<Scalar, 4>(T1.shuffle(std::array{0, 3, 1, 2}));
            auto T1_mat  = Eigen::Map<const MatrixType>(T1.data(), md * wR, mL * mR);
            auto T2_mat  = Eigen::Map<MatrixType>(T2.data(), wL * wd, mL * mR);
            auto mpo_mat = Eigen::Map<const MatrixType>(mpo_shf.data(), wL * wd, md * wR);
            T2_mat       = gemm_highprecision_fp80(mpo_mat, T1_mat, 1);
        }

        {
            T2          = Eigen::Tensor<Scalar, 4>(T2.shuffle(std::array{1, 3, 2, 0}));
            auto T2_mat = Eigen::Map<const MatrixType>(T2.data(), wd * mR, mL * wL);
            res_mat     = gemm_highprecision_fp80(T2_mat, envL_mat, 1);
            res         = res_shf.shuffle(std::array{0, 2, 1});
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
    }

    template<typename Scalar>
    Info<Scalar> contract_with_quadprod(auto &res, const auto &mps, const auto &mpo, const auto &envL, const auto &envR) {
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        auto gemm_highprecision_fp128 = [](const Eigen::Ref<const MatrixType> &A_in, const Eigen::Ref<const MatrixType> &B_in,
                                           [[maybe_unused]] Eigen::Index BK) -> MatrixType {
            // Multiply in FP64, accumulate in long double, return FP64.
            // - If BK == 1: do scalar FMAs (double multiply) into long double accumulator.
            // - If BK  > 1: do GEMM in double for each k-block, then add that block result into long double accumulator.
            //
            // Requirements: A.cols() == B.rows().

            const Eigen::Index m = A_in.rows();
            const Eigen::Index k = A_in.cols();
            const Eigen::Index n = B_in.cols();

            assert(B_in.rows() == A_in.cols());
            assert(BK >= 1);
            using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, cx128, fp128>;

            const auto &A = A_in.derived();
            const auto &B = B_in.derived();

            // high precision accumulator and compensation
            Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic> acc;
            Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic> comp;
            // Resize and initialize to zero
            acc.setZero(m, n);
            comp.setZero(m, n);

#pragma omp parallel
            {
                Eigen::Matrix<ScalarL, Eigen::Dynamic, 1> aL(m), y(m), tmp(m), prod(m);
                for(Eigen::Index kk = 0; kk < k; ++kk) {
                    aL.noalias()     = A.col(kk).template cast<ScalarL>(); // length m
                    const auto b_row = B.row(kk);
#pragma omp for schedule(static)
                    for(Eigen::Index j = 0; j < n; ++j) {
                        const ScalarL bL      = static_cast<ScalarL>(b_row(j));
                        prod.noalias()        = aL * bL;
                        y.noalias()           = prod - comp.col(j);
                        tmp.noalias()         = acc.col(j) + y;
                        comp.col(j).noalias() = (tmp - acc.col(j)) - y;
                        acc.col(j).noalias()  = tmp;
                    }
                }
            }
            return acc.template cast<Scalar>();
        };
        Info<Scalar> info;
        info.contract_left = mps.dimension(1) >= mps.dimension(2);

        Eigen::Tensor<Scalar, 4> mpo_shf  = mpo.shuffle(std::array{0, 3, 2, 1});
        Eigen::Tensor<Scalar, 3> envL_shf = envL.shuffle(std::array{0, 2, 1});

        Eigen::Index md       = mps.dimension(0);
        Eigen::Index mL       = mps.dimension(1);
        Eigen::Index mR       = mps.dimension(2);
        Eigen::Index wL       = mpo.dimension(0);
        Eigen::Index wR       = mpo.dimension(1);
        Eigen::Index wd       = mpo.dimension(3);
        auto         envR_mat = Eigen::Map<const MatrixType>(envR.data(), mR, mR * wR);
        auto         envL_mat = Eigen::Map<const MatrixType>(envL_shf.data(), mL * wL, mL);
        auto         res_shf  = Eigen::Tensor<Scalar, 3>(wd, mR, mL);
        auto         res_mat  = Eigen::Map<MatrixType>(res_shf.data(), wd * mR, mL);

        thread_local Eigen::Tensor<Scalar, 4> T1;
        thread_local Eigen::Tensor<Scalar, 4> T2;
        T1.resize(md, mL, mR, wR);
        T2.resize(wL, wd, mL, mR);

        auto mps_mat = Eigen::Map<const MatrixType>(mps.data(), md * mL, mR);

        {
            auto T1_mat = Eigen::Map<MatrixType>(T1.data(), md * mL, mR * wR);
            T1_mat      = gemm_highprecision_fp128(mps_mat, envR_mat, 1);
        }

        {
            T1           = Eigen::Tensor<Scalar, 4>(T1.shuffle(std::array{0, 3, 1, 2}));
            auto T1_mat  = Eigen::Map<const MatrixType>(T1.data(), md * wR, mL * mR);
            auto T2_mat  = Eigen::Map<MatrixType>(T2.data(), wL * wd, mL * mR);
            auto mpo_mat = Eigen::Map<const MatrixType>(mpo_shf.data(), wL * wd, md * wR);
            T2_mat       = gemm_highprecision_fp128(mpo_mat, T1_mat, 1);
        }

        {
            T2          = Eigen::Tensor<Scalar, 4>(T2.shuffle(std::array{1, 3, 2, 0}));
            auto T2_mat = Eigen::Map<const MatrixType>(T2.data(), wd * mR, mL * wL);
            res_mat     = gemm_highprecision_fp128(T2_mat, envL_mat, 1);
            res         = res_shf.shuffle(std::array{0, 2, 1});
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
    }

}

template<typename Scalar>
void tools::common::contraction::vector_matrix_product(Scalar             *res_ptr,                                 //
                                                       const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                       const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                       const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                       const Scalar *const envR_ptr, std::array<long, 3> envR_dims  //
) {
    // This applies the mpo's with corresponding environments to local multisite mps
    // This is usually the operation H|psi>  or H²|psi>
    using RealScalar         = decltype(std::real(std::declval<Scalar>()));
    constexpr bool use_tblis = settings::tblis_enabled and (std::is_same_v<RealScalar, fp32> or std::is_same_v<RealScalar, fp64>);

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

    auto mps_norm  = internal_vmp::get_norm(mps_ptr, mps_dims);
    auto mpo_norm  = internal_vmp::get_norm(mpo_ptr, mpo_dims);
    auto envL_norm = internal_vmp::get_norm(envL_ptr, envL_dims);
    auto envR_norm = internal_vmp::get_norm(envR_ptr, envR_dims);

    bool contract_left = mps_dims[1] >= mps_dims[2];

    [[maybe_unused]] auto ST1 = contract_left ? mps_norm * envL_norm : mps_norm * envR_norm;
    [[maybe_unused]] auto ST2 = ST1 * mpo_norm;
    [[maybe_unused]] auto ST3 = contract_left ? ST2 * envR_norm : ST2 * envL_norm;

    internal_vmp::Info<Scalar>       info;
    [[maybe_unused]] std::string msg;
    if(ST1 > info.highprec_threshold) {
        // if(true) {
        if constexpr(settings::debug_contraction) msg = fmt::format("| running highprecision: ST1 {:.4e} > {:.4e}", fp(ST1), fp(info.highprec_threshold));
        info = internal_vmp::contract_with_longprod<Scalar>(res, mps, mpo, envL, envR);
        // info = internal_vmp::contract_with_quadprod<Scalar>(res, mps, mpo, envL, envR);
        // info = internal_vmp::contract_with_longsum<Scalar>(res, mps, mpo, envL, envR);
        if constexpr(use_tblis) {
            Eigen::Tensor<Scalar, 3> resd(mps_dims);
            auto                     infod = internal_vmp::contract_with_tblis<Scalar>(resd, mps, mpo, envL, envR);
            RealScalar               diff  = (tenx::VectorMap(res) - tenx::VectorMap(resd)).norm();
            if constexpr(settings::debug_contraction) msg += fmt::format(" diff={:.4e}", fp(diff));
        }
    } else {
        if constexpr(use_tblis) {
            info = internal_vmp::contract_with_tblis<Scalar>(res, mps, mpo, envL, envR);
        } else {
            info = internal_vmp::contract_with_eigen<Scalar>(res, mps, mpo, envL, envR);
        }
    }
    using namespace internal_vmp;
    if constexpr(settings::debug_contraction)
        if(!msg.empty())
            tools::log->info("res {:.4e} mps {:.4e} envL {:.4e} envR {:.4e} mpo {:.4e} ST1 {:.4e} ST2 {:.4e} ST3 {:.4e} cf: {:.4e} {}",
                             fp(get_norm(res_ptr, mps_dims)), fp(mps_norm), fp(envL_norm), fp(envR_norm), fp(mpo_norm), fp(info.ST1), fp(info.ST2),
                             fp(info.ST3), fp(info.cancelation_factor), msg);
}

template<typename Scalar, typename mpo_type>
void tools::common::contraction::vector_matrix_product(Scalar *res_ptr, const Scalar *const mps_ptr, std::array<long, 3> mps_dims,
                                                       const std::vector<mpo_type> &mpos_shf, const Scalar *const envL_ptr, std::array<long, 3> envL_dims,
                                                       const Scalar *const envR_ptr, std::array<long, 3> envR_dims) {
    // Make sure the mpos are pre-shuffled. If not, shuffle and call this function again
    bool is_shuffled = mpos_shf.front().dimension(2) == envL_dims[2] and mpos_shf.back().dimension(3) == envR_dims[2];
    if(not is_shuffled) {
        // mpos_shf are not actually shuffled. Let's shuffle.
        std::vector<Eigen::Tensor<Scalar, 4>> mpos_really_shuffled;
        for(const auto &mpo : mpos_shf) { mpos_really_shuffled.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1})); }
        return vector_matrix_product(res_ptr, mps_ptr, mps_dims, mpos_really_shuffled, envL_ptr, envL_dims, envR_ptr, envR_dims);
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
