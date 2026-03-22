#pragma once

#include "../../svd.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <vector>

#ifndef lapack_complex_float
    #define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
    #define lapack_complex_double std::complex<double>
#endif

// complex must be included before lapacke!
#if defined(_LAPACKE_H_)
    #pragma message "LAPACKE header was already included elsewhere"
#endif

#if defined(MKL_AVAILABLE)
    #include <mkl_lapacke.h>
#elif defined(OPENBLAS_AVAILABLE)
    #include <openblas/lapacke.h>
#elif defined(FLEXIBLAS_AVAILABLE)
    #include <flexiblas/lapacke.h>
#else
    #include <lapacke.h>
#endif

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
    concept lapacke_scalar = std::same_as<Scalar, fp32> || std::same_as<Scalar, fp64> || std::same_as<Scalar, cx32> || std::same_as<Scalar, cx64>;

    template<lapacke_scalar Scalar>
    struct Workspace {
        // Kept separate from the wrapper implementation so the call site can later
        // switch from stack-local vectors to a thread-local cache without changing
        // the wrapper API.
        std::vector<int>                     iwork;
        std::vector<Scalar>                  cwork;
        std::vector<svd::RealScalar<Scalar>> rwork;
    };

    template<lapacke_scalar Scalar>
    struct Context {
        using real_type = svd::RealScalar<Scalar>;

        svd::MatrixType<Scalar>            &A;
        svd::MatrixType<Scalar>            &U;
        svd::VectorType<real_type>         &S;
        svd::MatrixType<Scalar>            &V;
        svd::MatrixType<Scalar>            &VT;
        Workspace<Scalar>                  &workspace;
        int                                 rowsA;
        int                                 colsA;
        int                                 sizeS;
        int                                 rowsU;
        int                                 colsU;
        int                                 rowsVT;
        int                                 colsVT;
        int                                 rowsV;
        int                                 colsV;
        int                                 lda;
        int                                 ldu;
        int                                 ldvt;
        int                                 ldv;
        int                                 mx;
        int                                 mn;
        long                                rank_max;
        double                              truncation_lim;
        const std::optional<svdx_select_t> &svdx_select;

        Context(svd::MatrixType<Scalar> &A_,
                svd::MatrixType<Scalar> &U_,
                svd::VectorType<real_type> &S_,
                svd::MatrixType<Scalar> &V_,
                svd::MatrixType<Scalar> &VT_,
                Workspace<Scalar> &workspace_,
                int rowsA_,
                int colsA_,
                int sizeS_,
                int rowsU_,
                int colsU_,
                int rowsVT_,
                int colsVT_,
                int rowsV_,
                int colsV_,
                int lda_,
                int ldu_,
                int ldvt_,
                int ldv_,
                int mx_,
                int mn_,
                long rank_max_,
                double truncation_lim_,
                const std::optional<svdx_select_t> &svdx_select_)
            : A(A_)
            , U(U_)
            , S(S_)
            , V(V_)
            , VT(VT_)
            , workspace(workspace_)
            , rowsA(rowsA_)
            , colsA(colsA_)
            , sizeS(sizeS_)
            , rowsU(rowsU_)
            , colsU(colsU_)
            , rowsVT(rowsVT_)
            , colsVT(colsVT_)
            , rowsV(rowsV_)
            , colsV(colsV_)
            , lda(lda_)
            , ldu(ldu_)
            , ldvt(ldvt_)
            , ldv(ldv_)
            , mx(mx_)
            , mn(mn_)
            , rank_max(rank_max_)
            , truncation_lim(truncation_lim_)
            , svdx_select(svdx_select_) {}
    };

    template<lapacke_scalar Scalar>
    constexpr char type_prefix() {
        if constexpr(std::same_as<Scalar, fp32>) return 's';
        if constexpr(std::same_as<Scalar, fp64>) return 'd';
        if constexpr(std::same_as<Scalar, cx32>) return 'c';
        return 'z';
    }

    template<lapacke_scalar Scalar>
    constexpr std::string_view error_prefix() {
        if constexpr(std::same_as<Scalar, fp32>) return "Lapacke SVD s";
        if constexpr(std::same_as<Scalar, fp64>) return "Lapacke SVD d";
        if constexpr(std::same_as<Scalar, cx32>) return "c";
        return "z";
    }

    template<lapacke_scalar Scalar>
    struct GesvdxSelection {
        svd::RealScalar<Scalar> vl;
        svd::RealScalar<Scalar> vu;
        int                     il;
        int                     iu;
        char                    range;
    };

    template<lapacke_scalar Scalar>
    inline GesvdxSelection<Scalar> make_gesvdx_selection(const Context<Scalar> &ctx, svd::RealScalar<Scalar> vl_default,
                                                         svd::RealScalar<Scalar> vu_default) {
        auto vl    = vl_default;
        auto vu    = vu_default;
        int  il    = 1;
        int  iu    = std::min<int>(ctx.sizeS, safe_cast<int>(ctx.rank_max));
        char range = ctx.rank_max < ctx.sizeS ? 'I' : 'V';

        if(ctx.svdx_select.has_value()) {
            if(std::holds_alternative<svdx_indices_t>(ctx.svdx_select.value())) {
                auto sel = std::get<svdx_indices_t>(ctx.svdx_select.value());
                iu       = std::min<int>(safe_cast<int>(sel.iu), safe_cast<int>(ctx.rank_max));
                il       = std::min<int>(safe_cast<int>(sel.il), safe_cast<int>(ctx.rank_max));
                range    = 'I';
            } else if(std::holds_alternative<svdx_values_t>(ctx.svdx_select.value())) {
                auto sel = std::get<svdx_values_t>(ctx.svdx_select.value());
                if(std::isfinite(sel.vl)) vl = static_cast<svd::RealScalar<Scalar>>(sel.vl);
                if(std::isfinite(sel.vu)) vu = static_cast<svd::RealScalar<Scalar>>(sel.vu);
                range = 'V';
            }
        }
        return {.vl = vl, .vu = vu, .il = il, .iu = iu, .range = range};
    }

    template<typename Scalar>
        requires std::same_as<Scalar, fp32>
    int sgesvd(Context<Scalar> &ctx);
    extern template int sgesvd<fp32>(Context<fp32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgesvd(Context<Scalar> &ctx);
    extern template int dgesvd<fp64>(Context<fp64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgesvd(Context<Scalar> &ctx);
    extern template int cgesvd<cx32>(Context<cx32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx64>
    int zgesvd(Context<Scalar> &ctx);
    extern template int zgesvd<cx64>(Context<cx64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp32>
    int sgesvj(Context<Scalar> &ctx);
    extern template int sgesvj<fp32>(Context<fp32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgesvj(Context<Scalar> &ctx);
    extern template int dgesvj<fp64>(Context<fp64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgesvj(Context<Scalar> &ctx);
    extern template int cgesvj<cx32>(Context<cx32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx64>
    int zgesvj(Context<Scalar> &ctx);
    extern template int zgesvj<cx64>(Context<cx64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp32>
    int sgejsv(Context<Scalar> &ctx);
    extern template int sgejsv<fp32>(Context<fp32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgejsv(Context<Scalar> &ctx);
    extern template int dgejsv<fp64>(Context<fp64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgejsv(Context<Scalar> &ctx);
    extern template int cgejsv<cx32>(Context<cx32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx64>
    int zgejsv(Context<Scalar> &ctx);
    extern template int zgejsv<cx64>(Context<cx64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp32>
    int sgesdd(Context<Scalar> &ctx);
    extern template int sgesdd<fp32>(Context<fp32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgesdd(Context<Scalar> &ctx);
    extern template int dgesdd<fp64>(Context<fp64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgesdd(Context<Scalar> &ctx);
    extern template int cgesdd<cx32>(Context<cx32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx64>
    int zgesdd(Context<Scalar> &ctx);
    extern template int zgesdd<cx64>(Context<cx64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp32>
    int sgesvdx(Context<Scalar> &ctx);
    extern template int sgesvdx<fp32>(Context<fp32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgesvdx(Context<Scalar> &ctx);
    extern template int dgesvdx<fp64>(Context<fp64> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgesvdx(Context<Scalar> &ctx);
    extern template int cgesvdx<cx32>(Context<cx32> &ctx);

    template<typename Scalar>
        requires std::same_as<Scalar, cx64>
    int zgesvdx(Context<Scalar> &ctx);
    extern template int zgesvdx<cx64>(Context<cx64> &ctx);
}
