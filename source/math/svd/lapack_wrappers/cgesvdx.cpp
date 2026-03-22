#include "lapack_wrappers.h"

// References:
// MKL ?gesvdx: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/gesvdx.html
// Netlib ?gesvdx: https://www.netlib.org/lapack/explore-html/dc/d4a/group__gesvdx.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgesvdx(Context<Scalar> &ctx) {
        auto &iwork = ctx.workspace.iwork;
        auto &cwork = ctx.workspace.cwork;
        auto &rwork = ctx.workspace.rwork;
        auto  sel   = make_gesvdx_selection(ctx, std::min<fp32>(1e-10f, static_cast<fp32>(ctx.truncation_lim) / 5.0f),
                                          std::max<fp32>(1e+10f, static_cast<fp32>(ctx.truncation_lim) / 5.0f));

        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.S.resize(ctx.sizeS);
        ctx.VT.resize(ctx.rowsVT, ctx.colsVT);

        int ns     = 0;
        int lcwork = std::max(1, ctx.mn * 2 + ctx.mx);
        int lrwork = ctx.mn * (ctx.mn * 2 + 15 * ctx.mn);
        int liwork = 12 * ctx.mn;

        cwork.resize(static_cast<size_t>(lcwork));
        rwork.resize(static_cast<size_t>(lrwork));
        iwork.resize(static_cast<size_t>(liwork));

        int info = LAPACKE_cgesvdx_work(LAPACK_COL_MAJOR, 'V', 'V', sel.range, ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, sel.vl, sel.vu, sel.il, sel.iu,
                                        &ns, ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(), ctx.ldvt, cwork.data(), -1, rwork.data(), iwork.data());
        if(info != 0) return info;

        lcwork = safe_cast<int>(std::real(cwork[0]));
        cwork.resize(safe_cast<size_t>(std::max(1, lcwork)));

        info = LAPACKE_cgesvdx_work(LAPACK_COL_MAJOR, 'V', 'V', 'V', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, sel.vl, sel.vu, sel.il, sel.iu, &ns,
                                    ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(), ctx.ldvt, cwork.data(), lcwork, rwork.data(), iwork.data());
        if(info != 0) return info;

        ctx.U.conservativeResize(Eigen::NoChange, ns);
        ctx.S.conservativeResize(ns);
        ctx.VT.conservativeResize(ns, Eigen::NoChange);
        return info;
    }

    template int cgesvdx<cx32>(Context<cx32> &ctx);
}
