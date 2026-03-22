#include "lapack_wrappers.h"

// References:
// MKL ?gesvdx: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/gesvdx.html
// Netlib ?gesvdx: https://www.netlib.org/lapack/explore-html/dc/d4a/group__gesvdx.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgesvdx(Context<Scalar> &ctx) {
        auto &iwork = ctx.workspace.iwork;
        auto &rwork = ctx.workspace.rwork;
        auto  sel   = make_gesvdx_selection(ctx, std::min(1e-10, ctx.truncation_lim / 5.0), std::max(1e+10, ctx.truncation_lim / 5.0));

        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.S.resize(ctx.sizeS);
        ctx.VT.resize(ctx.rowsVT, ctx.colsVT);

        int ns     = 0;
        int lrwork = std::max(1, ctx.mn * 2 + ctx.mx);
        int liwork = std::max(1, 12 * ctx.mn);
        rwork.resize(safe_cast<size_t>(lrwork));
        iwork.resize(safe_cast<size_t>(liwork));

        int info = LAPACKE_dgesvdx_work(LAPACK_COL_MAJOR, 'V', 'V', sel.range, ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, sel.vl, sel.vu, sel.il, sel.iu,
                                        &ns, ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(), ctx.ldvt, rwork.data(), -1, iwork.data());
        if(info != 0) return info;

        lrwork = safe_cast<int>(std::real(rwork[0]));
        rwork.resize(safe_cast<size_t>(std::max(1, lrwork)));

        info = LAPACKE_dgesvdx_work(LAPACK_COL_MAJOR, 'V', 'V', sel.range, ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, sel.vl, sel.vu, sel.il, sel.iu, &ns,
                                    ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(), ctx.ldvt, rwork.data(), lrwork, iwork.data());
        if(info != 0) return info;

        ctx.U  = ctx.U.leftCols(ns).eval();
        ctx.S  = ctx.S.head(ns).eval();
        ctx.VT = ctx.VT.topRows(ns).eval();
        return info;
    }

    template int dgesvdx<fp64>(Context<fp64> &ctx);
}
