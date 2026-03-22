#include "lapack_wrappers.h"

// References:
// MKL ?gesdd: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/gesdd.html
// Netlib ?gesdd: https://www.netlib.org/lapack/explore-html/df/d22/group__gesdd.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, fp32>
    int sgesdd(Context<Scalar> &ctx) {
        auto &iwork = ctx.workspace.iwork;
        auto &rwork = ctx.workspace.rwork;

        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.S.resize(ctx.sizeS);
        ctx.VT.resize(ctx.rowsVT, ctx.colsVT);

        int lrwork = 1;
        int liwork = std::max(1, 8 * ctx.mn);
        rwork.resize(safe_cast<size_t>(lrwork));
        iwork.resize(safe_cast<size_t>(liwork));

        int info = LAPACKE_sgesdd_work(LAPACK_COL_MAJOR, 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu,
                                       ctx.VT.data(), ctx.ldvt, rwork.data(), -1, iwork.data());
        if(info != 0) return info;

        lrwork = safe_cast<int>(rwork[0]);
        rwork.resize(safe_cast<size_t>(std::max(1, lrwork)));

        return LAPACKE_sgesdd_work(LAPACK_COL_MAJOR, 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu,
                                   ctx.VT.data(), ctx.ldvt, rwork.data(), lrwork, iwork.data());
    }

    template int sgesdd<fp32>(Context<fp32> &ctx);
}
