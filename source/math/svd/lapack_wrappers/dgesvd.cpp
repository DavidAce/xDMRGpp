#include "lapack_wrappers.h"

// References:
// MKL ?gesvd: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/gesvd.html
// Netlib ?gesvd: https://www.netlib.org/lapack/explore-html/d1/d7f/group__gesvd.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgesvd(Context<Scalar> &ctx) {
        auto &rwork = ctx.workspace.rwork;

        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.S.resize(ctx.sizeS);
        ctx.VT.resize(ctx.rowsVT, ctx.colsVT);
        rwork.resize(1ul);

        int info = LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'S', 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu,
                                       ctx.VT.data(), ctx.ldvt, rwork.data(), -1);
        if(info != 0) return info;

        int lrwork = safe_cast<int>(rwork[0]);
        rwork.resize(safe_cast<size_t>(std::max(1, lrwork)));

        return LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'S', 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(),
                                   ctx.ldvt, rwork.data(), lrwork);
    }

    template int dgesvd<fp64>(Context<fp64> &ctx);
}
