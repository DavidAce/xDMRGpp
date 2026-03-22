#include "lapack_wrappers.h"

// References:
// MKL ?gesvd: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/gesvd.html
// Netlib ?gesvd: https://www.netlib.org/lapack/explore-html/d1/d7f/group__gesvd.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgesvd(Context<Scalar> &ctx) {
        auto &cwork = ctx.workspace.cwork;
        auto &rwork = ctx.workspace.rwork;

        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.S.resize(ctx.sizeS);
        ctx.VT.resize(ctx.rowsVT, ctx.colsVT);

        int lcwork = 1;
        int lrwork = std::max(1, 5 * ctx.mn);
        cwork.resize(safe_cast<size_t>(lcwork));
        rwork.resize(safe_cast<size_t>(lrwork));

        int info = LAPACKE_cgesvd_work(LAPACK_COL_MAJOR, 'S', 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu,
                                       ctx.VT.data(), ctx.ldvt, cwork.data(), -1, rwork.data());
        if(info != 0) return info;

        lcwork = safe_cast<int>(std::real(cwork[0]));
        cwork.resize(safe_cast<size_t>(std::max(1, lcwork)));

        return LAPACKE_cgesvd_work(LAPACK_COL_MAJOR, 'S', 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(),
                                   ctx.ldvt, cwork.data(), lcwork, rwork.data());
    }

    template int cgesvd<cx32>(Context<cx32> &ctx);
}
