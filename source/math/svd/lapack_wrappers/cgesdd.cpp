#include "lapack_wrappers.h"

// References:
// MKL ?gesdd: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/gesdd.html
// Netlib ?gesdd: https://www.netlib.org/lapack/explore-html/df/d22/group__gesdd.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, cx32>
    int cgesdd(Context<Scalar> &ctx) {
        auto &iwork = ctx.workspace.iwork;
        auto &cwork = ctx.workspace.cwork;
        auto &rwork = ctx.workspace.rwork;

        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.S.resize(ctx.sizeS);
        ctx.VT.resize(ctx.rowsVT, ctx.colsVT);

        int lcwork = 1;
        int lrwork = std::max(1, std::max(5 * ctx.mn * ctx.mn + 5 * ctx.mn, 2 * ctx.mx * ctx.mn + 2 * ctx.mn * ctx.mn + ctx.mn));
        int liwork = std::max(1, 8 * ctx.mn);

        cwork.resize(static_cast<size_t>(lcwork));
        rwork.resize(static_cast<size_t>(lrwork));
        iwork.resize(static_cast<size_t>(liwork));

        int info = LAPACKE_cgesdd_work(LAPACK_COL_MAJOR, 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(),
                                       ctx.ldvt, cwork.data(), -1, rwork.data(), iwork.data());
        if(info != 0) return info;

        lcwork = safe_cast<int>(std::real(cwork[0]));
        cwork.resize(safe_cast<size_t>(std::max(1, lcwork)));

        return LAPACKE_cgesdd_work(LAPACK_COL_MAJOR, 'S', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(), ctx.ldu, ctx.VT.data(),
                                   ctx.ldvt, cwork.data(), lcwork, rwork.data(), iwork.data());
    }

    template int cgesdd<cx32>(Context<cx32> &ctx);
}
