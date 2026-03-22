#include "lapack_wrappers.h"

// References:
// MKL ?gejsv: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/gejsv.html
// Netlib ?gejsv: https://www.netlib.org/lapack/explore-html/d8/d78/group__gejsv.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, cx64>
    int zgejsv(Context<Scalar> &ctx) {
        auto &iwork = ctx.workspace.iwork;
        auto &cwork = ctx.workspace.cwork;
        auto &rwork = ctx.workspace.rwork;

        ctx.S.resize(ctx.sizeS);
        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.V.resize(ctx.rowsV, ctx.colsV);

        cwork.resize(2ul);
        rwork.resize(1ul);
        iwork.resize(1ul);

        int info = LAPACKE_zgejsv_work(LAPACK_COL_MAJOR, 'F', 'U', 'V', 'N', 'T', 'N', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(),
                                       ctx.ldu, ctx.V.data(), ctx.ldv, cwork.data(), -1, rwork.data(), -1, iwork.data());
        if(info != 0) return info;

        int lcwork = safe_cast<int>(std::real(cwork[0]));
        int lrwork = safe_cast<int>(rwork[0]);
        int liwork = safe_cast<int>(iwork[0]);
        cwork.resize(safe_cast<size_t>(std::max({2, 5 * ctx.colsA + 2 * ctx.colsA * ctx.colsA, lcwork})));
        rwork.resize(safe_cast<size_t>(std::max({7, 2 * ctx.rowsA, lrwork})));
        iwork.resize(safe_cast<size_t>(std::max({1, ctx.rowsA + ctx.colsA, liwork})));

        info = LAPACKE_zgejsv_work(LAPACK_COL_MAJOR, 'F', 'U', 'V', 'N', 'T', 'N', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(),
                                   ctx.ldu, ctx.V.data(), ctx.ldv, cwork.data(), lcwork, rwork.data(), lrwork, iwork.data());
        if(info != 0) return info;

        ctx.VT = ctx.V.adjoint();
        ctx.V.resize(0, 0);
        return info;
    }

    template int zgejsv<cx64>(Context<cx64> &ctx);
}
