#include "lapack_wrappers.h"

// References:
// MKL ?gejsv: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/gejsv.html
// Netlib ?gejsv: https://www.netlib.org/lapack/explore-html/d8/d78/group__gejsv.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, fp32>
    int sgejsv(Context<Scalar> &ctx) {
        auto &iwork = ctx.workspace.iwork;
        auto &rwork = ctx.workspace.rwork;

        ctx.S.resize(ctx.sizeS);
        ctx.U.resize(ctx.rowsU, ctx.colsU);
        ctx.V.resize(ctx.rowsV, ctx.colsV);

        int lrwork = std::max(2 * ctx.rowsA + ctx.colsA, 6 * ctx.colsA + 2 * ctx.colsA * ctx.colsA);
        int liwork = std::max(3, ctx.rowsA + 3 * ctx.colsA);

        rwork.resize(safe_cast<size_t>(lrwork));
        iwork.resize(safe_cast<size_t>(liwork));

        int info = LAPACKE_sgejsv_work(LAPACK_COL_MAJOR, 'F', 'U', 'V', 'N', 'T', 'N', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.U.data(),
                                       ctx.ldu, ctx.V.data(), ctx.ldv, rwork.data(), lrwork, iwork.data());
        if(info != 0) return info;

        ctx.VT = ctx.V.adjoint();
        ctx.V.resize(0, 0);
        return info;
    }

    template int sgejsv<fp32>(Context<fp32> &ctx);
}
