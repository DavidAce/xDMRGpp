#include "lapack_wrappers.h"

// References:
// MKL ?gesvj: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/gesvj.html
// Netlib ?gesvj: https://www.netlib.org/lapack/explore-html/d9/deb/group__gesvj.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, cx64>
    int zgesvj(Context<Scalar> &ctx) {
        auto &cwork = ctx.workspace.cwork;
        auto &rwork = ctx.workspace.rwork;

        ctx.S.resize(ctx.sizeS);
        ctx.V.resize(ctx.rowsV, ctx.colsV);

        int lcwork = std::max(1, ctx.rowsA + ctx.colsA);
        int lrwork = std::max(6, ctx.colsA);

        cwork.resize(static_cast<size_t>(lcwork));
        rwork.resize(static_cast<size_t>(lrwork));

        int info = LAPACKE_zgesvj_work(LAPACK_COL_MAJOR, 'G', 'U', 'V', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.ldv, ctx.V.data(), ctx.ldv,
                                       cwork.data(), lcwork, rwork.data(), lrwork);
        if(info != 0) return info;

        ctx.U  = std::move(ctx.A);
        ctx.VT = ctx.V.adjoint();
        ctx.V.resize(0, 0);
        return info;
    }

    template int zgesvj<cx64>(Context<cx64> &ctx);
}
