#include "lapack_wrappers.h"

// References:
// MKL ?gesvj: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/gesvj.html
// Netlib ?gesvj: https://www.netlib.org/lapack/explore-html/d9/deb/group__gesvj.html

namespace svd::internal::lapack_wrappers {
    template<typename Scalar>
        requires std::same_as<Scalar, fp64>
    int dgesvj(Context<Scalar> &ctx) {
        auto &rwork = ctx.workspace.rwork;

        ctx.S.resize(ctx.sizeS);
        ctx.V.resize(ctx.rowsV, ctx.colsV);

        int lrwork = std::max(6, ctx.rowsA + ctx.colsA);
        rwork.resize(safe_cast<size_t>(lrwork));

        int info = LAPACKE_dgesvj_work(LAPACK_COL_MAJOR, 'G', 'U', 'V', ctx.rowsA, ctx.colsA, ctx.A.data(), ctx.lda, ctx.S.data(), ctx.ldv, ctx.V.data(),
                                       ctx.ldv, rwork.data(), lrwork);
        if(info != 0) return info;

        ctx.U  = std::move(ctx.A);
        ctx.VT = ctx.V.adjoint();
        return info;
    }

    template int dgesvj<fp64>(Context<fp64> &ctx);
}
