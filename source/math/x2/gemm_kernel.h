#pragma once
#include "cursor.h"
#include "util.h"
#include "view.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <type_traits>
#include <utility>


template<typename Scalar, int rankC, int rankA, int rankB>
void gemm_x2_kernel_fused_packed_Cx2_Ax2_Bx2(
    x2::TensorAsMatrixView<Scalar, rankC>            &C_hi, //
    x2::TensorAsMatrixView<Scalar, rankC>            &C_lo, //
    const x2::ConstTensorAsMatrixView<Scalar, rankA> &A_hi, //
    const x2::ConstTensorAsMatrixView<Scalar, rankA> &A_lo, //
    const x2::ConstTensorAsMatrixView<Scalar, rankB> &B_hi, //
    const x2::ConstTensorAsMatrixView<Scalar, rankB> &B_lo)
{
    constexpr Eigen::Index IB = 64;
    constexpr Eigen::Index JB = 8;
    constexpr Eigen::Index BK = 128;

    const Eigen::Index m = A_hi.rows();
    const Eigen::Index k = A_hi.cols();
    const Eigen::Index n = B_hi.cols();

    assert(A_lo.rows() == m && A_lo.cols() == k);
    assert(B_hi.rows() == k && B_hi.cols() == n);
    assert(B_lo.rows() == k && B_lo.cols() == n);
    assert(C_hi.rows() == m && C_hi.cols() == n);
    assert(C_lo.rows() == m && C_lo.cols() == n);

#pragma omp parallel
    {
        x2::X2MatrixCursor<Scalar, rankA, x2::Access::ReadOnly>  Acur(A_hi, A_lo);
        x2::X2MatrixCursor<Scalar, rankB, x2::Access::ReadOnly>  Bcur(B_hi, B_lo);
        x2::X2MatrixCursor<Scalar, rankC, x2::Access::ReadWrite> Ccur(C_hi, C_lo);

        alignas(64) Scalar sum_hi_buf[JB * IB];
        alignas(64) Scalar sum_lo_buf[JB * IB];

        alignas(64) Scalar Apack_hi[BK * IB];
        alignas(64) Scalar Apack_lo[BK * IB];
        alignas(64) Scalar Bpack_hi[BK * JB];
        alignas(64) Scalar Bpack_lo[BK * JB];

        // Parallelize only over j0 so each thread owns disjoint C column-panels.
#pragma omp for schedule(static)
        for(Eigen::Index j0 = 0; j0 < n; j0 += JB) {
            const Eigen::Index jb = std::min<Eigen::Index>(JB, n - j0);

            for(Eigen::Index k0 = 0; k0 < k; k0 += BK) {
                const Eigen::Index kb = std::min<Eigen::Index>(BK, k - k0);

                // ---- PACK B ONCE for this (j0,k0): (kb x jb) -> Bpack[kk][j] ----
                for(Eigen::Index kk = 0; kk < kb; ++kk) {
                    Scalar            *bh  = Bpack_hi + kk * JB;
                    Scalar            *bl  = Bpack_lo + kk * JB;
                    const Eigen::Index row = k0 + kk;

                    Bcur.init(row, j0);
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        bh[j] = *Bcur.ptr_hi();
                        bl[j] = *Bcur.ptr_lo();
                        if(j + 1 < jb) { Bcur.stepCol(); }
                    }
                }

                // ---- REUSE packed B for all i0 blocks ----
                for(Eigen::Index i0 = 0; i0 < m; i0 += IB) {
                    const Eigen::Index ib = std::min<Eigen::Index>(IB, m - i0);

                    // zero partial sums for THIS (i0,k0) panel update
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        std::fill_n(sum_hi_buf + j * IB, ib, Scalar(0));
                        std::fill_n(sum_lo_buf + j * IB, ib, Scalar(0));
                    }

                    // PACK A for THIS (i0,k0): (ib x kb) -> Apack[kk][i]
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        Scalar            *ah  = Apack_hi + kk * IB;
                        Scalar            *al  = Apack_lo + kk * IB;
                        const Eigen::Index col = k0 + kk;

                        Acur.init(i0, col);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            ah[i] = *Acur.ptr_hi();
                            al[i] = *Acur.ptr_lo();
                            if(i + 1 < ib) { Acur.stepRow(); }
                        }
                    }

                    // COMPUTE partial update using packed A and packed B
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Scalar *ah = Apack_hi + kk * IB;
                        const Scalar *al = Apack_lo + kk * IB;
                        const Scalar *bh = Bpack_hi + kk * JB;
                        const Scalar *bl = Bpack_lo + kk * JB;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Scalar bhj = bh[j];
                            const Scalar blj = bl[j];

                            Scalar *__restrict sh = sum_hi_buf + j * IB;
                            Scalar *__restrict sl = sum_lo_buf + j * IB;

                            for(Eigen::Index i = 0; i < ib; ++i) {
                                const Scalar ahi = ah[i];
                                const Scalar ali = al[i];

                                const Scalar p_hi = ahi * bhj;
                                const Scalar p_lo = x2_detail::mul_residual(ahi, bhj, p_hi);
                                x2_detail::accumulate(sh[i], sl[i], p_hi, p_lo);

                                if constexpr(std::is_floating_point_v<Scalar>) {
                                    sl[i] = std::fma(ahi, blj, sl[i]);
                                    sl[i] = std::fma(ali, bhj, sl[i]);
                                    sl[i] = std::fma(ali, blj, sl[i]);
                                } else {
                                    sl[i] += ahi * blj;
                                    sl[i] += ali * bhj;
                                    sl[i] += ali * blj;
                                }
                            }
                        }
                    }

                    // STORE/ACCUMULATE into C for this k0-panel.
                      for(Eigen::Index j = 0; j < jb; ++j) {
                        const Scalar *__restrict sh = sum_hi_buf + j * IB;
                        const Scalar *__restrict sl = sum_lo_buf + j * IB;

                        Ccur.init(i0, j0 + j);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            Scalar s, e;

                            if(k0 == 0) {
                                // First panel: just renorm the partial sum into (hi,lo)
                                x2_detail::add_with_residual(sh[i], sl[i], s, e);
                            } else {
                                // Later panels: accumulate into existing C, then renorm
                                Scalar ch = *Ccur.ptr_hi();
                                Scalar cl = *Ccur.ptr_lo();

                                x2_detail::accumulate(ch, cl, sh[i], sl[i]); // (ch,cl) += (sh,sl)
                                x2_detail::add_with_residual(ch, cl, s, e);   // renorm

                            }

                            *Ccur.ptr_hi() = s;
                            *Ccur.ptr_lo() = e;

                            if(i + 1 < ib) { Ccur.stepRow(); }
                        }
                    }
                }
            }
        }
    }
}

template<typename Scalar, int rankC, int rankA, int rankB>
void gemm_x2_kernel_fused_packed_Cx2_A_Bx2(
    x2::TensorAsMatrixView<Scalar, rankC>            &C_hi, //
    x2::TensorAsMatrixView<Scalar, rankC>            &C_lo, //
    const x2::ConstTensorAsMatrixView<Scalar, rankA> &A,    //
    const x2::ConstTensorAsMatrixView<Scalar, rankB> &B_hi, //
    const x2::ConstTensorAsMatrixView<Scalar, rankB> &B_lo)
{
    constexpr Eigen::Index IB = 64;
    constexpr Eigen::Index JB = 8;
    constexpr Eigen::Index BK = 128;

    const Eigen::Index m = A.rows();
    const Eigen::Index k = A.cols();
    const Eigen::Index n = B_hi.cols();

    assert(A.rows() == m && A.cols() == k);
    assert(B_hi.rows() == k && B_hi.cols() == n);
    assert(B_lo.rows() == k && B_lo.cols() == n);
    assert(C_hi.rows() == m && C_hi.cols() == n);
    assert(C_lo.rows() == m && C_lo.cols() == n);

#pragma omp parallel
    {
        x2::MatrixCursor<Scalar, rankA, x2::Access::ReadOnly>     Acur(A);
        x2::X2MatrixCursor<Scalar, rankB, x2::Access::ReadOnly>  Bcur(B_hi, B_lo);
        x2::X2MatrixCursor<Scalar, rankC, x2::Access::ReadWrite> Ccur(C_hi, C_lo);

        alignas(64) Scalar sum_hi_buf[JB * IB];
        alignas(64) Scalar sum_lo_buf[JB * IB];

        alignas(64) Scalar Apack[BK * IB];    // [kk * IB + i]
        alignas(64) Scalar Bpack_hi[BK * JB]; // [kk * JB + j]
        alignas(64) Scalar Bpack_lo[BK * JB]; // [kk * JB + j]

#pragma omp for schedule(static)
        for(Eigen::Index j0 = 0; j0 < n; j0 += JB) {
            const Eigen::Index jb = std::min<Eigen::Index>(JB, n - j0);

            for(Eigen::Index k0 = 0; k0 < k; k0 += BK) {
                const Eigen::Index kb = std::min<Eigen::Index>(BK, k - k0);

                // ---- PACK B_hi/B_lo ONCE for this (j0,k0): (kb x jb) -> Bpack[kk][j] ----
                for(Eigen::Index kk = 0; kk < kb; ++kk) {
                    Scalar            *bh  = Bpack_hi + kk * JB;
                    Scalar            *bl  = Bpack_lo + kk * JB;
                    const Eigen::Index row = k0 + kk;

                    Bcur.init(row, j0);
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        bh[j] = *Bcur.ptr_hi();
                        bl[j] = *Bcur.ptr_lo();
                        if(j + 1 < jb) { Bcur.stepCol(); }
                    }
                }

                // ---- REUSE packed B for all i0 blocks ----
                for(Eigen::Index i0 = 0; i0 < m; i0 += IB) {
                    const Eigen::Index ib = std::min<Eigen::Index>(IB, m - i0);

                    // zero partial sums for THIS (i0,k0) update
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        std::fill_n(sum_hi_buf + j * IB, ib, Scalar(0));
                        std::fill_n(sum_lo_buf + j * IB, ib, Scalar(0));
                    }

                    // ---- PACK A ONCE for this (i0,k0): (ib x kb) -> Apack[kk][i] ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        Scalar            *ap  = Apack + kk * IB;
                        const Eigen::Index col = k0 + kk;

                        Acur.init(i0, col);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            ap[i] = *Acur.ptr();
                            if(i + 1 < ib) { Acur.stepRow(); }
                        }
                    }

                    // ---- COMPUTE partial update ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Scalar *ap = Apack + kk * IB;
                        const Scalar *bh = Bpack_hi + kk * JB;
                        const Scalar *bl = Bpack_lo + kk * JB;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Scalar bhj = bh[j];
                            const Scalar blj = bl[j];

                            Scalar *__restrict sh = sum_hi_buf + j * IB;
                            Scalar *__restrict sl = sum_lo_buf + j * IB;

                            for(Eigen::Index i = 0; i < ib; ++i) {
                                const Scalar api = ap[i];

                                const Scalar p_hi = api * bhj;
                                const Scalar p_lo = x2_detail::mul_residual(api, bhj, p_hi);
                                x2_detail::accumulate(sh[i], sl[i], p_hi, p_lo);

                                // first-order cross term: api * blj
                                if constexpr(std::is_floating_point_v<Scalar>) {
                                    sl[i] = std::fma(api, blj, sl[i]);
                                } else {
                                    sl[i] += api * blj;
                                }
                            }
                        }
                    }

                    // ---- STORE/ACCUMULATE into C for this k0-panel ----
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        const Scalar *__restrict sh = sum_hi_buf + j * IB;
                        const Scalar *__restrict sl = sum_lo_buf + j * IB;

                        Ccur.init(i0, j0 + j);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            Scalar s, e;

                            if(k0 == 0) {
                                x2_detail::add_with_residual(sh[i], sl[i], s, e);
                            } else {
                                Scalar ch = *Ccur.ptr_hi();
                                Scalar cl = *Ccur.ptr_lo();

                                x2_detail::accumulate(ch, cl, sh[i], sl[i]);
                                x2_detail::add_with_residual(ch, cl, s, e);
                            }

                            *Ccur.ptr_hi() = s;
                            *Ccur.ptr_lo() = e;

                            if(i + 1 < ib) { Ccur.stepRow(); }
                        }
                    }
                }
            }
        }
    }
}


template<typename Scalar, int rankC, int rankA, int rankB>
void gemm_x2_kernel_fused_packed_Cx2_Ax2_B(
    x2::TensorAsMatrixView<Scalar, rankC>            &C_hi, //
    x2::TensorAsMatrixView<Scalar, rankC>            &C_lo, //
    const x2::ConstTensorAsMatrixView<Scalar, rankA> &A_hi, //
    const x2::ConstTensorAsMatrixView<Scalar, rankA> &A_lo, //
    const x2::ConstTensorAsMatrixView<Scalar, rankB> &B)
{
    constexpr Eigen::Index IB = 64;
    constexpr Eigen::Index JB = 8;
    constexpr Eigen::Index BK = 128;

    const Eigen::Index m = A_hi.rows();
    const Eigen::Index k = A_hi.cols();
    const Eigen::Index n = B.cols();

    assert(A_lo.rows() == m && A_lo.cols() == k);
    assert(B.rows() == k && B.cols() == n);
    assert(C_hi.rows() == m && C_hi.cols() == n);
    assert(C_lo.rows() == m && C_lo.cols() == n);

#pragma omp parallel
    {
        x2::X2MatrixCursor<Scalar, rankA, x2::Access::ReadOnly>  Acur(A_hi, A_lo);
        x2::MatrixCursor<Scalar, rankB, x2::Access::ReadOnly>    Bcur(B);
        x2::X2MatrixCursor<Scalar, rankC, x2::Access::ReadWrite> Ccur(C_hi, C_lo);

        alignas(64) Scalar sum_hi_buf[JB * IB];
        alignas(64) Scalar sum_lo_buf[JB * IB];

        alignas(64) Scalar Apack_hi[BK * IB];
        alignas(64) Scalar Apack_lo[BK * IB];
        alignas(64) Scalar Bpack[BK * JB];

#pragma omp for schedule(static)
        for(Eigen::Index j0 = 0; j0 < n; j0 += JB) {
            const Eigen::Index jb = std::min<Eigen::Index>(JB, n - j0);

            for(Eigen::Index k0 = 0; k0 < k; k0 += BK) {
                const Eigen::Index kb = std::min<Eigen::Index>(BK, k - k0);

                // ---- PACK B ONCE for this (j0,k0): (kb x jb) -> Bpack[kk][j] ----
                for(Eigen::Index kk = 0; kk < kb; ++kk) {
                    Scalar            *bp  = Bpack + kk * JB;
                    const Eigen::Index row = k0 + kk;

                    Bcur.init(row, j0);
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        bp[j] = *Bcur.ptr();
                        if(j + 1 < jb) { Bcur.stepCol(); }
                    }
                }

                // ---- REUSE packed B for all i0 blocks ----
                for(Eigen::Index i0 = 0; i0 < m; i0 += IB) {
                    const Eigen::Index ib = std::min<Eigen::Index>(IB, m - i0);

                    // zero partial sums for THIS (i0,k0) update
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        std::fill_n(sum_hi_buf + j * IB, ib, Scalar(0));
                        std::fill_n(sum_lo_buf + j * IB, ib, Scalar(0));
                    }

                    // ---- PACK A_hi/A_lo for this (i0,k0): (ib x kb) -> Apack[kk][i] ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        Scalar            *ah  = Apack_hi + kk * IB;
                        Scalar            *al  = Apack_lo + kk * IB;
                        const Eigen::Index col = k0 + kk;

                        Acur.init(i0, col);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            ah[i] = *Acur.ptr_hi();
                            al[i] = *Acur.ptr_lo();
                            if(i + 1 < ib) { Acur.stepRow(); }
                        }
                    }

                    // ---- COMPUTE partial update ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Scalar *ah = Apack_hi + kk * IB;
                        const Scalar *al = Apack_lo + kk * IB;
                        const Scalar *bp = Bpack + kk * JB;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Scalar bpj = bp[j];

                            Scalar *__restrict sh = sum_hi_buf + j * IB;
                            Scalar *__restrict sl = sum_lo_buf + j * IB;

                            for(Eigen::Index i = 0; i < ib; ++i) {
                                const Scalar ahi = ah[i];
                                const Scalar ali = al[i];

                                const Scalar p_hi = ahi * bpj;
                                const Scalar p_lo = x2_detail::mul_residual(ahi, bpj, p_hi);
                                x2_detail::accumulate(sh[i], sl[i], p_hi, p_lo);

                                // first-order cross term: ali * bpj
                                if constexpr(std::is_floating_point_v<Scalar>) {
                                    sl[i] = std::fma(ali, bpj, sl[i]);
                                } else {
                                    sl[i] += ali * bpj;
                                }
                            }
                        }
                    }

                    // ---- STORE/ACCUMULATE into C for this k0-panel ----
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        const Scalar *__restrict sh = sum_hi_buf + j * IB;
                        const Scalar *__restrict sl = sum_lo_buf + j * IB;

                        Ccur.init(i0, j0 + j);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            Scalar s, e;

                            if(k0 == 0) {
                                x2_detail::add_with_residual(sh[i], sl[i], s, e);
                            } else {
                                Scalar ch = *Ccur.ptr_hi();
                                Scalar cl = *Ccur.ptr_lo();

                                x2_detail::accumulate(ch, cl, sh[i], sl[i]);
                                x2_detail::add_with_residual(ch, cl, s, e);
                            }

                            *Ccur.ptr_hi() = s;
                            *Ccur.ptr_lo() = e;

                            if(i + 1 < ib) { Ccur.stepRow(); }
                        }
                    }
                }
            }
        }
    }
}

template<typename Scalar, int rankC, int rankA, int rankB>
void gemm_x2_kernel_fused_packed_Cx2_A_B(
    x2::TensorAsMatrixView<Scalar, rankC>            &C_hi, //
    x2::TensorAsMatrixView<Scalar, rankC>            &C_lo, //
    const x2::ConstTensorAsMatrixView<Scalar, rankA> &A,    //
    const x2::ConstTensorAsMatrixView<Scalar, rankB> &B)
{
    constexpr Eigen::Index IB = 64;
    constexpr Eigen::Index JB = 8;
    constexpr Eigen::Index BK = 128;

    const Eigen::Index m = A.rows();
    const Eigen::Index k = A.cols();
    const Eigen::Index n = B.cols();

    assert(C_hi.rows() == m && C_hi.cols() == n);
    assert(C_lo.rows() == m && C_lo.cols() == n);
    assert(B.rows() == k);

#pragma omp parallel
    {
        x2::MatrixCursor<Scalar, rankA, x2::Access::ReadOnly>     Acur(A);
        x2::MatrixCursor<Scalar, rankB, x2::Access::ReadOnly>     Bcur(B);
        x2::X2MatrixCursor<Scalar, rankC, x2::Access::ReadWrite>  Ccur(C_hi, C_lo);

        alignas(64) Scalar sum_hi_buf[JB * IB];
        alignas(64) Scalar sum_lo_buf[JB * IB];

        alignas(64) Scalar Apack[BK * IB];
        alignas(64) Scalar Bpack[BK * JB];

#pragma omp for schedule(static)
        for(Eigen::Index j0 = 0; j0 < n; j0 += JB) {
            const Eigen::Index jb = std::min<Eigen::Index>(JB, n - j0);

            for(Eigen::Index k0 = 0; k0 < k; k0 += BK) {
                const Eigen::Index kb = std::min<Eigen::Index>(BK, k - k0);

                // ---- PACK B ONCE for this (j0,k0): (kb x jb) -> Bpack[kk][j] ----
                for(Eigen::Index kk = 0; kk < kb; ++kk) {
                    Scalar            *bp  = Bpack + kk * JB;
                    const Eigen::Index row = k0 + kk;

                    Bcur.init(row, j0);
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        bp[j] = *Bcur.ptr();
                        if(j + 1 < jb) { Bcur.stepCol(); }
                    }
                }

                // ---- REUSE packed B for all i0 blocks ----
                for(Eigen::Index i0 = 0; i0 < m; i0 += IB) {
                    const Eigen::Index ib = std::min<Eigen::Index>(IB, m - i0);

                    // zero partial sums for THIS (i0,k0) update
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        std::fill_n(sum_hi_buf + j * IB, ib, Scalar(0));
                        std::fill_n(sum_lo_buf + j * IB, ib, Scalar(0));
                    }

                    // ---- PACK A ONCE for this (i0,k0): (ib x kb) -> Apack[kk][i] ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        Scalar            *ap  = Apack + kk * IB;
                        const Eigen::Index col = k0 + kk;

                        Acur.init(i0, col);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            ap[i] = *Acur.ptr();
                            if(i + 1 < ib) { Acur.stepRow(); }
                        }
                    }

                    // ---- COMPUTE partial update ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Scalar *ap = Apack + kk * IB;
                        const Scalar *bp = Bpack + kk * JB;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Scalar bpj = bp[j];

                            Scalar *__restrict sh = sum_hi_buf + j * IB;
                            Scalar *__restrict sl = sum_lo_buf + j * IB;

                            for(Eigen::Index i = 0; i < ib; ++i) {
                                const Scalar api = ap[i];

                                const Scalar p_hi = api * bpj;
                                const Scalar p_lo = x2_detail::mul_residual(api, bpj, p_hi);
                                x2_detail::accumulate(sh[i], sl[i], p_hi, p_lo);
                            }
                        }
                    }

                    // ---- STORE/ACCUMULATE into C for this k0-panel ----
                    for(Eigen::Index j = 0; j < jb; ++j) {
                        const Scalar *__restrict sh = sum_hi_buf + j * IB;
                        const Scalar *__restrict sl = sum_lo_buf + j * IB;

                        Ccur.init(i0, j0 + j);
                        for(Eigen::Index i = 0; i < ib; ++i) {
                            Scalar s, e;

                            if(k0 == 0) {
                                x2_detail::add_with_residual(sh[i], sl[i], s, e);
                            } else {
                                Scalar ch = *Ccur.ptr_hi();
                                Scalar cl = *Ccur.ptr_lo();

                                x2_detail::accumulate(ch, cl, sh[i], sl[i]);
                                x2_detail::add_with_residual(ch, cl, s, e);
                            }

                            *Ccur.ptr_hi() = s;
                            *Ccur.ptr_lo() = e;

                            if(i + 1 < ib) { Ccur.stepRow(); }
                        }
                    }
                }
            }
        }
    }
}



