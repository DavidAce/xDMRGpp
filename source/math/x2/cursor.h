#pragma once
#include "util.h"
#include <array>
#include <cassert>
#include <Eigen/Core>
namespace x2_detail {

    template<int rank>
    struct AxisCursorCore {
        using Index = Eigen::Index;

        int                     axes_rank = 0;
        std::array<Index, rank> dim{};   // dim[t] for t in [0,axes_rank)
        std::array<Index, rank> st{};    // stride[t] for t in [0,axes_rank)
        std::array<Index, rank> wrap{};  // st[t] * dim[t]
        std::array<Index, rank> idx{};   // current coord (only first axes_rank used)
        Index                   off = 0; // current offset contribution

        AxisCursorCore() = default;

        template<class View>
        AxisCursorCore(const View &v, const std::array<int, rank> &axes, int ar) {
            axes_rank = ar;
            for(int t = 0; t < axes_rank; ++t) {
                const int   ax  = axes[size_t(t)];
                const Index d   = v.dims[size_t(ax)];
                const Index s   = v.stride[size_t(ax)];
                dim[size_t(t)]  = d;
                st[size_t(t)]   = s;
                wrap[size_t(t)] = s * d;
            }
        }

        EIGEN_STRONG_INLINE void init_from_linear(Index lin) {
            off = 0;
            for(int t = 0; t < axes_rank; ++t) {
                const Index d = dim[size_t(t)];
                assert(d > 0);
                const Index q  = lin / d;
                const Index r  = lin - q * d; // avoids % if you care
                idx[size_t(t)] = r;
                lin            = q;
                off += r * st[size_t(t)];
            }
            // if you want, you can assert(lin==0) in debug for in-range checks
        }

        EIGEN_STRONG_INLINE void step() {
            // increment least-significant axis = t=0
            for(int t = 0; t < axes_rank; ++t) {
                idx[size_t(t)] += 1;
                off += st[size_t(t)];
                if(idx[size_t(t)] < dim[size_t(t)]) return;

                // carry
                idx[size_t(t)] = 0;
                off -= wrap[size_t(t)];
            }
            // if axes_rank==0, this just does nothing
        }
    };

} // namespace x2_detail

namespace x2 {

    // Forward declare your view types
    template<typename Scalar_, int rank, Access A>
    struct TensorAsMatrixViewT;

    template<typename Scalar_, int rank, Access A>
    struct X2TensorAsMatrixViewT;

    namespace detail {
        template<int rank>
        struct MatrixCursorState {
            using Index = Eigen::Index;

            std::array<Index, rank> dims{};
            std::array<Index, rank> stride{};
            std::array<int, rank>   row_axes{};
            std::array<int, rank>   col_axes{};
            int                     row_rank = 0;
            int                     col_rank = 0;

            std::array<Index, rank> coord{}; // current tensor coordinates for each axis
            Index                   off = 0; // current linear offset (col-major)

            EIGEN_STRONG_INLINE void init_from_linear(Index r, Index c) {
                coord.fill(0);
                off = 0;

                // decode r into row_axes (least significant axis first)
                {
                    Index tmp = r;
                    for(int t = 0; t < row_rank; ++t) {
                        const int   ax  = row_axes[size_t(t)];
                        const Index dim = dims[size_t(ax)];
                        const Index v   = tmp % dim;
                        tmp /= dim;
                        coord[size_t(ax)] = v;
                        off += v * stride[size_t(ax)];
                    }
                    assert(tmp == 0);
                }

                // decode c into col_axes
                {
                    Index tmp = c;
                    for(int t = 0; t < col_rank; ++t) {
                        const int   ax  = col_axes[size_t(t)];
                        const Index dim = dims[size_t(ax)];
                        const Index v   = tmp % dim;
                        tmp /= dim;
                        coord[size_t(ax)] = v;
                        off += v * stride[size_t(ax)];
                    }
                    assert(tmp == 0);
                }
            }

            // increment along a mixed-radix counter for the given axes, updating off
            EIGEN_STRONG_INLINE void step_axes(const std::array<int, rank> &axes, int axes_rank) {
                for(int t = 0; t < axes_rank; ++t) {
                    const int   ax  = axes[size_t(t)];
                    auto       &cv  = coord[size_t(ax)];
                    const Index dim = dims[size_t(ax)];
                    const Index st  = stride[size_t(ax)];

                    cv += 1;
                    off += st;

                    if(cv < dim) return;

                    // wrap this axis back to 0 and carry
                    cv = 0;
                    off -= dim * st;
                }
                // if we get here you stepped past the end of the row/col range
                assert(false);
            }

            EIGEN_STRONG_INLINE void stepRow() { step_axes(row_axes, row_rank); }
            EIGEN_STRONG_INLINE void stepCol() { step_axes(col_axes, col_rank); }
        };
    }

    // Single-buffer cursor
    template<typename Scalar_, int rank, Access A>
    struct MatrixCursor {
        using Scalar = Scalar_;
        using Index  = Eigen::Index;
        using Ptr    = std::conditional_t<A == Access::ReadOnly, const Scalar *, Scalar *>;

        Ptr                             hi_base = nullptr; // name chosen to match ptr_hi() usage style
        detail::MatrixCursorState<rank> st;

        MatrixCursor() = default;

        MatrixCursor(const TensorAsMatrixViewT<Scalar, rank, A> &V) {
            V.assert_valid();
            assert(V.base != nullptr);
            hi_base     = V.base;
            st.dims     = V.dims;
            st.stride   = V.stride;
            st.row_axes = V.row_axes;
            st.col_axes = V.col_axes;
            st.row_rank = V.row_rank;
            st.col_rank = V.col_rank;
        }

        EIGEN_STRONG_INLINE void init(Index r, Index c) { st.init_from_linear(r, c); }

        EIGEN_STRONG_INLINE Ptr ptr() const { return hi_base + st.off; }

        EIGEN_STRONG_INLINE void stepRow() { st.stepRow(); }
        EIGEN_STRONG_INLINE void stepCol() { st.stepCol(); }
    };

    // Dual-buffer cursor
    template<typename Scalar_, int rank, Access A>
    struct X2MatrixCursor {
        using Scalar = Scalar_;
        using Index  = Eigen::Index;

        using PtrHi = std::conditional_t<A == Access::ReadOnly, const Scalar *, Scalar *>;
        using PtrLo = std::conditional_t<A == Access::ReadOnly, const Scalar *, Scalar *>;

        PtrHi hi_base = nullptr;
        PtrLo lo_base = nullptr;

        detail::MatrixCursorState<rank> st;

        X2MatrixCursor() = default;

        // Construct from separate hi/lo matrix views (must share identical mapping)
        X2MatrixCursor(const TensorAsMatrixViewT<Scalar, rank, A> &hi, const TensorAsMatrixViewT<Scalar, rank, A> &lo) {
#ifndef NDEBUG
            hi.assert_valid();
            lo.assert_valid();
            assert(hi.dims == lo.dims);
            assert(hi.stride == lo.stride);
            assert(hi.row_rank == lo.row_rank && hi.col_rank == lo.col_rank);
            for(int t = 0; t < rank; ++t) {
                assert(hi.row_axes[size_t(t)] == lo.row_axes[size_t(t)]);
                assert(hi.col_axes[size_t(t)] == lo.col_axes[size_t(t)]);
            }
            assert(hi.base != nullptr && lo.base != nullptr);
            assert(hi.base != lo.base);
#endif
            hi_base = hi.base;
            lo_base = lo.base;

            st.dims     = hi.dims;
            st.stride   = hi.stride;
            st.row_axes = hi.row_axes;
            st.col_axes = hi.col_axes;
            st.row_rank = hi.row_rank;
            st.col_rank = hi.col_rank;
        }

        // Convenience: construct from X2TensorAsMatrixViewT
        X2MatrixCursor(const X2TensorAsMatrixViewT<Scalar, rank, A> &V) : X2MatrixCursor(V.hi, V.lo) {}

        EIGEN_STRONG_INLINE void init(Index r, Index c) { st.init_from_linear(r, c); }

        EIGEN_STRONG_INLINE PtrHi ptr_hi() const { return hi_base + st.off; }
        EIGEN_STRONG_INLINE PtrLo ptr_lo() const { return lo_base + st.off; }

        EIGEN_STRONG_INLINE void stepRow() { st.stepRow(); }
        EIGEN_STRONG_INLINE void stepCol() { st.stepCol(); }
    };

}