#pragma once
#include "util.h"
#include <array>
#include <cassert>
#include <Eigen/Core>
#include <initializer_list>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>

namespace x2 {

    // A lightweight, non-owning view that interprets an Eigen::Tensor buffer as a matrix:
    //   rows  = product(dims[row_axes...])
    //   cols  = product(dims[col_axes...])
    //

    // Convention: within row_axes (and col_axes), axis[0] is the fastest-varying (least significant)
    // in the mixed-radix encoding, matching "shuffle + reshape" with Eigen::Tensor col-major storage.
    template<typename Scalar_, int rank, Access A>
    struct TensorAsMatrixViewT {
        using Scalar = Scalar_;
        using Index  = Eigen::Index;
        using Ptr    = std::conditional_t<A == Access::ReadOnly, const Scalar *, Scalar *>;

        Ptr base = nullptr;

        std::array<Index, rank> dims{};
        std::array<Index, rank> stride{};
        std::array<int, rank>   row_axes{};
        std::array<int, rank>   col_axes{};
        int                     row_rank = 0;
        int                     col_rank = 0;

        Index rows() const {
            Index r = 1;
            for(int t = 0; t < row_rank; ++t) r *= dims[size_t(row_axes[size_t(t)])];
            return r;
        }
        Index cols() const {
            Index c = 1;
            for(int t = 0; t < col_rank; ++t) c *= dims[size_t(col_axes[size_t(t)])];
            return c;
        }

        void assert_valid() const {
#if !defined(NDEBUG)
            assert(base != nullptr);

            // axes must be in [0,rank)
            for(int t = 0; t < row_rank; ++t) assert(0 <= row_axes[static_cast<size_t>(t)] && row_axes[static_cast<size_t>(t)] < rank);
            for(int t = 0; t < col_rank; ++t) assert(0 <= col_axes[static_cast<size_t>(t)] && col_axes[static_cast<size_t>(t)] < rank);

            // row_axes and col_axes should be disjoint and cover all axes exactly once for a pure reshape
            std::array<int, rank> seen{};
            seen.fill(0);
            for(int t = 0; t < row_rank; ++t) seen[static_cast<size_t>(row_axes[static_cast<size_t>(t)])] += 1;
            for(int t = 0; t < col_rank; ++t) seen[static_cast<size_t>(col_axes[static_cast<size_t>(t)])] += 1;
            for(int ax = 0; ax < rank; ++ax) assert(seen[static_cast<size_t>(ax)] == 1);

            // basic stride sanity for col-major contiguous tensors
            assert(stride[0] == 1);
            for(int ax = 1; ax < rank; ++ax) {
                assert(stride[static_cast<size_t>(ax)] == stride[static_cast<size_t>(ax - 1)] * dims[static_cast<size_t>(ax - 1)]);
            }
#endif
        }

        Index offset(Index r, Index c) const {
            // Decode r into indices along row_axes
            std::array<Index, rank> idx{};
            {
                Index tmp = r;
                for(int t = 0; t < row_rank; ++t) {
                    const int   ax  = row_axes[static_cast<size_t>(t)];
                    const Index dim = dims[static_cast<size_t>(ax)];
                    assert(dim > 0);
                    idx[static_cast<size_t>(ax)]  = tmp % dim;
                    tmp                          /= dim;
                }
                assert(tmp == 0); // r in-range for computed rows()
            }

            // Decode c into indices along col_axes
            {
                Index tmp = c;
                for(int t = 0; t < col_rank; ++t) {
                    const int   ax  = col_axes[static_cast<size_t>(t)];
                    const Index dim = dims[static_cast<size_t>(ax)];
                    assert(dim > 0);
                    idx[static_cast<size_t>(ax)]  = tmp % dim;
                    tmp                          /= dim;
                }
                assert(tmp == 0); // c in-range for computed cols()
            }

            // Compute tensor linear offset using col-major tensor strides
            Index off = 0;
            for(int ax = 0; ax < rank; ++ax) off += idx[static_cast<size_t>(ax)] * stride[static_cast<size_t>(ax)];
            return off;
        }

        const Scalar *ptr(Index r, Index c) const { return base + offset(r, c); }
        Scalar       *ptr(Index r, Index c) requires(A == Access::ReadWrite)
        {
            return base + offset(r, c);
        }

        const Scalar &operator()(Index r, Index c) const { return *ptr(r, c); }
        Scalar       &operator()(Index r, Index c) requires(A == Access::ReadWrite)
        {
            return *ptr(r, c);
        }

        Index size() { return x2_detail::size_from_dims(dims); }
    };

    template<typename Scalar_, int rank, Access A>
    struct X2TensorAsMatrixViewT {
        using Scalar = Scalar_;
        using Index  = Eigen::Index;

        TensorAsMatrixViewT<Scalar, rank, A> hi;
        TensorAsMatrixViewT<Scalar, rank, A> lo;

        Index rows() const {
            Index r = hi.rows();
            assert(r == lo.rows());
            return r;
        }
        Index cols() const {
            Index c = hi.cols();
            assert(c == lo.cols());
            return c;
        }
        Index size() const {
            assert(hi.dims == lo.dims);
            return x2_detail::size_from_dims(hi.dims);
        }

        void assert_valid() const {
            hi.assert_valid();
            lo.assert_valid();
            assert(hi.dims == lo.dims);
            assert(hi.base != nullptr && lo.base != nullptr);
            assert(hi.base != lo.base);
        }

        void renorm() requires(A == Access::ReadWrite)
        {
            assert_valid();
            x2_detail::renorm(hi.base, lo.base, size());
        }
    };

    // Friendly aliases
    template<typename Scalar, int rank>
    using TensorAsMatrixView = TensorAsMatrixViewT<Scalar, rank, Access::ReadWrite>;
    template<typename Scalar, int rank>
    using ConstTensorAsMatrixView = TensorAsMatrixViewT<Scalar, rank, Access::ReadOnly>;

    template<typename Scalar, int rank>
    using X2TensorAsMatrixView = X2TensorAsMatrixViewT<Scalar, rank, Access::ReadWrite>;
    template<typename Scalar, int rank>
    using ConstX2TensorAsMatrixView = X2TensorAsMatrixViewT<Scalar, rank, Access::ReadOnly>;
    namespace internal {
        template<typename Scalar, int rank, x2::Access A>
        inline x2::TensorAsMatrixViewT<Scalar, rank, A> make_view(typename x2::TensorAsMatrixViewT<Scalar, rank, A>::Ptr data,
                                                                  const Eigen::array<Eigen::Index, rank> &dims, std::initializer_list<int> row_axes,
                                                                  std::initializer_list<int> col_axes) {
            x2::TensorAsMatrixViewT<Scalar, rank, A> V;
            V.base = data;
            for(int i = 0; i < rank; ++i) V.dims[static_cast<size_t>(i)] = dims[static_cast<size_t>(i)];
            V.stride = x2_detail::colmajor_strides<rank>(V.dims);
            x2_detail::fill_axes<rank>(V.row_axes, V.row_rank, row_axes);
            x2_detail::fill_axes<rank>(V.col_axes, V.col_rank, col_axes);
            V.assert_valid();
            return V;
        }
    }
    template<typename Scalar, int rank>
    inline TensorAsMatrixView<Scalar, rank> as_matrix(Scalar *data, const Eigen::array<Eigen::Index, rank> &dims, std::initializer_list<int> row_axes,
                                                      std::initializer_list<int> col_axes) {
        return internal::make_view<Scalar, rank, Access::ReadWrite>(data, dims, row_axes, col_axes);
    }

    template<typename Scalar, int rank>
    inline ConstTensorAsMatrixView<Scalar, rank> as_const_matrix(const Scalar *const data, const Eigen::array<Eigen::Index, rank> &dims,
                                                                 std::initializer_list<int> row_axes, std::initializer_list<int> col_axes) {
        return internal::make_view<Scalar, rank, Access::ReadOnly>(data, dims, row_axes, col_axes);
    }

    template<typename Scalar, int rank>
    inline TensorAsMatrixView<Scalar, rank> as_matrix(Eigen::Tensor<Scalar, rank> &T, std::initializer_list<int> row_axes,
                                                      std::initializer_list<int> col_axes) {
        return internal::make_view<Scalar, rank, Access::ReadWrite>(T.data(), T.dimensions(), row_axes, col_axes);
    }
    template<typename Scalar, int rank>
    inline ConstTensorAsMatrixView<Scalar, rank> as_const_matrix(const Eigen::Tensor<Scalar, rank> &T, std::initializer_list<int> row_axes,
                                                                 std::initializer_list<int> col_axes) {
        return internal::make_view<Scalar, rank, Access::ReadOnly>(T.data(), T.dimensions(), row_axes, col_axes);
    }

    template<typename Scalar, int rank>
    inline TensorAsMatrixView<Scalar, rank> as_matrix(Eigen::TensorMap<Eigen::Tensor<Scalar, rank>> &T, std::initializer_list<int> row_axes,
                                                      std::initializer_list<int> col_axes) {
        return internal::make_view<Scalar, rank, Access::ReadWrite>(T.data(), T.dimensions(), row_axes, col_axes);
    }
    template<typename Scalar, int rank>
    inline ConstTensorAsMatrixView<Scalar, rank> as_const_matrix(const Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>> &T,
                                                                 std::initializer_list<int> row_axes, std::initializer_list<int> col_axes) {
        return internal::make_view<Scalar, rank, Access::ReadOnly>(T.data(), T.dimensions(), row_axes, col_axes);
    }

    // If you want the x2 (hi, lo) pair in one return object:
    // This expects your x2::Tensor exposes hi_data()/lo_data() and dimensions() returning Eigen::array<Index,rank>.
    template<typename Scalar, int rank, typename X2Tensor>
    inline X2TensorAsMatrixView<Scalar, rank> as_matrix_x2(X2Tensor &T, std::initializer_list<int> row_axes, std::initializer_list<int> col_axes) {
        X2TensorAsMatrixView<Scalar, rank> V;
        V.hi = as_matrix<Scalar, rank>(T.hi_data(), T.dimensions(), row_axes, col_axes);
        V.lo = as_matrix<Scalar, rank>(T.lo_data(), T.dimensions(), row_axes, col_axes);
        V.assert_valid();
        return V;
    }
    template<typename Scalar, int rank, typename X2Tensor>
    inline ConstX2TensorAsMatrixView<Scalar, rank> as_const_matrix_x2(const X2Tensor &T, std::initializer_list<int> row_axes,
                                                                      std::initializer_list<int> col_axes) {
        ConstX2TensorAsMatrixView<Scalar, rank> V;
        V.hi = as_const_matrix<Scalar, rank>(T.hi_data(), T.dimensions(), row_axes, col_axes);
        V.lo = as_const_matrix<Scalar, rank>(T.lo_data(), T.dimensions(), row_axes, col_axes);
        V.assert_valid();
        return V;
    }
}
