#pragma once
#ifndef lapack_complex_float
    #define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
    #define lapack_complex_double std::complex<double>
#endif

#include "math/cast.h"
#include "math/float.h"
#include "tenx/eval.h"
#include "tenx/sfinae.h"
#include "tenx/span.h"
#include "tenx/threads.h"
#include <array>
#include <complex>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace Eigen {
#if defined(DMRG_USE_QUADMATH) || defined(DMRG_USE_FLOAT128)
    template<>
    struct NumTraits<fp128> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        typedef fp128 Real;
        typedef fp128 NonInteger;
        typedef fp128 Nested;

        enum { IsComplex = 0, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 1, AddCost = 3, MulCost = 3 };
    };

    template<>
    struct NumTraits<cx128> : NumTraits<cx64> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        typedef fp128 Real;
        typedef fp128 NonInteger;
        typedef fp128 Nested;
        enum { IsComplex = 1, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 1, AddCost = 6, MulCost = 6 };
    };
#endif
}

/*! \brief **tenx**: "Tensor Extra". Provides extra functionality to Eigen::Tensor.*/

/*!
 *  \namespace tenx
 *  This namespace makes shorthand typedef's to Eigen's unsupported Tensor module, and provides handy functions
 *  to interface between `Eigen::Tensor` and `Eigen::Matrix` objects.
 *  The contents of this namespace is co clear it is self-documenting ;)
 */

/*clang-format off */

namespace tenx {
    template<typename Scalar>
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename Scalar>
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    template<Eigen::Index rank>
    using array = std::array<Eigen::Index, rank>;

    template<typename Scalar>
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    template<typename Derived>
    using epsilon_t = RealScalar<typename Derived::Scalar>;
    template<typename Derived>
    constexpr auto epsilon() {
        return std::numeric_limits<epsilon_t<Derived>>::epsilon();
    }

    using array8 = array<8>;
    using array7 = array<7>;
    using array6 = array<6>;
    using array5 = array<5>;
    using array4 = array<4>;
    using array3 = array<3>;
    using array2 = array<2>;
    using array1 = array<1>;
    using array0 = array<0>;

    // Shorthand for the list of index pairs.
    template<auto N>
    using idxlistpair = std::array<Eigen::IndexPair<Eigen::Index>, N>;

    inline constexpr idxlistpair<0> idx() { return {}; }

    template<std::size_t N, typename idxType>
    constexpr idxlistpair<N> idx(const std::array<idxType, N> &list1, const std::array<idxType, N> &list2) {
        /*! Use numpy-style indexing for contraction. Each list contains a list of indices to be contracted for the respective
         * tensors. This function zips them together into pairs as used in Eigen::Tensor module. This does not sort the indices in decreasing order.
         */
        idxlistpair<N> pairlistOut;
        for(size_t i = 0; i < N; i++) {
            if constexpr(std::is_signed_v<idxType>) {
                assert(list1[i] >= 0);
                assert(list2[i] >= 0);
            }
            pairlistOut[i] = Eigen::IndexPair<Eigen::Index>{
                static_cast<Eigen::Index>(list1[i]), //
                static_cast<Eigen::Index>(list2[i])  //
            };
        }
        return pairlistOut;
    }

    template<std::size_t N, typename idxType>
    constexpr idxlistpair<N> idx(const idxType (&list1)[N], const idxType (&list2)[N]) {
        /*! Use numpy-style indexing for contraction. Each list contains a list of indices to be contracted for the respective
         * tensors. This function zips them together into pairs as used in Eigen::Tensor module. This does not sort the indices in decreasing order.
         */
        idxlistpair<N> pairlistOut;
        for(size_t i = 0; i < N; i++) {
            if constexpr(std::is_signed_v<idxType>) {
                assert(list1[i] >= 0);
                assert(list2[i] >= 0);
            }
            pairlistOut[i] = Eigen::IndexPair<Eigen::Index>{
                static_cast<Eigen::Index>(list1[i]), //
                static_cast<Eigen::Index>(list2[i])  //
            };
        }
        return pairlistOut;
    }

    struct idx_dim_pair {
        Eigen::Index idxA;
        Eigen::Index idxB;
        Eigen::Index dimB;
    };

    template<std::size_t NB, std::size_t N>
    constexpr idxlistpair<N> sortIdx(const std::array<Eigen::Index, NB> &dimensions, const Eigen::Index (&idx_ctrct_A)[N],
                                     const Eigen::Index (&idx_ctrct_B)[N]) {
        // When doing contractions, some indices may be larger than others. For performance, you want to
        // contract the largest indices first. This will return a sorted index list in decreasing order.
        std::array<idx_dim_pair, N> idx_dim_pair_list;
        for(size_t i = 0; i < N; i++) { idx_dim_pair_list[i] = {idx_ctrct_A[i], idx_ctrct_B[i], dimensions[idx_ctrct_B[i]]}; }
        std::sort(idx_dim_pair_list.begin(), idx_dim_pair_list.end(), [](const auto &i, const auto &j) { return i.dimB > j.dimB; });
        idxlistpair<N> pairlistOut;
        for(size_t i = 0; i < N; i++) { pairlistOut[i] = Eigen::IndexPair<long>{idx_dim_pair_list[i].idxA, idx_dim_pair_list[i].idxB}; }
        return pairlistOut;
    }

    //
    //    //***************************************//
    //    //Different views for rank 1 and 2 tensors//
    //    //***************************************//
    //

    template<typename T>
    auto extractDiagonal(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr) {
        auto tensor         = tenx::asEval(expr);
        using Scalar        = typename decltype(tensor)::Scalar;
        using DimType       = typename decltype(tensor)::Dimensions;
        constexpr auto rank = Eigen::internal::array_size<DimType>::value;
        static_assert(rank == 2 and "extractDiagonal(): expression must be a tensor of rank 2");
        auto tensorMap = Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>>(tensor.data(), tensor.dimensions());
        auto size      = tensor.size();
        auto rows      = static_cast<Eigen::Index>(std::floor(std::sqrt(size)));
        return Eigen::Tensor<Scalar, 1>(tensorMap.reshape(array1{size}).stride(array1{rows + 1}));
    }

    template<typename T>
    auto asDiagonal(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr) {
        auto tensor              = tenx::asEval(expr);
        using Scalar             = typename decltype(tensor)::Scalar;
        using DimType            = typename decltype(tensor)::Dimensions;
        constexpr auto rank      = Eigen::internal::array_size<DimType>::value;
        auto           size      = tensor.size();
        auto           tensorMap = Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>>(tensor.data(), tensor.dimensions());
        return static_cast<Eigen::Tensor<Scalar, 2>>(tensorMap.inflate(array1{size + 1}).reshape(array2{size, size}));
    }

    template<typename TensorL, typename TensorM, typename TensorR>
    void asDiagonalContract(TensorR &R, const TensorL &L, const TensorM &M, long Mdim) {
        using ScalarL = typename TensorL::Scalar;
        using ScalarM = typename TensorM::Scalar;
        using ScalarR = typename TensorR::Scalar;
        static_assert(std::is_convertible_v<ScalarL, ScalarM>);
        static_assert(std::is_convertible_v<ScalarM, ScalarR>);
        static_assert(TensorM::NumDimensions == TensorR::NumDimensions);

        assert(M.dimension(Mdim) == L.size());
        assert(Mdim < TensorM::NumDimensions);
        assert(R.dimensions() == M.dimensions());
        for(long i = 0; i < L.size(); ++i) {
            R.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarM { return v * static_cast<ScalarM>(L(i)); }).template cast<ScalarR>();
        }
    }
    template<typename TensorL, typename TensorM, typename TensorR>
    void asDiagonalInverseContract(TensorR &R, const TensorL &L, const TensorM &M, long Mdim) {
        using ScalarL = typename TensorL::Scalar;
        using ScalarM = typename TensorM::Scalar;
        using ScalarR = typename TensorR::Scalar;
        static_assert(std::is_convertible_v<ScalarL, ScalarM>);
        static_assert(std::is_convertible_v<ScalarM, ScalarR>);
        static_assert(TensorM::NumDimensions == TensorR::NumDimensions);

        assert(M.dimension(Mdim) == L.size());
        assert(Mdim < TensorM::NumDimensions);
        assert(R.dimensions() == M.dimensions());
        for(long i = 0; i < L.size(); ++i) {
            R.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarM { return v / static_cast<ScalarM>(L(i)); }).template cast<ScalarR>();
        }
    }

    template<typename TensorL, typename TensorM>
    auto asDiagonalContract(const TensorL &L, const TensorM &M, long Mdim) {
        using ScalarM                     = typename TensorM::Scalar;
        constexpr auto               rank = TensorM::NumDimensions;
        Eigen::Tensor<ScalarM, rank> R(M.dimensions());
        asDiagonalContract(R, L, M, Mdim);
        return R;
    }
    template<typename TensorL, typename TensorM>
    auto asDiagonalInverseContract(const TensorL &L, const TensorM &M, long Mdim) {
        using ScalarM                     = typename TensorM::Scalar;
        constexpr auto               rank = TensorM::NumDimensions;
        Eigen::Tensor<ScalarM, rank> R(M.dimensions());
        asDiagonalInverseContract(R, L, M, Mdim);
        return R;
    }

    template<typename Scalar>
    Eigen::Tensor<Scalar, 2> asDiagonalSquared(const Eigen::Tensor<Scalar, 1> &tensor) {
        auto csquare = [](auto z) -> double { return std::abs(std::conj(z) * z); };
        return tensor.unaryExpr(csquare).inflate(array1{tensor.size() + 1}).reshape(array2{tensor.size(), tensor.size()});
    }
    template<typename Scalar>
    Eigen::Tensor<Scalar, 2> asDiagonalSqrt(const Eigen::Tensor<Scalar, 1> &tensor) {
        return tensor.sqrt().inflate(array1{tensor.size() + 1}).reshape(array2{tensor.size(), tensor.size()});
    }

    template<typename Scalar>
    Eigen::Tensor<Scalar, 2> asDiagonalInversed(const Eigen::Tensor<Scalar, 1> &tensor) {
        return tensor.inverse().inflate(array1{tensor.size() + 1}).reshape(array2{tensor.size(), tensor.size()});
    }

    template<typename Scalar>
    Eigen::Tensor<Scalar, 2> asDiagonalInversed(const Eigen::Tensor<Scalar, 2> &tensor) {
        if(tensor.dimension(0) != tensor.dimension(1)) throw std::runtime_error("tenx::asDiagonalInversed expects a square tensor");
        return asDiagonalInversed(extractDiagonal(tensor));
    }
    template<typename Scalar>
    bool isDiagonal(const Eigen::Tensor<Scalar, 2> &tensor, RealScalar<Scalar> threshold = Eigen::NumTraits<Scalar>::dummy_precision()) {
        if(tensor.dimension(0) != tensor.dimension(1)) return false;
        for(long col = 0; col < tensor.dimension(1); ++col) {
            for(long row = 0; row < tensor.dimension(0); ++row) {
                if(col == row) continue;
                if(abs(tensor(row, col)) > threshold) return false;
            }
        }
        return true;
    }

    template<typename T, typename Device = Eigen::DefaultDevice>
    auto norm(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr) {
        // auto csquare = [](auto z) -> double { return std::abs(std::conj(z) * z); };
        // return asEval(expr.unaryExpr(csquare).sum().sqrt())->coeff(0);
        auto tensor  = tenx::asEval(expr);
        using Scalar = typename decltype(tensor)::Scalar;
        return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(tensor.data(), tensor.size()).norm();
    }

    template<typename T>
    Eigen::Tensor<T, 1> broadcast(Eigen::Tensor<T, 1> &tensor, const std::array<long, 1> &bcast) {
        // Use this function to avoid a bug in Eigen when broadcasting complex tensors of rank 1, with compiler option -mfma
        // See more here https://gitlab.com/libeigen/eigen/-/issues/2351
        std::array<long, 2> bcast2 = {bcast[0], 1};
        std::array<long, 2> shape2 = {tensor.size(), 1};
        std::array<long, 1> shape1 = {tensor.size() * bcast[0]};
        return tensor.reshape(shape2).broadcast(bcast2).reshape(shape1);
    }

    template<typename T, typename Device = Eigen::DefaultDevice>
    auto asNormalized(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr) {
        auto tensor                          = tenx::asEval(expr);
        using Scalar                         = typename decltype(tensor)::Scalar;
        using DimType                        = typename decltype(tensor)::Dimensions;
        constexpr auto           rank        = Eigen::internal::array_size<DimType>::value;
        auto                     tensorMap   = Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>>(tensor.data(), tensor.dimensions());
        auto                     csquare     = [](auto z) -> double { return std::abs(std::conj(z) * z); };
        Eigen::Tensor<double, 0> normInverse = tensorMap.unaryExpr(csquare).sum().sqrt().inverse();
        return Eigen::Tensor<Scalar, rank>(tensorMap * tensorMap.constant(normInverse.coeff(0)));
    }

    template<typename Scalar, auto rank>
    void normalize(Eigen::Tensor<Scalar, rank> &tensor) {
        Eigen::Map<VectorType<Scalar>> map(tensor.data(), tensor.size());
        map.normalize();
    }

    template<typename Scalar, auto rank>
    auto TensorConstant(Scalar constant, const std::array<Eigen::Index, rank> &dims) {
        Eigen::Tensor<Scalar, rank> tensor(dims);
        tensor.setConstant(constant);
        return tensor;
    }

    template<typename Scalar, typename... Dims>
    auto TensorConstant(Scalar constant, const Dims... dims) {
        return TensorConstant(constant, std::array<Eigen::Index, sizeof...(Dims)>{dims...});
    }

    template<typename Scalar, auto rank>
    auto TensorRandom(const std::array<Eigen::Index, rank> &dims) {
        Eigen::Tensor<Scalar, rank> tensor(dims);
        tensor.setRandom();
        return tensor;
    }

    template<typename Scalar, typename... Dims>
    auto TensorRandom(const Dims... dims) {
        return TensorRandom<Scalar>(std::array<Eigen::Index, sizeof...(Dims)>{dims...});
    }

    template<typename Scalar>
    auto TensorIdentity(long diagSize) {
        Eigen::Tensor<Scalar, 1> tensor(diagSize);
        tensor.setConstant(1.0);
        return asDiagonal(tensor);
    }

    template<typename Scalar, auto rank>
    Eigen::Tensor<Scalar, rank - 2> trace(const Eigen::Tensor<Scalar, rank> &tensor, const idxlistpair<1> &idx_pair) {
        static_assert(rank >= 2, "Rank must be >= 2 for trace of an index pair");
        long idx0 = idx_pair[0].first;
        long idx1 = idx_pair[0].second;
        printf("Tracing pair %ld %ld of rank %ld tensor", idx0, idx1, rank);
        if(tensor.dimension(idx0) != tensor.dimension(idx1)) throw std::logic_error("Can't trace index pair of different dimensions");
        long                     dim0 = tensor.dimension(idx0);
        Eigen::Tensor<Scalar, 1> id(dim0);
        id.setConstant(1.0);
        return asDiagonal(id).contract(tensor, tenx::idx({0l, 1l}, {idx0, idx1}));
    }

    template<typename Scalar, auto rank>
    Eigen::Tensor<Scalar, rank - 4> trace(const Eigen::Tensor<Scalar, rank> &tensor, const idxlistpair<2> &idx_pair) {
        static_assert(rank >= 4, "Rank must be >= 4 for trace of 2 index pairs");
        long idx00 = idx_pair[0].first;
        long idx01 = idx_pair[0].second;
        long idx10 = idx_pair[1].first;
        long idx11 = idx_pair[1].second;
        // idx_pair[0] connect to each other, and then idx_pair[1] connect to each other
        if(tensor.dimension(idx00) != tensor.dimension(idx01)) throw std::logic_error("Can't trace index pair of different dimensions");
        if(tensor.dimension(idx10) != tensor.dimension(idx11)) throw std::logic_error("Can't trace index pair of different dimensions");
        long                     dim00 = tensor.dimension(idx00);
        long                     dim11 = tensor.dimension(idx11);
        Eigen::Tensor<Scalar, 1> id00(dim00), id11(dim11);
        id00.setConstant(1.0);
        id11.setConstant(1.0);
        return asDiagonal(id00)
            .contract(asDiagonal(id11), tenx::idx()) // Outer product
            .contract(tensor, tenx::idx({0l, 1l, 2l, 3l}, {idx00, idx01, idx10, idx11}));
    }

    //    //****************************//
    //    //Matrix to tensor conversions//
    //    //****************************//

    // Detects if Derived is a plain object, like "MatrixXd" or similar.
    // std::decay removes pointer or ref qualifiers if present
    template<typename Derived>
    using is_plain_object = std::is_base_of<Eigen::PlainObjectBase<std::decay_t<Derived>>, std::decay_t<Derived>>;
    template<typename Derived>
    inline constexpr bool is_plain_object_v = is_plain_object<Derived>::value;
    template<typename Derived>
    inline constexpr bool is_fixed_size_v = Derived::RowsAtCompileTime != Eigen::Dynamic && Derived::ColsAtCompileTime != Eigen::Dynamic;
    template<typename Derived>
    inline constexpr bool is_dynamic_size_v = !is_fixed_size_v<Derived>;

    template<typename Derived, typename T, auto rank>
    Eigen::Tensor<typename Derived::Scalar, rank> TensorCast(const Eigen::EigenBase<Derived> &matrix, const std::array<T, rank> &dims) {
        // static_assert(is_dynamic_size_v<Derived>);
        if constexpr(is_plain_object<Derived>::value) {
            assert(matrix.size() == Eigen::internal::array_prod(dims));
            return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>(matrix.derived().data(), dims);
        } else {
            return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>(matrix.derived().eval().data(), dims);
        }
    }

    // template<typename Derived>
    // auto TensorFixedSizeCast(const Eigen::EigenBase<Derived> &matrix) {
    //     static_assert(is_fixed_size_v<Derived>);
    //     if constexpr(is_plain_object<Derived>::value) {
    //         assert(matrix.size() == Eigen::internal::array_prod(dims));
    //         return Eigen::TensorMap<
    //             const Eigen::TensorFixedSize<const typename Derived::Scalar, Eigen::Sizes<Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>>(
    //             matrix.derived().data(), Derived::RowsAtCompileTime, Derived::ColsAtCompileTime);
    //     } else {
    //         return Eigen::TensorMap<
    //             const Eigen::TensorFixedSize<const typename Derived::Scalar, Eigen::Sizes<Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>>(
    //             matrix.derived().eval().data(), Derived::RowsAtCompileTime, Derived::ColsAtCompileTime);
    //     }
    // }
    // template<typename Dimensions, typename Derived>
    // auto TensorFixedSizeCast(const Eigen::EigenBase<Derived> &matrix) {
    //     static_assert(is_fixed_size_v<Derived>);
    //     if constexpr(is_plain_object<Derived>::value) {
    //         assert(matrix.size() == Eigen::internal::array_prod(Dimensions{}));
    //         return Eigen::TensorMap<
    //             const Eigen::TensorFixedSize<const typename Derived::Scalar, Dimensions>>(
    //             matrix.derived().data(), Dimensions{});
    //     } else {
    //         return Eigen::TensorMap<
    //             const Eigen::TensorFixedSize<const typename Derived::Scalar, Dimensions>>(
    //             matrix.derived().eval().data(), Dimensions{});
    //     }
    // }

    template<typename Derived, typename T, auto rank>
    Eigen::Tensor<typename Derived::Scalar, rank> TensorCast(const Eigen::EigenBase<Derived> &matrix, const Eigen::DSizes<T, rank> &dims) {
        // static_assert(is_dynamic_size_v<Derived>);
        if constexpr(is_plain_object_v<Derived>) {
            assert(matrix.size() == Eigen::internal::array_prod(dims));
            return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>(matrix.derived().data(), dims);
        } else {
            return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>(matrix.derived().eval().data(), dims);
        }
    }

    // Helpful overload
    template<typename Derived, typename... Dims>
    auto TensorCast(const Eigen::EigenBase<Derived> &matrix, const Dims... dims) {
        static_assert(sizeof...(Dims) > 0, "TensorCast: sizeof... (Dims) must be larger than 0");
        return TensorCast(matrix, std::array<Eigen::Index, sizeof...(Dims)>{dims...});
    }

    template<typename Derived>
    auto TensorCast(const Eigen::EigenBase<Derived> &matrix) {
        if constexpr(Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
            return TensorCast(matrix, matrix.size());
        } else {
            return TensorCast(matrix, matrix.rows(), matrix.cols());
        }
    }

    template<typename Derived, auto rank>
    auto TensorMap(Eigen::PlainObjectBase<Derived> &matrix, const std::array<long, rank> &dims) {
        return Eigen::TensorMap<Eigen::Tensor<typename Derived::Scalar, rank>>(matrix.derived().data(), dims);
    }
    template<typename Derived, auto rank>
    auto TensorMap(const Eigen::PlainObjectBase<Derived> &matrix, const std::array<long, rank> &dims) {
        return Eigen::TensorMap<const Eigen::Tensor<typename Derived::Scalar, rank>>(matrix.derived().data(), dims);
    }
    template<typename Derived, typename... Dims>
    auto TensorMap(Eigen::PlainObjectBase<Derived> &matrix, const Dims... dims) {
        return TensorMap(matrix, std::array<long, static_cast<long>(sizeof...(Dims))>{dims...});
    }
    template<typename Derived, typename... Dims>
    auto TensorMap(const Eigen::PlainObjectBase<Derived> &matrix, const Dims... dims) {
        return TensorMap(matrix, std::array<long, static_cast<long>(sizeof...(Dims))>{dims...});
    }
    template<typename Derived>
    auto TensorMap(Eigen::PlainObjectBase<Derived> &matrix) {
        if constexpr(Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
            return TensorMap(matrix, matrix.size());
        } else {
            return TensorMap(matrix, matrix.rows(), matrix.cols());
        }
    }
    template<typename Derived>
    auto TensorMap(const Eigen::PlainObjectBase<Derived> &matrix) {
        if constexpr(Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
            return TensorMap(matrix, matrix.size());
        } else {
            return TensorMap(matrix, matrix.rows(), matrix.cols());
        }
    }
    //
    //    //****************************//
    //    //Tensor to matrix conversions//
    //    //****************************//
    //
    template<typename Scalar>
    auto MatrixCast(const Eigen::Tensor<Scalar, 2> &tensor) {
        return static_cast<MatrixType<Scalar>>(Eigen::Map<const MatrixType<Scalar>>(tensor.data(), tensor.dimension(0), tensor.dimension(1)));
    }
    template<typename T, typename sizeType>
    auto MatrixCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr, const sizeType rows, const sizeType cols) {
        auto tensor  = asEval(expr);
        using Scalar = typename decltype(tensor)::Scalar;
        return MatrixType<Scalar>(Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols));
    }

    template<typename T>
    auto VectorCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr) {
        auto tensor  = asEval(expr);
        using Scalar = typename decltype(tensor)::Scalar;
        //        auto size    = tensor.size();
        auto size = Eigen::internal::array_prod(tensor.dimensions());
        return VectorType<Scalar>(Eigen::Map<const VectorType<Scalar>>(tensor.data(), size));
    }

    template<typename Scalar, auto rank, typename sizeType>
    auto MatrixMap(const Eigen::Tensor<Scalar, rank> &tensor, const sizeType rows, const sizeType cols) {
        return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
    }
    template<typename Scalar, auto rank, typename sizeType>
    auto MatrixMap(Eigen::Tensor<Scalar, rank> &tensor, const sizeType rows, const sizeType cols) {
        return Eigen::Map<MatrixType<Scalar>>(tensor.data(), rows, cols);
    }
    template<typename Scalar, auto rank, typename sizeType>
    auto MatrixMap(const Eigen::Tensor<Scalar, rank> &&tensor, const sizeType rows, const sizeType cols) = delete; // Prevent map from temporary

    template<typename Scalar>
    auto MatrixMap(const Eigen::Tensor<Scalar, 2> &tensor) {
        return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
    }
    template<typename Scalar>
    auto MatrixMap(Eigen::Tensor<Scalar, 2> &tensor) {
        return Eigen::Map<MatrixType<Scalar>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
    }
    template<typename Scalar>
    auto MatrixMap(const Eigen::Tensor<Scalar, 2> &&tensor) = delete; // Prevent map from temporary

    template<typename Scalar, auto rank>
    auto VectorMap(const Eigen::Tensor<Scalar, rank> &tensor) {
        return Eigen::Map<const VectorType<Scalar>>(tensor.data(), tensor.size());
    }
    template<typename Scalar, auto rank>
    auto VectorMap(Eigen::Tensor<Scalar, rank> &tensor) {
        return Eigen::Map<VectorType<Scalar>>(tensor.data(), tensor.size());
    }
    template<typename Derived>
    auto VectorMap(const Eigen::TensorMap<Derived> &tensor) {
        return Eigen::Map<const VectorType<typename Derived::Scalar>>(tensor.data(), tensor.size());
    }
    template<typename Derived>
    auto VectorMap(Eigen::TensorMap<Derived> &tensor) {
        return Eigen::Map<VectorType<typename Derived::Scalar>>(tensor.data(), tensor.size());
    }
    template<typename Scalar, auto rank>
    auto VectorMap(const Eigen::Tensor<Scalar, rank> &&tensor) = delete; // Prevent map from temporary

    //************************//
    // change storage layout //
    //************************//
    template<typename Scalar, auto rank>
    Eigen::Tensor<Scalar, rank, Eigen::RowMajor> to_RowMajor(const Eigen::Tensor<Scalar, rank, Eigen::ColMajor> tensor) {
        std::array<long, rank> neworder;
        std::iota(std::begin(neworder), std::end(neworder), 0);
        std::reverse(neworder.data(), neworder.data() + neworder.size());
        return tensor.swap_layout().shuffle(neworder);
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> to_RowMajor(const Eigen::MatrixBase<Derived> &matrix) {
        if(matrix.IsRowMajor) { return matrix; }
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrowmajor = matrix;
        return matrowmajor;
    }

    template<typename Scalar, auto rank>
    Eigen::Tensor<Scalar, rank, Eigen::ColMajor> to_ColMajor(const Eigen::Tensor<Scalar, rank, Eigen::RowMajor> tensor) {
        std::array<long, rank> neworder;
        std::iota(std::begin(neworder), std::end(neworder), 0);
        std::reverse(neworder.data(), neworder.data() + neworder.size());
        return tensor.swap_layout().shuffle(neworder);
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> to_ColMajor(const Eigen::MatrixBase<Derived> &matrix) {
        if(not matrix.IsRowMajor) { return matrix; }
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matrowmajor = matrix;
        return matrowmajor;
    }

    //******************************************************//
    // Tests for real/complex //
    //******************************************************//

    template<typename Derived>
    bool isPositive(const Eigen::EigenBase<Derived> &obj) {
        return (obj.derived().array().real() >= 0).all();
    }

    template<typename Scalar, auto rank>
    bool isPositive(const Eigen::Tensor<Scalar, rank> &tensor) {
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vector(tensor.data(), tensor.size());
        return isPositive(vector);
    }

    template<typename Derived, typename... Args>
    bool isReal(const Eigen::EigenBase<Derived> &obj, Args &&...args) {
        using Scalar     = typename Derived::Scalar;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        if constexpr(sfinae::is_std_complex_v<Scalar> and std::is_arithmetic_v<RealScalar>) {
            return obj.derived().imag().isZero(std::forward<Args>(args)...);
        } else {
            return true;
        }
    }

    template<typename T, typename... Args>
    bool isReal(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr, Args &&...args) {
        auto tensor      = tenx::asEval(expr);
        using Scalar     = typename decltype(tensor)::Scalar;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        if constexpr(sfinae::is_std_complex_v<Scalar> and std::is_arithmetic_v<RealScalar>) {
            auto vector = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(tensor.data(), tensor.size());
            return vector.imag().isZero(std::forward<Args>(args)...);
        } else {
            return true;
        }
    }

    template<typename Scalar, auto rank, typename... Args>
    bool isZero(const Eigen::Tensor<Scalar, rank> &tensor, Args &&...args) {
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vector(tensor.data(), tensor.size());
        return vector.isZero(std::forward<Args>(args)...);
    }

    template<typename Scalar, typename... Args>
    bool isIdentity(const Eigen::Tensor<Scalar, 2> &tensor, Args &&...args) {
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> matrix(tensor.data(), tensor.dimension(0), tensor.dimension(1));
        return matrix.isIdentity(std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    auto isIdentity(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr, Args &&...args) {
        auto tensor         = tenx::asEval(expr);
        using Scalar        = typename decltype(tensor)::Scalar;
        using DimType       = typename decltype(tensor)::Dimensions;
        constexpr auto rank = Eigen::internal::array_size<DimType>::value;
        static_assert(rank == 2 and "isIdentity(): expression must be a tensor of rank 2");
        auto size      = tensor.size();
        auto matrixMap = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
        return matrixMap.isIdentity(std::forward<Args>(args)...);
    }

    template<typename Derived>
    bool hasNaN(const Eigen::EigenBase<Derived> &obj) {
        return obj.derived().hasNaN();
    }

    template<typename Scalar, auto rank>
    bool hasNaN(const Eigen::Tensor<Scalar, rank> &tensor) {
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vector(tensor.data(), tensor.size());
        return hasNaN(vector);
    }

    template<typename Scalar>
    bool hasNaN(const std::vector<Scalar> &vec) {
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vector(vec.data(), safe_cast<long>(vec.size()));
        return hasNaN(vector);
    }

    template<typename Derived>
    auto subtract_phase(Eigen::MatrixBase<Derived> &v) {
        using Scalar = typename Derived::Scalar;
        std::vector<double> angles;
        if constexpr(std::is_same<Scalar, std::complex<double>>::value) {
            for(int i = 0; i < v.cols(); i++) {
                if(v.col(i)(0).imag() == 0.0) {
                    angles.emplace_back(0.0);
                    continue;
                }
                angles.emplace_back(std::arg(v.col(i)(0)));
                Scalar inv_phase     = Scalar(0.0, -1.0) * angles.back();
                Scalar exp_inv_phase = std::exp(inv_phase);
                v.col(i) *= exp_inv_phase;
                v.col(i) = (v.col(i).array().imag().cwiseAbs() > 1e-15).select(v.col(i), v.col(i).real());
            }
        }
        return angles;
    }

    template<typename Scalar, auto rank>
    auto subtract_phase(Eigen::Tensor<Scalar, rank> &tensor) {
        auto map = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(tensor.data(), tensor.size());
        return subtract_phase(map);
    }

    template<typename Derived>
    void add_phase(Eigen::MatrixBase<Derived> &v, std::vector<double> &angles) {
        using Scalar = typename Derived::Scalar;
        if constexpr(std::is_same<Scalar, std::complex<double>>::value) {
            if(v.cols() != angles.size()) { throw std::runtime_error("Mismatch in columns and angles supplied"); }
            for(int i = 0; i < v.cols(); i++) {
                Scalar exp_phase = std::exp(Scalar(0.0, 1.0) * angles[i]);
                v.col(i) *= exp_phase;
                v.col(i) = (v.col(i).array().imag().cwiseAbs() > 1e-15).select(v.col(i), v.col(i).real());
            }
        }
    }

    template<typename Scalar, auto rank>
    void add_phase(Eigen::Tensor<Scalar, rank> &tensor, std::vector<double> &angles) {
        auto map = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(tensor.data(), tensor.size());
        add_phase(map, angles);
    }

    // Compute sparcity
    template<typename Scalar, auto rank>
    double sparcity(const Eigen::Tensor<Scalar, rank> &tensor) {
        auto map = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(tensor.data(), tensor.size());
        return static_cast<double>((map.array() != 0.0).count()) / static_cast<double>(map.size());
    }
    template<typename Derived>
    double sparcity(const Eigen::EigenBase<Derived> &matrix) {
        return static_cast<double>((matrix.derived().array() != 0.0).count()) / static_cast<double>(matrix.derived().size());
    }

    template<typename Scalar, typename T>
    requires tenx::sfinae::is_eigen_tensor_v<std::remove_cvref_t<T>>
    decltype(auto) asScalarType(T &&tensor) {
        static_assert(sfinae::is_any_v<Scalar, fp32, fp64, fp128, cx32, cx64, cx128>);
        using EigenType = std::remove_cvref_t<decltype(tensor)>;
        using OldScalar = typename EigenType::Scalar;
        if constexpr(std::is_same_v<OldScalar, Scalar>) {
            return (tensor); // Returns the exact same object as a reference
        } else if constexpr(sfinae::is_std_complex_v<OldScalar> and !sfinae::is_std_complex_v<Scalar>) {
            // Complex to Real
            using DimType       = typename EigenType::Dimensions;
            constexpr auto rank = Eigen::internal::array_size<DimType>::value;
            return Eigen::Tensor<Scalar, rank>(tensor.real().template cast<Scalar>());
        } else {
            // Cast between different precisions, e.g. fp64 to fp32
            using DimType       = typename EigenType::Dimensions;
            constexpr auto rank = Eigen::internal::array_size<DimType>::value;
            return Eigen::Tensor<Scalar, rank>(tensor.template cast<Scalar>());
        }
    }
    template<typename Scalar, typename T>
    requires tenx::sfinae::is_eigen_matrix_v<std::remove_cvref_t<T>>
    decltype(auto) asScalarType(T &&obj) {
        static_assert(sfinae::is_any_v<Scalar, fp32, fp64, fp128, cx32, cx64, cx128>);
        using EigenType = std::remove_cvref_t<decltype(obj)>;
        using OldScalar = typename EigenType::Scalar;
        if constexpr(std::is_same_v<OldScalar, Scalar>) {
            return (obj); // Returns the exact same object as a reference
        } else if constexpr(sfinae::is_std_complex_v<OldScalar> and !sfinae::is_std_complex_v<Scalar>) {
            // Complex to Real
            return Eigen::Matrix<Scalar, EigenType::RowsAtCompileTime, EigenType::ColsAtCompileTime>(obj.real().template cast<Scalar>());
        } else {
            // Cast between different precisions, e.g. fp64 to fp32
            return Eigen::Matrix<Scalar, EigenType::RowsAtCompileTime, EigenType::ColsAtCompileTime>(obj.template cast<Scalar>());
        }
    }
    template<typename Scalar, typename T>
    requires tenx::sfinae::is_eigen_array_v<std::remove_cvref_t<T>>
    decltype(auto) asScalarType(T &&obj) {
        static_assert(sfinae::is_any_v<Scalar, fp32, fp64, fp128, cx32, cx64, cx128>);
        using EigenType = std::remove_cvref_t<decltype(obj)>;
        using OldScalar = typename EigenType::Scalar;
        if constexpr(std::is_same_v<OldScalar, Scalar>) {
            return (obj); // Returns the exact same object as a reference
        } else if constexpr(sfinae::is_std_complex_v<OldScalar> and !sfinae::is_std_complex_v<Scalar>) {
            // Complex to Real
            return Eigen::Array<Scalar, EigenType::RowsAtCompileTime, EigenType::ColsAtCompileTime>(obj.real().template cast<Scalar>());
        } else {
            // Cast between different precisions, e.g. fp64 to fp32
            return Eigen::Array<Scalar, EigenType::RowsAtCompileTime, EigenType::ColsAtCompileTime>(obj.template cast<Scalar>());
        }
    }

    // static_assert(std::is_same_v<decltype(asScalarType<fp32>(Eigen::Tensor<cx64,3>())), Eigen::Tensor<fp32,3>>);
    // using crefTensor = const Eigen::Tensor<cx64,3> &;
    // static_assert(std::is_same_v<decltype(asScalarType<cx64>(crefTensor())), const Eigen::Tensor<cx64,3> &>);
    // static_assert(!std::is_same_v<decltype(asScalarType<cx64>(Eigen::Tensor<cx64,3>())), Eigen::Tensor<cx64,3>>);
    // static_assert(std::is_same_v<decltype(asScalarType<cx64>(Eigen::Tensor<cx64,3>())), Eigen::Tensor<cx64,3> &>);
    // template<typename Scalar, typename T, int AccessorLevel>
    // decltype(auto) asScalarType(const Eigen::TensorBase<T, AccessorLevel> &expr) {
    //     if constexpr(!tenx::sfinae::has_expression_v<T>) {
    //         const auto &cref = *static_cast<const T *>(&expr);
    //         if constexpr(sfinae::is_std_complex_v<typename T::Scalar> and !sfinae::is_std_complex_v<Scalar>)
    //             return cref.real().template cast<Scalar>(); // Return the plain tensor as const ref
    //         else {
    //             cref.template cast<Scalar>();
    //         }
    //     } else {
    //         auto tensor              = tenx::asEval(expr);
    //         using OldScalar          = typename decltype(tensor)::Scalar;
    //         using DimType            = typename decltype(tensor)::Dimensions;
    //         constexpr bool cx2fp     = sfinae::is_std_complex_v<OldScalar> and !sfinae::is_std_complex_v<Scalar>;
    //         constexpr auto rank      = Eigen::internal::array_size<DimType>::value;
    //         auto           tensorMap = Eigen::TensorMap<const Eigen::Tensor<OldScalar, rank>>(tensor.data(), tensor.dimensions());
    //         if constexpr(std::is_same_v<Scalar, OldScalar>) {
    //             return tensorMap;
    //         } else if constexpr(cx2fp) {
    //             assert(tenx::isReal(tensorMap));
    //             return Eigen::Tensor<Scalar, rank>(tensorMap.real().template cast<Scalar>());
    //         } else {
    //             return Eigen::Tensor<Scalar, rank>(tensorMap.template cast<Scalar>());
    //         }
    //     }
    // }

    //******************************//
    // BLAS operations with tensors //
    //******************************//
    template<typename Scalar>
    void gemm(Eigen::Tensor<Scalar, 2> &lhs, const Eigen::Tensor<Scalar, 2> &rhs1, const Eigen::Tensor<Scalar, 2> &rhs2) {
        lhs.resize(rhs1.dimension(0), rhs2.dimension(1));
        auto lhs_map      = Eigen::Map<MatrixType<Scalar>>(lhs.data(), rhs1.dimension(0), rhs2.dimension(1));
        auto rhs1_map     = MatrixMap(rhs1);
        auto rhs2_map     = MatrixMap(rhs2);
        lhs_map.noalias() = rhs1_map * rhs2_map; // Calls BLAS GEMM
    }
    template<typename Scalar>
    Eigen::Tensor<Scalar, 2> gemm(const Eigen::Tensor<Scalar, 2> &rhs1, const Eigen::Tensor<Scalar, 2> &rhs2) {
        auto lhs          = Eigen::Tensor<Scalar, 2>(rhs1.dimension(0), rhs2.dimension(1));
        auto lhs_map      = Eigen::Map<MatrixType<Scalar>>(lhs.data(), rhs1.dimension(0), rhs2.dimension(1));
        auto rhs1_map     = MatrixMap(rhs1);
        auto rhs2_map     = MatrixMap(rhs2);
        lhs_map.noalias() = rhs1_map * rhs2_map; // Calls BLAS GEMM
        return lhs;
    }
    template<typename Scalar>
    Eigen::Tensor<Scalar, 4> gemm_mpo(const Eigen::Tensor<Scalar, 4> &rhs1, const Eigen::Tensor<Scalar, 4> &rhs2) {
        /*!       2                2                  1                2                  1    4                  2    3
         *        |                |                  |                |                  |    |                  |    |
         * 0---[rhs1]---1    ---[rhs2]---   =  0---[rhs1]---3   0---[rhs2]---1  =   0---[rhs1rhs2]---3   =  0---[rhs1rhs2]---1
         *        |                |                  |                |                  |    |                  |    |
         *        3                3                  2                3                  2    5                  4    5
         */
        if constexpr(sfinae::is_std_complex_v<Scalar>) {
            if(isReal(rhs1) and isReal(rhs2)) {
                // We get a speedup by multiplying reals instead of complex
                using TensorValueType = Eigen::Tensor<typename Scalar::value_type, 4>;
                return gemm_mpo(TensorValueType(rhs1.real()), TensorValueType(rhs2.real())).template cast<Scalar>();
            }
        }
        auto dim6     = array6{rhs1.dimension(0), rhs1.dimension(2), rhs1.dimension(3), rhs2.dimension(1), rhs2.dimension(2), rhs2.dimension(3)};
        auto dim_lhsA = array4{rhs1.dimension(0), rhs1.dimension(2) * rhs1.dimension(3), rhs2.dimension(1), rhs2.dimension(2) * rhs2.dimension(3)};
        auto dim_lhsB = array4{rhs1.dimension(0), rhs2.dimension(1), rhs1.dimension(2) * rhs2.dimension(2), rhs1.dimension(3) * rhs2.dimension(3)};
        auto dim_rhs1 = rhs1.dimensions();
        auto dim_rhs2 = rhs2.dimensions();
        auto shf6     = tenx::array6{0, 3, 1, 4, 2, 5};
        Eigen::Tensor<Scalar, 4> lhs(dim_lhsA);
        Eigen::Tensor<Scalar, 4> rhs1_shf4 = rhs1.shuffle(array4{0, 2, 3, 1}); // Prepare for gemm
        auto                     lhs_map   = Eigen::Map<MatrixType<Scalar>>(lhs.data(), dim_lhsA[0] * dim_lhsA[1], dim_lhsA[2] * dim_lhsA[3]);
        auto                     rhs1_map  = MatrixMap(rhs1_shf4, dim_rhs1[0] * dim_rhs1[2] * dim_rhs1[3], dim_rhs1[1]);
        auto                     rhs2_map  = MatrixMap(rhs2, dim_rhs2[0], dim_rhs2[1] * dim_rhs2[2] * dim_rhs2[3]);
        lhs_map.noalias()                  = rhs1_map * rhs2_map; // Calls BLAS GEMM
        return lhs.reshape(dim6).shuffle(shf6).reshape(dim_lhsB);
    }
}
/*clang-format on */