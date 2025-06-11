#pragma once
#include "config/debug.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include "tools/common/log.h"
#include <Eigen/Core>

template<typename Derived>
void assert_allfinite(const Eigen::MatrixBase<Derived> &X, const std::source_location &location = std::source_location::current()) {
    if constexpr(settings::debug) {
        if(X.cols() == 0) return;
        bool allFinite = X.allFinite();
        if(!allFinite) {
            tools::log->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            tools::log->warn("X is not all finite: \n{}\n", linalg::matrix::to_string(X, 8));
            throw except::runtime_error("{}:{}: {}: matrix has non-finite elements", location.file_name(), location.line(), location.function_name());
        }
    }
}

template<typename Derived>
void assert_orthonormal_cols(const Eigen::MatrixBase<Derived> &X,
                             typename Derived::RealScalar      threshold = std::numeric_limits<typename Derived::RealScalar>::epsilon() * 10000,
                             const std::source_location       &location  = std::source_location::current()) {
    if constexpr(settings::debug) {
        if(X.cols() == 0) return;
        using Scalar         = typename Derived::Scalar;
        using MatrixType     = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        MatrixType XX        = X.adjoint() * X;
        auto       orthError = (XX - MatrixType::Identity(XX.cols(), XX.rows())).norm();
        if(orthError > threshold) {
            tools::log->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            tools::log->warn("X.adjoint()*X: \n{}\n", linalg::matrix::to_string(XX, 8));
            tools::log->warn("X orthError: {:.5e}", orthError);
            throw except::runtime_error("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > threshold = {:.5e}", location.file_name(), location.line(),
                                        location.function_name(), orthError, threshold);
        }
    }
}

template<typename Derived>
void assert_orthonormal_rows(const Eigen::MatrixBase<Derived> &X,
                             typename Derived::RealScalar      threshold = std::numeric_limits<typename Derived::RealScalar>::epsilon() * 10000,
                             const std::source_location       &location  = std::source_location::current()) {
    if constexpr(settings::debug) {
        if(X.cols() == 0) return;
        using Scalar         = typename Derived::Scalar;
        using MatrixType     = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        MatrixType XX        = X * X.adjoint();
        auto       orthError = (XX - MatrixType::Identity(XX.cols(), XX.rows())).norm();
        if(orthError > threshold) {
            tools::log->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            tools::log->warn("X.adjoint()*X: \n{}\n", linalg::matrix::to_string(XX, 8));
            tools::log->warn("X orthError: {:.5e}", orthError);
            throw except::runtime_error("{}:{}: {}: matrix has non-orthonormal rows: error = {:.5e} > threshold = {:.5e}", location.file_name(),
                                        location.line(), location.function_name(), orthError, threshold);
        }
    }
}

template<Eigen::Index axis, typename Derived>
void assert_orthonormal(const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors> &expr,
                        typename Derived::RealScalar                                threshold = std::numeric_limits<typename Derived::RealScalar>::epsilon(),
                        const std::source_location                                 &location  = std::source_location::current()) {
    if constexpr(settings::debug) {
        static_assert(axis == 1 or axis == 2);
        auto tensor         = tenx::asEval(expr);
        using DimType       = typename decltype(tensor)::Dimensions;
        constexpr auto rank = Eigen::internal::array_size<DimType>::value;
        static_assert(rank == 3 and "assert_orthonormal: expression must be a tensor of rank 3");

        using Scalar     = typename decltype(tensor)::Scalar;
        using TensorType = Eigen::Tensor<Scalar, rank>;
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using MapType    = Eigen::Map<const MatrixType>;

        auto X = Eigen::TensorMap<const TensorType>(tensor.data(), tensor.dimensions());
        if(X.size() == 0) return;

        Eigen::Tensor<Scalar, 2> G;
        Eigen::Index             oxis    = axis == 1l ? 2l : 1l; // other axis
        auto                     idxpair = tenx::idx({0l, oxis}, {0l, oxis});
        G                                = X.conjugate().contract(X, idxpair); // Gram matrix
        auto Gm                          = MapType(G.data(), G.dimension(0), G.dimension(1));
        auto orthError                   = (Gm - MatrixType::Identity(Gm.cols(), Gm.rows())).norm();
        if(orthError > 100 * threshold * Gm.rows()) {
            auto Xm = MapType(X.data(), X.dimension(0) * X.dimension(oxis), X.dimension(axis));
            // tools::log->warn("X: \n{}\n", linalg::matrix::to_string(Xm, 8));
            tools::log->warn("G: \n{}\n", linalg::matrix::to_string(Gm, 8));
            tools::log->warn("orthError: {:.5e}", fp(orthError));
            throw except::runtime_error("{}:{}: {}: matrix is non-orthonormal along axis [{}]: error = {:.5e} > threshold = {:.5e}", location.file_name(),
                                        location.line(), location.function_name(), axis, fp(orthError), fp(threshold));
        }
    }
}