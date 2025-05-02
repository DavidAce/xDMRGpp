#pragma once

#include "debug/exceptions.h"
#include "math/cast.h"
#include "solution.h"
#include <fmt/format.h>
#include <unsupported/Eigen/CXX11/Tensor>
namespace eig::view {

    template<typename T>
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename T>
    using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    template<typename Scalar>
    Eigen::Map<VectorType<Scalar>> get_eigvals(const eig::solution &result, bool converged_only = true) {
        if(not result.meta.eigvals_found) throw except::runtime_error("Results have not been obtained yet: eigvals found [{}]", result.meta.eigvals_found);
        if constexpr(std::is_same_v<Scalar, fp64>)
            if(not result.eigvals_are_real()) throw except::runtime_error("Can't view real eigenvalues: the solution has complex eigenvalues");

        auto &eigvals = result.get_eigvals<Scalar>();
        if(eigvals.empty()) throw except::runtime_error("The requested eigenvalues are empty. Did you request the correct type?");
        long size = converged_only ? safe_cast<long>(result.meta.nev_converged) : safe_cast<long>(eigvals.size());
        return Eigen::Map<VectorType<Scalar>>(eigvals.data(), size);
    }

    template<typename Scalar, typename integral_type = long, typename = std::enable_if<std::is_integral_v<integral_type>>>
    Scalar get_eigval(const eig::solution &result, integral_type num = 0) {
        return get_eigvals<Scalar>(result, false)(safe_cast<long>(num));
    }

    template<typename Scalar>
    Eigen::Map<MatrixType<Scalar>> get_eigvecs(const eig::solution &result, Side side = Side::R, bool converged_only = true) {
        if(side == Side::R and not result.meta.eigvecsR_found)
            throw except::runtime_error("Results have not been obtained yet: eigvecs R found [{}]", result.meta.eigvecsR_found);
        else if(side == Side::L and not result.meta.eigvecsL_found)
            throw except::runtime_error("Results have not been obtained yet: eigvecs L found [{}]", result.meta.eigvecsL_found);

        if(side == Side::LR) throw except::runtime_error("Cannot get both L and R eigenvectors. Choose one");

        if constexpr(sfinae::is_any_v<Scalar, fp32, fp64, fp128>)
            if(not result.eigvecs_are_real()) throw except::runtime_error("Can't view real eigenvectors: solution has complex eigenvectors");
        auto &eigvecs = result.get_eigvecs<Scalar>(side);
        if(eigvecs.empty()) throw except::runtime_error("The requested eigenvectors are empty. Did you request the correct type?");
        if(result.meta.rows * result.meta.cols != safe_cast<eig::size_type>(eigvecs.size())) throw std::logic_error("Size mismatch in results");
        auto rows = result.meta.rows;
        auto cols = converged_only ? result.meta.nev_converged : result.meta.cols;
        return Eigen::Map<MatrixType<Scalar>>(eigvecs.data(), rows, cols);
    }

    template<typename Scalar>
    Eigen::Map<VectorType<Scalar>> get_eigvec(const eig::solution &result, long num = 0, Side side = Side::R) {
        auto eigvecmap = get_eigvecs<Scalar>(result, side, false).col(num);
        return Eigen::Map<VectorType<Scalar>>(eigvecmap.data(), eigvecmap.size());
    }

    template<typename Scalar>
    Eigen::TensorMap<Eigen::Tensor<Scalar, 3>> get_eigvec(const eig::solution &result, const std::array<long, 3> &dims, long num = 0, Side side = Side::R) {
        if(result.meta.rows != dims[0] * dims[1] * dims[2])
            throw except::range_error("Given tensor dimensions do not match eigenvector size: size {} != {} * {} * {} = {}", result.meta.rows, dims[0], dims[1],
                                      dims[2], dims[0] * dims[1] * dims[2]);
        auto eigvecmap = get_eigvec<Scalar>(result, num, side);
        return Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(eigvecmap.data(), dims);
    }
}
