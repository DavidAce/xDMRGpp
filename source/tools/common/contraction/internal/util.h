#pragma once
#include <Eigen/Core>
#include <type_traits>
namespace tools::common::contraction::internal {

    auto get_size(const auto &dims) -> Eigen::Index {
        Eigen::Index size = 1;
        for(Eigen::Index i = 0; i < static_cast<Eigen::Index>(dims.size()); ++i) size *= dims[i];
        return size;
    }
    template<typename Scalar>
    auto get_norm(const Scalar *const ptr, const auto &dims) -> decltype(std::real(std::declval<Scalar>())) {
        return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(ptr, get_size(dims)).norm();
    }

}