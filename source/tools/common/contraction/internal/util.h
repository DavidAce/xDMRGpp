#pragma once
#include <Eigen/Core>
namespace tools::common::contraction::internal {
    template<typename Scalar>
    struct Info {
        using RealScalar              = decltype(std::real(std::declval<Scalar>()));
        bool       contract_left      = true;
        RealScalar mps_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar mpo_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar envL_norm          = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar envR_norm          = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST1                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST2                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST3                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar cancelation_factor = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar highprec_threshold = RealScalar{10} / std::sqrt(std::numeric_limits<RealScalar>::epsilon());
        template<typename T>
        Info<Scalar> &operator=(const Info<T> &info_) {
            this->contract_left      = info_.contract_left;
            this->mps_norm           = static_cast<RealScalar>(info_.mps_norm);
            this->mpo_norm           = static_cast<RealScalar>(info_.mpo_norm);
            this->envL_norm          = static_cast<RealScalar>(info_.envL_norm);
            this->envR_norm          = static_cast<RealScalar>(info_.envR_norm);
            this->ST1                = static_cast<RealScalar>(info_.ST1);
            this->ST2                = static_cast<RealScalar>(info_.ST2);
            this->ST3                = static_cast<RealScalar>(info_.ST3);
            this->cancelation_factor = static_cast<RealScalar>(info_.cancelation_factor);
            return *this;
        }
    };


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