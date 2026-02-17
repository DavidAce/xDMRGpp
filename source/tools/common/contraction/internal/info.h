#pragma once
#include <limits>
#include <type_traits>
namespace tools::common::contraction::internal {
    template<typename Scalar>
    struct StatsMv {
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
        StatsMv<Scalar> &operator=(const StatsMv<T> &stats_) {
            this->contract_left      = stats_.contract_left;
            this->mps_norm           = static_cast<RealScalar>(stats_.mps_norm);
            this->mpo_norm           = static_cast<RealScalar>(stats_.mpo_norm);
            this->envL_norm          = static_cast<RealScalar>(stats_.envL_norm);
            this->envR_norm          = static_cast<RealScalar>(stats_.envR_norm);
            this->ST1                = static_cast<RealScalar>(stats_.ST1);
            this->ST2                = static_cast<RealScalar>(stats_.ST2);
            this->ST3                = static_cast<RealScalar>(stats_.ST3);
            this->cancelation_factor = static_cast<RealScalar>(stats_.cancelation_factor);
            return *this;
        }
    };
    template<typename Scalar>
    struct StatsEnv {
        using RealScalar              = decltype(std::real(std::declval<Scalar>()));
        RealScalar mps_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar mpo_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar env_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar res_norm           = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST1                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar ST2                = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar cancelation_factor = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar highprec_threshold = RealScalar{10} / std::sqrt(std::numeric_limits<RealScalar>::epsilon());
        template<typename T>
        StatsEnv<Scalar> &operator=(const StatsEnv<T> &stats_) {
            this->mps_norm           = static_cast<RealScalar>(stats_.mps_norm);
            this->mpo_norm           = static_cast<RealScalar>(stats_.mpo_norm);
            this->env_norm           = static_cast<RealScalar>(stats_.env_norm);
            this->res_norm           = static_cast<RealScalar>(stats_.res_norm);
            this->ST1                = static_cast<RealScalar>(stats_.ST1);
            this->ST2                = static_cast<RealScalar>(stats_.ST2);
            this->cancelation_factor = static_cast<RealScalar>(stats_.cancelation_factor);
            return *this;
        }
    };

}