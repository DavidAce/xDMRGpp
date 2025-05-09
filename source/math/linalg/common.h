#pragma once
#include "math/tenx/fwd_decl.h"

namespace Eigen {
    template<typename Idx>
    struct IndexPair;
}

namespace linalg {
    template<typename T>
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename T>
    using RealScalar = decltype(std::real(std::declval<T>()));
    template<typename T>
    using CplxScalar = std::complex<RealScalar<T>>;
    template<typename TA, typename TB>
    using CommonRealScalar = std::common_type_t<RealScalar<TA>, RealScalar<TB>>;

    template<typename Derived>
    using is_PlainObject = std::is_base_of<Eigen::PlainObjectBase<std::decay_t<Derived>>, std::decay_t<Derived>>;

    template<template<class...> class Template, class... Args>
    void is_specialization_impl(const Template<Args...> &);
    template<class T, template<class...> class Template>
    concept is_specialization_v = requires(const T &t) { is_specialization_impl<Template>(t); };

    template<typename T>
    concept is_std_complex_v = is_specialization_v<T, std::complex>;

    template<typename TA, typename TB>
    using cplx_or_real_t = std::conditional_t<is_std_complex_v<TA> or is_std_complex_v<TB>, //
                                                     CplxScalar<CommonRealScalar<TA, TB>>,         //
                                                     RealScalar<CommonRealScalar<TA, TB>>>;

}