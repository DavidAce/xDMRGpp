#pragma once
#include <complex>
namespace eig::sfinae {
    template<class>
    inline constexpr bool invalid_type_v = false;
    // A bit of sfinae deduction
    template<typename T1, typename T2>
    concept type_is = std::same_as<std::remove_cvref_t<T1>, T2>;
    template<typename T, typename... Ts>
    concept is_any_v = (type_is<T, Ts> || ...);

    template<typename T, typename = std::void_t<>>
    struct has_RawEigenvaluesImag : public std::false_type {};
    template<typename T>
    struct has_RawEigenvaluesImag<T, std::void_t<decltype(std::declval<T>().size())>> : public std::true_type {};
    template<typename T>
    inline constexpr bool has_RawEigenvaluesImag_v = has_RawEigenvaluesImag<T>::value;

    template<template<class...> class Template, class... Args>
    void is_specialization_impl(const Template<Args...> &);
    template<class T, template<class...> class Template>
    concept is_specialization_v = requires(const T &t) { is_specialization_impl<Template>(t); };

    template<typename T>
    concept is_std_complex_v = is_specialization_v<T, std::complex>;

    template<typename T>
    concept is_single_prec_v = type_is<T, fp32> or type_is<T, cx32>;

    template<typename T>
    concept is_double_prec_v = type_is<T, fp64> or type_is<T, cx64>;

    template<typename T>
    concept is_quadruple_prec_v = type_is<T, fp128> or type_is<T, cx128>;

    template<typename T>
    constexpr auto type_name() {
        std::string_view name, prefix, suffix;
#ifdef __clang__
        name   = __PRETTY_FUNCTION__;
        prefix = "auto eig::sfinae::type_name() [T = ";
        suffix = "]";
#elif defined(__GNUC__)
        name   = __PRETTY_FUNCTION__;
        prefix = "constexpr auto eig::sfinae::type_name() [with T = ";
        suffix = "]";
#elif defined(_MSC_VER)
        name   = __FUNCSIG__;
        prefix = "auto __cdecl eig::sfinae::type_name<";
        suffix = ">(void)";
#endif
        name.remove_prefix(prefix.size());
        name.remove_suffix(suffix.size());
        return name;
    }

}