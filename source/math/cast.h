#pragma once

#include <cmath>
#include <complex>
#include <concepts>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

[[noreturn]] inline void cast_fail(const char *msg) noexcept {
    std::fprintf(stderr, "cast failed: %s\n", msg);
    std::abort();
}

template<class To, class From>
requires std::is_arithmetic_v<std::remove_cvref_t<To>> && std::is_arithmetic_v<std::remove_cvref_t<From>>
inline std::remove_cvref_t<To> safe_cast(From x) noexcept {
#ifdef NDEBUG
    return static_cast<std::remove_cvref_t<To>>(x);
#else
    using T = std::remove_cvref_t<To>;
    using F = std::remove_cvref_t<From>;

    if constexpr(std::is_same_v<T, F>) {
        return x;

    } else if constexpr(std::integral<T> && std::integral<F>) {
        if(!std::in_range<T>(x)) cast_fail("integral -> integral out of range");
        return static_cast<T>(x);

    } else if constexpr(std::integral<T> && std::floating_point<F>) {
        // Reject NaN/Inf for float->int
        if(!std::isfinite(x)) cast_fail("float -> integral NaN/Inf");

        const long double v  = static_cast<long double>(x);
        const long double lo = static_cast<long double>(std::numeric_limits<T>::min());
        const long double hi = static_cast<long double>(std::numeric_limits<T>::max());
        if(v < lo || v > hi) cast_fail("float -> integral out of range");

        // Keep static_cast semantics: it truncates.
        // If you want to forbid truncation, uncomment:
        // if(std::trunc(x) != x) cast_fail("float -> integral would truncate");

        return static_cast<T>(x);

    } else if constexpr(std::floating_point<T> && std::integral<F>) {
        // Allow precision loss, but prevent overflow to Inf (range check)
        const long double v  = static_cast<long double>(x);
        const long double lo = static_cast<long double>(std::numeric_limits<T>::lowest());
        const long double hi = static_cast<long double>(std::numeric_limits<T>::max());
        if(v < lo || v > hi) cast_fail("integral -> float out of range");
        return static_cast<T>(x);

    } else if constexpr(std::floating_point<T> && std::floating_point<F>) {
        // Keep NaN/Inf, but range-check finite values
        if(std::isfinite(x)) {
            const long double v  = static_cast<long double>(x);
            const long double lo = static_cast<long double>(std::numeric_limits<T>::lowest());
            const long double hi = static_cast<long double>(std::numeric_limits<T>::max());
            if(v < lo || v > hi) cast_fail("float -> float out of range");
        }
        return static_cast<T>(x);

    } else {
        return static_cast<T>(x);
    }
#endif
}
namespace detail {
    template<class T>
    struct is_std_complex : std::false_type {};
    template<class U>
    struct is_std_complex<std::complex<U>> : std::true_type {};
    template<class T>
    inline constexpr bool is_std_complex_v = is_std_complex<std::remove_cvref_t<T>>::value;

    template<class T>
    struct real_type {
        using type = std::remove_cvref_t<T>;
    };
    template<class U>
    struct real_type<std::complex<U>> {
        using type = U;
    };
    template<class T>
    using real_type_t = typename real_type<std::remove_cvref_t<T>>::type;

    template<class ToReal, class FromReal>
    [[nodiscard]] inline ToReal narrow_cast_fp(FromReal x) noexcept {
        static_assert(std::floating_point<ToReal>);
        static_assert(std::floating_point<FromReal>);

#ifndef NDEBUG
        // Range check finite values only. Keep NaN/Inf.
        if(std::isfinite(x)) {
            const long double v  = static_cast<long double>(x);
            const long double lo = static_cast<long double>(std::numeric_limits<ToReal>::lowest());
            const long double hi = static_cast<long double>(std::numeric_limits<ToReal>::max());
            if(v < lo || v > hi) cast_fail("narrow_cast: float out of range");
        }
#endif

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wfloat-conversion"
#elif defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-conversion"
#elif defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4244) // possible loss of data
#endif

        const ToReal y = static_cast<ToReal>(x);

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
#elif defined(_MSC_VER)
    #pragma warning(pop)
#endif

        return y;
    }

    template<class T>
    concept real_or_complex_fp = (std::floating_point<std::remove_cvref_t<T>> || (is_std_complex_v<T> && std::floating_point<real_type_t<T>>) );
}

template<class To, class From>
requires detail::real_or_complex_fp<To> && detail::real_or_complex_fp<From>
[[nodiscard]] inline std::remove_cvref_t<To> narrow_cast(From x) noexcept {
    using T        = std::remove_cvref_t<To>;
    using F        = std::remove_cvref_t<From>;
    using ToReal   = detail::real_type_t<T>;
    using FromReal = detail::real_type_t<F>;

    if constexpr(detail::is_std_complex_v<T> && detail::is_std_complex_v<F>) {
        const ToReal re = detail::narrow_cast_fp<ToReal>(static_cast<FromReal>(x.real()));
        const ToReal im = detail::narrow_cast_fp<ToReal>(static_cast<FromReal>(x.imag()));
        return T{re, im};

    } else if constexpr(detail::is_std_complex_v<T> && !detail::is_std_complex_v<F>) {
        const ToReal re = detail::narrow_cast_fp<ToReal>(static_cast<FromReal>(x));
        return T{re, ToReal{0}};

    } else if constexpr(!detail::is_std_complex_v<T> && detail::is_std_complex_v<F>) {
        // Must be enforced in both debug and release.
        if(static_cast<FromReal>(x.imag()) != FromReal{0}) cast_fail("narrow_cast: complex -> real requires imag == 0");

        return detail::narrow_cast_fp<T>(static_cast<FromReal>(x.real()));

    } else {
        return detail::narrow_cast_fp<T>(static_cast<FromReal>(x));
    }
}