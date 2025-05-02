#pragma once

#include <cassert>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <stdfloat>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
// This header defines the floating point types used throughout this program

using fp32 = float;
using fp64 = double;
using cx32 = std::complex<fp32>;
using cx64 = std::complex<fp64>;

#if defined(DMRG_USE_QUADMATH)
    #include <quadmath.h>
__extension__ typedef __float128 fp128;
using cx128 = std::complex<fp128>;
#elif defined(DMRG_USE_FLOAT128)
    #include <stdfloat>
using fp128 = std::float128_t;
using cx128 = std::complex<fp128>;
#else
using fp128 = long double;
using cx128 = std::complex<long double>;
#endif

namespace sfinae {
    template<typename T>
    concept is_native_float = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, long double>;

    template<typename T>
    concept is_extended_float = std::is_floating_point_v<T> && !is_native_float<T>;
}

template<typename Lhs, typename Rhs>
bool cmp_t(Lhs lhs, Rhs rhs) {
    if constexpr(std::is_same_v<Lhs, cx128> or std::is_same_v<Rhs, cx128>) {
        bool real_eq = static_cast<fp128>(std::real(lhs)) == static_cast<fp128>(std::real(rhs));
        bool imag_eq = static_cast<fp128>(std::imag(lhs)) == static_cast<fp128>(std::imag(rhs));
        return real_eq and imag_eq;
    } else if constexpr(std::is_same_v<Lhs, fp128> or std::is_same_v<Rhs, fp128>)
        return static_cast<fp128>(lhs) == static_cast<fp128>(rhs);
    else
        return lhs == rhs;
}


// template<typename T>
// auto abs_t(T val) -> fp128 {
//     if constexpr(std::is_arithmetic_v<T>) return std::abs(val);
// #if defined(DMRG_USE_QUADMATH)
//     else if constexpr(std::is_same_v<T, cx128>) {
//         __complex128 x;
//         __real__ x = val.real();
//         __imag__ x = val.imag();
//         return cabsq(x);
//     } else if constexpr(std::is_same_v<T, fp128>)
//         return fabsq(val);
// #else
//     else
//         return std::abs(val);
// #endif
// }


#if defined(DMRG_USE_QUADMATH)

inline const fp128 &conj(const fp128 &x) { return x; }
inline const fp128 &real(const fp128 &x) { return x; }
inline fp128        imag(const fp128 &) { return 0.; }
inline fp128        abs(const fp128 &x) { return fabsq(x); }
inline fp128        abs2(const fp128 &x) { return x * x; }

inline cx128 conj(const cx128 &x) {
    __complex128 y = conjq(*reinterpret_cast<const __complex128 *>(&x));
    return *reinterpret_cast<cx128 *>(&y);
}
inline fp128 real(const cx128 &x) { return crealq(*reinterpret_cast<const __complex128 *>(&x)); }
inline cx128 imag(const cx128 &x) { return cimagq(*reinterpret_cast<const __complex128 *>(&x)); }
inline fp128 abs(const cx128 &x) { return cabsq(*reinterpret_cast<const __complex128 *>(&x)); }
inline fp128 abs2(const cx128 &x) { return abs(conj(x) * x); }

inline fp128 pow(const fp128 &x, const fp128 &e) { return powq(x, e); }
inline fp128 log(const fp128 &x) { return logq(x); }
inline fp128 log10(const fp128 &x) { return log10q(x); }
inline bool  isnan(const fp128 &x) { return isnanq(x); }
inline bool  isinf(const fp128 &x) { return isinfq(x); }
inline fp128 floor(const fp128 &x) { return floorq(x); }
inline fp128 ceil(const fp128 &x) { return ceilq(x); }

#elif defined(DMRG_USE_FLOAT128)
// inline const fp128 &conj(const fp128 &x) { return x; }
// inline const fp128 &real(const fp128 &x) { return x; }
// inline fp128        imag(const fp128 &) { return 0.; }
// inline fp128        abs(const fp128 &x) { return abs_t(x); }
// inline fp128        abs2(const fp128 &x) { return x * x; }
//
// inline cx128 conj(const cx128 &x) { return cx128(x.real(), -x.imag()); }
// inline fp128 real(const cx128 &x) { return x.real(); }
// inline fp128 imag(const cx128 &) { return 0.; }
// inline fp128 abs(const cx128 &x) { return abs_t(x); }
// inline fp128 abs2(const cx128 &x) { return abs_t(conj(x) * x); }
// template<typename T>
// f128_base<T> operator*(const T &v) const {
// return val * v;
// }

#endif

// Wrapper around the float128 type

template<typename Scalar>
struct f128_base {
    protected:
    Scalar val = static_cast<Scalar>(0.0);

    public:
    virtual ~f128_base() = default;
    using format_type    = double;
    using value_type     = Scalar;
    f128_base()          = default;
    template<typename T>
    f128_base(const T &v) : val(v) {}
    operator Scalar() const { return val; }
    // operator double() const { return static_cast<double>(val); }
    [[nodiscard]] Scalar value() const { return val; }
    template<typename T>
    bool operator==(const T &v) const {
        static_assert(std::is_arithmetic_v<T> or std::is_same_v<T, Scalar>);
        if constexpr(std::is_same_v<T, f128_base>)
            return val == v.val;
        else
            return val == v;
    }
    template<typename T>
    f128_base<T> operator=(T v) {
        val = v;
        return val;
    }

    template<typename T>
    f128_base<T> operator*(const T &v) const {
        return val * v;
    }
    template<typename T>
    f128_base<T> operator/(const T &v) const {
        return val / v;
    }
    template<typename T>
    f128_base<T> operator+(const T &v) const {
        return val + v;
    }
    template<typename T>
    f128_base<T> operator-(const T &v) const {
        return val - v;
    }
    [[nodiscard]] virtual std::string string(int prec = 36, size_t width = 0, char pres = 'f', std::string_view align = "",
                                             std::string_view sign = "") const = 0;
};

#if defined(DMRG_USE_QUADMATH)
struct f128_t : f128_base<fp128> {
    using f128_base::f128_base;
    f128_t(std::string_view s) { val = strtoflt128(s.data(), nullptr); }
    f128_t(const char *c) : f128_t(std::string_view(c)) {}

    [[nodiscard]] std::string string(int prec = 36, size_t width = 0, char pres = 'f', std::string_view align = "", std::string_view sign = "") const final {
        if(prec < 0) prec = 36;
        char        fstr[16] = {0};
        const char *astr     = (align == "<") ? "-" : (align == ">" ? "" : align.data()); // alignment string
        auto        res      = std::snprintf(fstr, sizeof fstr, "%%%s%s%d.%uQ%c", astr, sign.data(), static_cast<int>(width), prec, pres);
        if(res < 0) throw std::runtime_error("f128_t.string(): snprintf() returned < 0");
        auto size = quadmath_snprintf(nullptr, 0, fstr, val);
        if(size < 0) { throw std::runtime_error("f128_t.string(): quadmath_snprintf() returned < 0"); }
        std::string buf;
        buf.resize(static_cast<size_t>(size));
        quadmath_snprintf(buf.data(), static_cast<size_t>(size + 1), fstr, val);
        return buf;
    }
};

struct c128_t : std::complex<f128_t> {
    using complex<f128_t>::complex;
    template<typename T>
    c128_t(const T &v) {
        this->real() = std::real(v);
        this->imag() = std::imag(v);
    }
};
#elif defined(DMRG_USE_FLOAT128)
struct f128_t : f128_base<fp128> {
    using f128_base::f128_base;
    f128_t(std::string_view s) {
        std::from_chars(s.data(), s.data() + s.size(), val);
        auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
        if(ec == std::errc::invalid_argument) throw std::runtime_error("f128_t(std::string_view s): invalid_argument");
        if(ec == std::errc::result_out_of_range) throw std::runtime_error("f128_t(std::string_view s): result_out_of_range");
    }
    [[nodiscard]] std::string string(int prec = 36, size_t width = 0, char pres = 'f', std::string_view align = "", std::string_view sign = "") const final {
        auto chfmt = std::chars_format::fixed;
        if(pres == 'f')
            chfmt = std::chars_format::fixed;
        else if(pres == 'e')
            chfmt = std::chars_format::scientific;
        else if(pres == 'g')
            chfmt = std::chars_format::general;
        else if(pres == 'h')
            chfmt = std::chars_format::hex;

        constexpr auto bufsize      = std::numeric_limits<fp128>::max_digits10 + 10;
        char           buf[bufsize] = {0};

        auto result = std::to_chars(buf, buf + bufsize, val, chfmt, prec);
        if(result.ec != std::errc()) throw std::runtime_error(std::make_error_code(result.ec).message());

        auto        length = strnlen(buf, bufsize);
        std::string sgn    = (buf[0] != '-' and sign == "+") ? "+" : "";
        if(sgn == "+") { length++; }
        width = std::max(length, width);
        std::string str;
        str.resize(width);
        assert(std::cmp_less(width - length, std::numeric_limits<int>::max()));
        assert(std::cmp_less(length, std::numeric_limits<int>::max()));
        int padLen = static_cast<int>(width - length); // Calc Padding length
        if(padLen < 0) padLen = 0;                     // Avoid negative length
        if(padLen == 0) {
            std::snprintf(str.data(), str.size() + 1, "%s%s", sgn.c_str(), buf); // No Padding
        } else if(align.find('<') != std::string_view::npos or align.find('-') != std::string_view::npos) {
            // Pad on right side
            std::string padding(padLen, ' ');
            std::snprintf(str.data(), str.size() + 1, "%s%s%*.*s", sgn.c_str(), buf, padLen, padLen, padding.c_str()); // RIGHT Padding

        } else {
            // pad on the left side
            std::string padding(padLen, ' ');
            std::snprintf(str.data(), str.size() + 1, "%*.*s%s%s", padLen, padLen, padding.c_str(), sgn.c_str(), buf); // LEFT Padding
        }
        return str;
    }
};

struct c128_t : std::complex<f128_t> {
    using complex<f128_t>::complex;
    template<typename T>
    c128_t(const T &v) {
        this->real() = std::real(v);
        this->imag() = std::imag(v);
    }
};

#else
// Dummy wrapper around the long double type
struct f128_t : f128_base<long double> {
    using f128_base::f128_base;
    f128_t(std::string_view s) { val = strtold(s.data(), nullptr); }
    f128_t(const char *c) : f128_t(std::string_view(c)) {}
    [[nodiscard]] std::string string(int prec = 36, size_t width = 0, char pres = 'f', std::string_view align = "", std::string_view sign = "") const final {
        char fstr[16];
        std::snprintf(fstr, sizeof fstr, "%%%s%s%d.%dL%c", align == "<" ? "-" : align.data(), sign.data(), static_cast<int>(width), prec, pres);
        auto size = std::snprintf(nullptr, 0ul, fstr, val);
        if(size < 0) throw std::runtime_error("f128_t.string(): quadmath_snprintf() returned size < 0");
        std::string buf;
        buf.resize(static_cast<size_t>(size));
        std::snprintf(buf.data(), buf.size() + 1ul, fstr, val);
        return buf;
    }
};
struct c128_t : std::complex<f128_t> {
    using complex<f128_t>::complex;
    template<typename T>
    c128_t(const T &v) {
        this->real() = std::real(v);
        this->imag() = std::imag(v);
    }
};
#endif
