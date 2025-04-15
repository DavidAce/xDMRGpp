#pragma once

#include <fmt/format.h>
#include <general/sfinae.h>

#if defined(FMT_HEADER_ONLY)
    #pragma message "{fmt} has been included as header-only library. This causes large compile-time overhead"
#endif

#if (defined(FMT_FORMAT_H_) || defined(FMT_CORE_H_)) && !defined(FMT_USE_COMPLEX) && FMT_VERSION < 110000
    #define FMT_USE_COMPLEX 1
    #include <complex>
    #include <type_traits>

template<typename T, typename Char>
struct fmt::formatter<std::complex<T>, Char> : fmt::formatter<T, Char> {
    private:
    typedef fmt::formatter<T, Char>         base;
    fmt::detail::dynamic_format_specs<Char> specs_;

    public:
    template<typename FormatCtx>
    auto format(const std::complex<T> &x, FormatCtx &ctx) const -> decltype(ctx.out()) {
        base::format(x.real(), ctx);
        if(x.imag() >= 0 && specs_.sign != sign::plus) fmt::format_to(ctx.out(), "+");
        base::format(x.imag(), ctx);
        return fmt::format_to(ctx.out(), "i");
    }
};

#endif
#if (defined(FMT_FORMAT_H_) || defined(FMT_CORE_H_)) && !defined(FMT_USE_REFWRAP) && FMT_VERSION < 110103
    #define FMT_USE_REFWRAP 1
    #include <type_traits>

template<typename T>
struct fmt::formatter<std::reference_wrapper<T>> : fmt::formatter<T> {
    private:
    typedef fmt::formatter<T> base;

    public:
    template<typename FormatCtx>
    auto format(const std::reference_wrapper<T> &x, FormatCtx &ctx) const -> decltype(ctx.out()) {
        base::format(static_cast<T>(x), ctx);
        return ctx.out();
    }
};
#else
    #include <fmt/std.h>
#endif

template<typename T>
struct fp {
    public:
    T value = static_cast<T>(0.0);
    fp(T value_) : value(value_) {}
};

template<typename T>
struct fmt::formatter<fp<T>> {
    fmt::detail::dynamic_format_specs<> specs_;
    template<typename ParseContext>
    FMT_CONSTEXPR auto parse(ParseContext &ctx) {
        auto type = detail::type_constant<double, char>::value;
        auto end  = detail::parse_format_specs(ctx.begin(), ctx.end(), specs_, ctx, type);
        return end;
    }
    auto format(fp<T> value, format_context &ctx) const -> format_context::iterator {
        // Map the parsed presentation to std::chars_format.
        std::chars_format fmtType;
        switch(specs_.type()) {
            case fmt::presentation_type::fixed: fmtType = std::chars_format::fixed; break;
            case fmt::presentation_type::exp: fmtType = std::chars_format::scientific; break;
            case fmt::presentation_type::general: fmtType = std::chars_format::general; break;
            default: fmtType = std::chars_format::general; break;
        }
        std::string_view sign = specs_.sign() == fmt::sign::plus ? "+" : "";

        auto to_chars_internal = [&](const auto &v, std::string_view sign_internal) -> std::string {
            using V = std::remove_cvref_t<decltype(v)>;
            static_assert(std::is_floating_point_v<V>);

            constexpr auto bsize = std::numeric_limits<V>::max_digits10 + std::numeric_limits<V>::max_exponent10 + 10;
            char           buffer[bsize]; // Temporary buffer for conversion
            size_t         off = 0;
            if(!sign_internal.empty() and v >= V(0.0)) {
                buffer[0] = '+';
                off       = 1;
            }

            std::to_chars_result result;
            // C++23 std::to_chars for floating-point has an overload to accept precision.
            if(specs_.precision >= 0) {
                result = std::to_chars(buffer + off, buffer + sizeof(buffer), v, fmtType, specs_.precision);
            } else {
                // Passing -1 (or not passing a precision) tells std::to_chars to produce
                // the shortest representation.
                result = std::to_chars(buffer + off, buffer + sizeof(buffer), v, fmtType);
            }
            if(result.ec != std::errc{}) { throw std::system_error{static_cast<int>(result.ec), std::system_category()}; }

            // Create a string_view for the converted result.
            return std::string(buffer, result.ptr - buffer);
        };
        std::string valstr;
        if constexpr(sfinae::is_std_complex_v<T>) {
            valstr = fmt::format("({},{})", to_chars_internal(std::real(value.value), sign), to_chars_internal(std::imag(value.value), sign));
        } else {
            valstr = to_chars_internal(value.value, sign);
        }

        switch(specs_.align()) {
            case fmt::align::left: return fmt::format_to(ctx.out(), "{}{:<{}}", sign, valstr, specs_.width);
            case fmt::align::right: return fmt::format_to(ctx.out(), "{}{:>{}}", sign, valstr, specs_.width);
            case fmt::align::center: return fmt::format_to(ctx.out(), "{}{:^{}}", sign, valstr, specs_.width);
            default: return fmt::format_to(ctx.out(), "{:<{}}", valstr, specs_.width);
        }
    }
};