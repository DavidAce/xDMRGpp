#pragma once
#if defined(FMT_HEADER_ONLY)
    #pragma message "{fmt} has been included as header-only library. This causes large compile-time overhead"
#endif

#if !defined(SPDLOG_FMT_EXTERNAL)
    #define SPDLOG_FMT_EXTERNAL
#endif
#if !defined(SPDLOG_COMPILED_LIB)
    #if !defined(SPDLOG_HEADER_ONLY)
        #define SPDLOG_HEADER_ONLY
    #endif
    #pragma message "SPDLOG_COMPILED_LIB is not defined. Define it and link libspdlog to speed up compilation"
#endif

#include "tools/common/log.h"
#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#if !defined(FMT_USE_COMPLEX) && FMT_VERSION > 110000
    #define FMT_USE_COMPLEX 1
    #include <complex>
// #include <type_traits>

// template<typename T, typename Char>
// struct fmt::formatter<std::complex<T>, Char> : fmt::formatter<T, Char> {
//     private:
//     typedef fmt::formatter<T, Char> base;
//     fmt::detail::dynamic_format_specs<Char> specs_;
//     public:
//     template<typename FormatCtx>
//     auto format(const std::complex<T> &x, FormatCtx &ctx) const -> decltype(ctx.out()) {
//         base::format(x.real(), ctx);
//         if(x.imag() >= 0 && specs_.sign != sign::plus) fmt::format_to(ctx.out(), "+");
//         base::format(x.imag(), ctx);
//         return fmt::format_to(ctx.out(), "i");
//     }
// };

#elif !defined(FMT_USE_COMPLEX) && FMT_VERSION > 90000
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

// #if defined(FMT_FORMAT_H_) && !defined(FMT_USE_COMPLEX)
//     #define FMT_USE_COMPLEX 1
//     #include <complex>
//     #include <type_traits>
// template<typename T, typename Char>
// struct fmt::formatter<std::complex<T>, Char> : fmt::formatter<T, Char> {
//     private:
//     typedef fmt::formatter<T, Char> base;
//     fmt::detail::dynamic_format_specs<Char> specs_;
//     public:
//     template<typename FormatCtx>
//     auto format(const std::complex<T> &x, FormatCtx &ctx) const -> decltype(ctx.out()) {
//         base::format(x.real(), ctx);
//         if(x.imag() >= 0 && specs_.sign != sign::plus) format_to(ctx.out(), "+");
//         base::format(x.imag(), ctx);
//         return format_to(ctx.out(), "i");
//     }
// };
// #endif

namespace eig {
    inline tools::LoggerHandle log = spdlog::get("eig") == nullptr ? spdlog::stdout_color_mt("eig", spdlog::color_mode::always) : spdlog::get("eig");
    extern void                setLevel(spdlog::level::level_enum level);
    extern void                setLevel(size_t level);
    extern void                setTimeStamp(std::string_view stamp = "[%Y-%m-%d %H:%M:%S][%n]%^[%=8l]%$ %v");
}
