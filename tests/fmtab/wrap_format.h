#pragma once

#include "payload.h"

#include <cmath>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace fmtab::wrapfmt {
    template<typename T>
    using bare_t = std::remove_cvref_t<T>;

    template<typename T>
    struct is_std_optional : std::false_type {};

    template<typename T>
    struct is_std_optional<std::optional<T>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_std_optional_v = is_std_optional<bare_t<T>>::value;

    template<typename T>
    struct is_std_complex : std::false_type {};

    template<typename T>
    struct is_std_complex<std::complex<T>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_std_complex_v = is_std_complex<bare_t<T>>::value;

    template<typename T>
    concept path_like = std::same_as<bare_t<T>, std::filesystem::path>;

    template<typename T>
    concept string_like = std::convertible_to<T, std::string_view>;

    template<typename T>
    concept range_like =
        requires(const bare_t<T> &value) {
            std::begin(value);
            std::end(value);
        } && !string_like<T> && !path_like<T>;

    inline std::string make_pattern(std::string_view spec) {
        return spec.empty() ? std::string("{}") : fmt::format("{{:{}}}", spec);
    }

    inline std::string_view element_spec(std::string_view spec) {
        if(spec.starts_with(':')) return spec.substr(1);
        return spec;
    }

    template<typename T>
    struct complex_view {
        const std::complex<T> &value;
    };

    template<typename T>
    struct optional_view {
        const std::optional<T> &value;
    };

    struct path_view {
        const std::filesystem::path &value;
    };

    template<typename Range>
    struct range_view {
        const Range &range;
    };

    template<typename T>
    complex_view<T> adapt_complex(const std::complex<T> &value) {
        return {value};
    }

    template<typename T>
    optional_view<T> adapt_optional(const std::optional<T> &value) {
        return {value};
    }

    inline path_view adapt_path(const std::filesystem::path &value) {
        return {value};
    }

    template<typename Range>
    range_view<Range> adapt_range(const Range &value) {
        return {value};
    }

    template<typename T>
    auto adapt(const T &value);

    template<typename T>
    std::string format_one(const T &value, std::string_view spec) {
        return fmt::format(fmt::runtime(make_pattern(spec)), adapt(value));
    }

    inline std::string quote_string(std::string_view value) {
        return fmt::format("\"{}\"", value);
    }

    inline std::string strip_width(std::string_view spec) {
        auto result = std::string{spec};
        auto dot    = result.find('.');
        if(dot == std::string::npos || dot == 0) return result;
        auto first_digit = result.find_first_of("0123456789");
        if(first_digit == std::string::npos || first_digit >= dot) return result;
        result.erase(first_digit, dot - first_digit);
        return result;
    }

    template<typename T>
    auto adapt(const T &value) {
        if constexpr(is_std_complex_v<T>) {
            return adapt_complex(value);
        } else if constexpr(path_like<T>) {
            return adapt_path(value);
        } else if constexpr(is_std_optional_v<T>) {
            return adapt_optional(value);
        } else if constexpr(range_like<T>) {
            return adapt_range(value);
        } else {
            return value;
        }
    }
}

template<typename T>
struct fmt::formatter<fmtab::wrapfmt::complex_view<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtab::wrapfmt::complex_view<T> &value, format_context &ctx) const {
        auto scalar_spec = fmtab::wrapfmt::strip_width(spec_);
        auto imag_spec   = scalar_spec.find('+') == std::string::npos ? "+" + scalar_spec : scalar_spec;
        auto real        = fmtab::wrapfmt::format_one(value.value.real(), scalar_spec);
        auto imag        = fmtab::wrapfmt::format_one(value.value.imag(), imag_spec);

        if(value.value.real() != T{}) {
            auto out = ctx.out();
            out      = fmt::format_to(out, "(");
            out      = fmt::format_to(out, "{}", real);
            out      = fmt::format_to(out, "{}", imag);
            out      = fmt::format_to(out, "i)");
            return out;
        }

        auto out = ctx.out();
        out      = fmt::format_to(out, "{}", imag);
        out      = fmt::format_to(out, "i");
        return out;
    }
};

template<>
struct fmt::formatter<fmtab::wrapfmt::path_view> : fmt::formatter<std::string_view> {
    auto format(const fmtab::wrapfmt::path_view &value, format_context &ctx) const {
        auto text = value.value.string();
        return fmt::formatter<std::string_view>::format(text, ctx);
    }
};

template<typename T>
struct fmt::formatter<fmtab::wrapfmt::optional_view<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtab::wrapfmt::optional_view<T> &value, format_context &ctx) const {
        if(!value.value) return fmt::format_to(ctx.out(), "none");
        auto out = ctx.out();
        out      = fmt::format_to(out, "optional(");
        out      = fmt::format_to(out, "{}", fmtab::wrapfmt::format_one(*value.value, spec_));
        out      = fmt::format_to(out, ")");
        return out;
    }
};

template<typename Range>
struct fmt::formatter<fmtab::wrapfmt::range_view<Range>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtab::wrapfmt::range_view<Range> &value, format_context &ctx) const {
        auto out       = ctx.out();
        auto elem_spec = fmtab::wrapfmt::element_spec(spec_);
        out            = fmt::format_to(out, "[");
        bool first     = true;
        for(const auto &elem : value.range) {
            if(!first) out = fmt::format_to(out, ", ");
            first = false;
            if constexpr(fmtab::wrapfmt::path_like<decltype(elem)> || fmtab::wrapfmt::string_like<decltype(elem)>) {
                out = fmt::format_to(out, "{}", fmtab::wrapfmt::quote_string(fmtab::wrapfmt::format_one(elem, elem_spec)));
            } else {
                out = fmt::format_to(out, "{}", fmtab::wrapfmt::format_one(elem, elem_spec));
            }
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

namespace fmtab {
    inline std::string render_wrap_report(int unit) {
        auto payload = make_payload(unit);
        auto lines   = std::string{};
        lines += fmt::format("unit {:02d}\n", unit);
        lines += fmt::format("path {}\n", wrapfmt::adapt(payload.path));
        lines += fmt::format("paths {}\n", wrapfmt::adapt(payload.paths));
        lines += fmt::format("opt_paths {}\n", wrapfmt::adapt(payload.opt_paths));
        lines += fmt::format("ids {}\n", wrapfmt::adapt(payload.ids));
        lines += fmt::format("opt_ids {}\n", wrapfmt::adapt(payload.opt_ids));
        lines += fmt::format("coeffs {::+9.2f}\n", wrapfmt::adapt(payload.coeffs));
        lines += fmt::format("phases {::+6.2f}\n", wrapfmt::adapt(payload.phases));
        lines += fmt::format("opt_phase {:+7.3f}\n", wrapfmt::adapt(payload.opt_phase));
        lines += fmt::format("path_list_again {}\n", wrapfmt::adapt(payload.paths));
        lines += fmt::format("coeffs_again {::+9.2f}\n", wrapfmt::adapt(payload.coeffs));
        lines += fmt::format("phases_again {::+6.2f}\n", wrapfmt::adapt(payload.phases));
        return lines;
    }
}
