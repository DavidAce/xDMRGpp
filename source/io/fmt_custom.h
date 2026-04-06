#pragma once

#include "general/sfinae.h"
#include <charconv>
#include <complex>
#include <filesystem>
#include <fmt/format.h>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

template<typename T>
struct fp {
    public:
    T value          = static_cast<T>(0.0);
    using value_type = T;
    fp(T value_) : value(value_) {}
};

template<typename T>
requires std::floating_point<T>
class fv {
    public:
    using value_type = fp<T>;

    const fp<T> *ptr_;
    std::size_t  len_;

    template<typename size_type>
    requires std::is_integral_v<size_type>
    fv(T *ptr, size_type len) noexcept : ptr_{reinterpret_cast<const fp<T> *>(ptr)}, len_{static_cast<std::size_t>(len)} {}
    fv(const T *bgn, const T *end) noexcept : ptr_{reinterpret_cast<const fp<T> *>(bgn)}, len_{static_cast<std::size_t>(std::distance(bgn, end))} {}

    template<template<typename, auto...> typename V, auto... Args>
    fv(const V<T, Args...> &v) noexcept : ptr_(reinterpret_cast<const fp<T> *>(v.data())), len_(v.size()) {}

    template<template<typename, auto, auto, typename> typename V, auto a, auto b, typename c>
    fv(const V<T, a, b, c> &v) noexcept : ptr_(reinterpret_cast<const fp<T> *>(v.data())), len_(v.size()) {}

    template<template<typename, typename> typename V, typename A>
    fv(const V<T, A> &v) noexcept : ptr_(reinterpret_cast<const fp<T> *>(v.data())), len_(v.size()) {}

    template<typename V>
    fv(const V &v) noexcept : ptr_(reinterpret_cast<const fp<T> *>(v.data())), len_(static_cast<size_t>(v.size())) {}

    fp<T>       &operator[](size_t i) noexcept { return ptr_[i]; }
    fp<T> const &operator[](size_t i) const noexcept { return ptr_[i]; }
    [[nodiscard]] std::size_t size() const noexcept { return len_; }

    fp<T>       *data() noexcept { return ptr_; }
    fp<T>       *begin() noexcept { return ptr_; }
    fp<T>       *end() noexcept { return ptr_ + len_; }
    const fp<T> *data() const noexcept { return ptr_; }
    const fp<T> *begin() const noexcept { return ptr_; }
    const fp<T> *end() const noexcept { return ptr_ + len_; }
};

template<typename T>
requires std::is_same_v<T, typename fp<T>::value_type>
class fpv {
    public:
    using value_type = T;

    const T    *ptr_;
    std::size_t len_;

    template<typename size_type>
    requires std::is_integral_v<size_type>
    fpv(T *ptr, size_type len) noexcept : ptr_{ptr}, len_{static_cast<std::size_t>(len)} {}
    fpv(const T *bgn, const T *end) noexcept : ptr_{bgn}, len_{static_cast<std::size_t>(std::distance(bgn, end))} {}

    template<template<typename, auto...> typename V, auto... Args>
    fpv(const V<T, Args...> &v) noexcept : ptr_(v.data()), len_(v.size()) {}

    template<template<typename, auto, auto, typename> typename V, auto a, auto b, typename c>
    fpv(const V<T, a, b, c> &v) noexcept : ptr_(v.data()), len_(v.size()) {}

    template<template<typename, typename> typename V, typename A>
    fpv(const V<T, A> &v) noexcept : ptr_(v.data()), len_(v.size()) {}

    template<typename V>
    fpv(const V &v) noexcept : ptr_(v.data()), len_(static_cast<size_t>(v.size())) {}

    T                        &operator[](size_t i) noexcept { return ptr_[i]; }
    T const                  &operator[](size_t i) const noexcept { return ptr_[i]; }
    [[nodiscard]] std::size_t size() const noexcept { return len_; }

    T       *data() noexcept { return ptr_; }
    T       *begin() noexcept { return ptr_; }
    T       *end() noexcept { return ptr_ + len_; }
    const T *data() const noexcept { return ptr_; }
    const T *begin() const noexcept { return ptr_; }
    const T *end() const noexcept { return ptr_ + len_; }
};

namespace fmtwrap {
    template<typename T>
    using bare_t = std::remove_cvref_t<T>;

    template<typename T>
    struct is_project_wrapper : std::false_type {};

    template<typename T>
    struct is_project_wrapper<fp<T>> : std::true_type {};

    template<typename T>
    struct is_project_wrapper<fv<T>> : std::true_type {};

    template<typename T>
    struct is_project_wrapper<fpv<T>> : std::true_type {};

    template<typename T>
    struct complex_scalar_type {
        static constexpr bool value = false;
        using type                  = void;
    };

    template<typename T>
    struct complex_scalar_type<std::complex<T>> {
        static constexpr bool value = true;
        using type                  = T;
    };

    template<typename T>
    concept native_fp_scalar = std::same_as<bare_t<T>, float> || std::same_as<bare_t<T>, double> || std::same_as<bare_t<T>, long double>;

    template<typename T>
    concept extended_fp_scalar = std::floating_point<bare_t<T>> && !native_fp_scalar<T>;

    template<typename T>
    concept complex_floating_scalar = complex_scalar_type<bare_t<T>>::value && std::floating_point<typename complex_scalar_type<bare_t<T>>::type>;

    template<typename T>
    concept path_like = std::same_as<bare_t<T>, std::filesystem::path>;

    template<typename T>
    concept optional_like = sfinae::is_std_optional_v<bare_t<T>>;

    template<typename T>
    concept string_like = sfinae::is_text_v<bare_t<T>>;

    template<typename T>
    concept iterable_range = requires(const bare_t<T> &value) {
        std::begin(value);
        std::end(value);
    } && !string_like<T> && !path_like<T> && !is_project_wrapper<bare_t<T>>::value;

    template<typename T>
    concept indexed_range =
        !iterable_range<T> && !string_like<T> && !path_like<T> && !optional_like<T> && !is_project_wrapper<bare_t<T>>::value &&
        requires(const bare_t<T> &value) {
        { value.size() } -> std::convertible_to<std::size_t>;
        value[0];
    };

    template<typename T>
    concept contiguous_storage = requires(const bare_t<T> &value) {
        { value.data() };
        requires std::is_pointer_v<decltype(value.data())>;
        { value.size() } -> std::convertible_to<std::size_t>;
    };

    template<typename T>
    requires contiguous_storage<T>
    using range_value_t = std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const bare_t<T> &>().data())>>;

    template<typename T>
    concept contiguous_floating_range =
        contiguous_storage<T> && !iterable_range<T> && !sfinae::is_text_v<bare_t<T>> && !is_project_wrapper<bare_t<T>>::value &&
        std::floating_point<range_value_t<T>>;

    struct path_view {
        const std::filesystem::path &value;
    };

    template<typename T>
    struct optional_view {
        const std::optional<T> &value;
    };

    template<typename Range>
    struct listed_view {
        const Range &range;
    };

    template<typename Range>
    struct indexed_view {
        const Range &range;
    };

    template<typename Range>
    struct joined_view {
        const Range       &range;
        std::string_view   separator;
    };

    template<typename T>
    struct dense_view {
        const T          *ptr;
        std::size_t       len;
    };

    template<>
    struct is_project_wrapper<path_view> : std::true_type {};

    template<typename T>
    struct is_project_wrapper<optional_view<T>> : std::true_type {};

    template<typename Range>
    struct is_project_wrapper<listed_view<Range>> : std::true_type {};

    template<typename Range>
    struct is_project_wrapper<indexed_view<Range>> : std::true_type {};

    template<typename Range>
    struct is_project_wrapper<joined_view<Range>> : std::true_type {};

    template<typename T>
    struct is_project_wrapper<dense_view<T>> : std::true_type {};

    inline path_view path(const std::filesystem::path &value) { return {value}; }

    template<typename T>
    optional_view<T> opt(const std::optional<T> &value) {
        return {value};
    }

    template<typename Range>
    listed_view<Range> listed(const Range &value) {
        return {value};
    }

    template<typename Range>
    indexed_view<Range> indexed(const Range &value) {
        return {value};
    }

    template<typename Range>
    joined_view<Range> joined(const Range &value, std::string_view separator) {
        return {value, separator};
    }

    template<typename Range>
    dense_view<range_value_t<Range>> dense(const Range &value) {
        return {value.data(), static_cast<std::size_t>(value.size())};
    }

    inline std::string make_pattern(std::string_view spec) {
        return spec.empty() ? std::string("{}") : fmt::format("{{:{}}}", spec);
    }

    inline std::string_view element_spec(std::string_view spec) {
        if(spec.starts_with(':')) return spec.substr(1);
        return spec;
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

    template<typename T>
    auto adapt(const T &value) {
        if constexpr(is_project_wrapper<bare_t<T>>::value) {
            return value;
        } else if constexpr(extended_fp_scalar<T> || complex_floating_scalar<T>) {
            return fp<bare_t<T>>(value);
        } else if constexpr(path_like<T>) {
            return path(value);
        } else if constexpr(optional_like<T>) {
            return opt(value);
        } else if constexpr(indexed_range<T>) {
            return indexed(value);
        } else if constexpr(iterable_range<T>) {
            return listed(value);
        } else if constexpr(contiguous_floating_range<T>) {
            return dense(value);
        } else {
            return value;
        }
    }

    template<typename T, typename = void>
    struct adapted_arg {
        using type = bare_t<T>;
    };

    template<typename T>
    struct adapted_arg<T, std::enable_if_t<is_project_wrapper<bare_t<T>>::value>> {
        using type = bare_t<T>;
    };

    template<typename T>
    struct adapted_arg<T, std::enable_if_t<extended_fp_scalar<T> || complex_floating_scalar<T>>> {
        using type = fp<bare_t<T>>;
    };

    template<typename T>
    struct adapted_arg<T, std::enable_if_t<path_like<T>>> {
        using type = path_view;
    };

    template<typename T>
    struct adapted_arg<T, std::enable_if_t<optional_like<T>>> {
        using type = optional_view<typename bare_t<T>::value_type>;
    };

    template<typename T>
    struct adapted_arg<T, std::enable_if_t<indexed_range<T>>> {
        using type = indexed_view<bare_t<T>>;
    };

    template<typename T>
    struct adapted_arg<T, std::enable_if_t<iterable_range<T>>> {
        using type = listed_view<bare_t<T>>;
    };

    template<typename T>
    struct adapted_arg<T, std::enable_if_t<!iterable_range<T> && contiguous_floating_range<T>>> {
        using type = dense_view<range_value_t<T>>;
    };

    template<typename T>
    using adapted_arg_t = typename adapted_arg<T>::type;
}

namespace fmw {
    using fmtwrap::adapted_arg_t;

    template<typename T>
    constexpr decltype(auto) wrap(T &&value) {
        return fmtwrap::adapt(std::forward<T>(value));
    }

    template<typename Range>
    constexpr auto join(const Range &value, std::string_view separator) {
        return fmtwrap::joined(value, separator);
    }
}

template<typename T>
struct fmt::formatter<fp<T>> {
    fmt::detail::dynamic_format_specs<> specs_;

    template<typename ParseContext>
    FMT_CONSTEXPR auto parse(ParseContext &ctx) {
        auto type = detail::type_constant<double, char>::value;
        return detail::parse_format_specs(ctx.begin(), ctx.end(), specs_, ctx, type);
    }

    auto format(fp<T> value, format_context &ctx) const -> format_context::iterator {
        std::chars_format fmt_type;
        switch(specs_.type()) {
            case fmt::presentation_type::fixed: fmt_type = std::chars_format::fixed; break;
            case fmt::presentation_type::exp: fmt_type = std::chars_format::scientific; break;
            case fmt::presentation_type::general: fmt_type = std::chars_format::general; break;
            default: fmt_type = std::chars_format::general; break;
        }

        auto to_chars_internal = [&](const auto &v, bool force_plus_sign) -> std::string {
            using V = std::remove_cvref_t<decltype(v)>;
            static_assert(std::is_floating_point_v<V>);

            constexpr auto bsize = std::numeric_limits<V>::max_digits10 + std::numeric_limits<V>::max_exponent10 + 10;
            char           buffer[bsize];
            size_t         offset = 0;

            if(force_plus_sign && v >= V{0}) {
                buffer[0] = '+';
                offset    = 1;
            }

            std::to_chars_result result;
            if(specs_.precision >= 0) result = std::to_chars(buffer + offset, buffer + sizeof(buffer), v, fmt_type, specs_.precision);
            else result = std::to_chars(buffer + offset, buffer + sizeof(buffer), v, fmt_type);

            if(result.ec != std::errc{}) throw std::system_error{static_cast<int>(result.ec), std::system_category()};
            return std::string(buffer, result.ptr - buffer);
        };

        std::string value_string;
        if constexpr(sfinae::is_std_complex_v<T>) {
            auto real_string = to_chars_internal(std::real(value.value), specs_.sign() == fmt::sign::plus);
            auto imag_string = to_chars_internal(std::imag(value.value), std::real(value.value) != typename T::value_type{} || specs_.sign() == fmt::sign::plus);

            if(std::real(value.value) != typename T::value_type{}) value_string = fmt::format("({}{}i)", real_string, imag_string);
            else value_string = fmt::format("{}i", imag_string);
        } else {
            value_string = to_chars_internal(value.value, specs_.sign() == fmt::sign::plus);
        }

        switch(specs_.align()) {
            case fmt::align::left: return fmt::format_to(ctx.out(), "{:<{}}", value_string, specs_.width);
            case fmt::align::right: return fmt::format_to(ctx.out(), "{:>{}}", value_string, specs_.width);
            case fmt::align::center: return fmt::format_to(ctx.out(), "{:^{}}", value_string, specs_.width);
            default: return fmt::format_to(ctx.out(), "{:>{}}", value_string, specs_.width);
        }
    }
};

template<typename T>
struct fmt::formatter<fv<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fv<T> &value, format_context &ctx) const {
        auto out       = ctx.out();
        auto elem_spec = fmtwrap::element_spec(spec_);
        out            = fmt::format_to(out, "[");
        for(std::size_t idx = 0; idx < value.size(); ++idx) {
            if(idx != 0) out = fmt::format_to(out, ", ");
            out = fmt::format_to(out, "{}", fmtwrap::format_one(value[idx], elem_spec));
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template<typename T>
struct fmt::formatter<fpv<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fpv<T> &value, format_context &ctx) const {
        auto out       = ctx.out();
        auto elem_spec = fmtwrap::element_spec(spec_);
        out            = fmt::format_to(out, "[");
        for(std::size_t idx = 0; idx < value.size(); ++idx) {
            if(idx != 0) out = fmt::format_to(out, ", ");
            out = fmt::format_to(out, "{}", fmtwrap::format_one(value[idx], elem_spec));
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template<typename T>
struct fmt::formatter<fmtwrap::dense_view<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtwrap::dense_view<T> &value, format_context &ctx) const {
        auto out       = ctx.out();
        auto elem_spec = fmtwrap::element_spec(spec_);
        out            = fmt::format_to(out, "[");
        for(std::size_t idx = 0; idx < value.len; ++idx) {
            if(idx != 0) out = fmt::format_to(out, ", ");
            out = fmt::format_to(out, "{}", fmtwrap::format_one(value.ptr[idx], elem_spec));
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template<>
struct fmt::formatter<fmtwrap::path_view> : fmt::formatter<std::string_view> {
    auto format(const fmtwrap::path_view &value, format_context &ctx) const {
        auto text = value.value.string();
        return fmt::formatter<std::string_view>::format(text, ctx);
    }
};

template<typename T>
struct fmt::formatter<fmtwrap::optional_view<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtwrap::optional_view<T> &value, format_context &ctx) const {
        if(!value.value) return fmt::format_to(ctx.out(), "none");
        auto out = ctx.out();
        out      = fmt::format_to(out, "optional(");
        out      = fmt::format_to(out, "{}", fmtwrap::format_one(*value.value, spec_));
        out      = fmt::format_to(out, ")");
        return out;
    }
};

template<typename Range>
struct fmt::formatter<fmtwrap::listed_view<Range>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtwrap::listed_view<Range> &value, format_context &ctx) const {
        auto out       = ctx.out();
        auto elem_spec = fmtwrap::element_spec(spec_);
        out            = fmt::format_to(out, "[");
        bool first     = true;
        for(const auto &elem : value.range) {
            if(!first) out = fmt::format_to(out, ", ");
            first = false;
            if constexpr(fmtwrap::path_like<decltype(elem)> || fmtwrap::string_like<decltype(elem)>) {
                out = fmt::format_to(out, "{}", fmtwrap::quote_string(fmtwrap::format_one(elem, elem_spec)));
            } else {
                out = fmt::format_to(out, "{}", fmtwrap::format_one(elem, elem_spec));
            }
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template<typename Range>
struct fmt::formatter<fmtwrap::indexed_view<Range>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtwrap::indexed_view<Range> &value, format_context &ctx) const {
        auto out       = ctx.out();
        auto elem_spec = fmtwrap::element_spec(spec_);
        out            = fmt::format_to(out, "[");
        for(std::size_t idx = 0; idx < static_cast<std::size_t>(value.range.size()); ++idx) {
            if(idx != 0) out = fmt::format_to(out, ", ");
            out = fmt::format_to(out, "{}", fmtwrap::format_one(value.range[idx], elem_spec));
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template<typename Range>
struct fmt::formatter<fmtwrap::joined_view<Range>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const fmtwrap::joined_view<Range> &value, format_context &ctx) const {
        auto out       = ctx.out();
        auto elem_spec = fmtwrap::element_spec(spec_);
        bool first     = true;
        for(const auto &elem : value.range) {
            if(!first) out = fmt::format_to(out, "{}", value.separator);
            first = false;
            out   = fmt::format_to(out, "{}", fmtwrap::format_one(elem, elem_spec));
        }
        return out;
    }
};
