#include <fmt/format.h>

#include <array>
#include <cassert>
#include <complex>
#include <filesystem>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace testfmt {
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
    concept string_like = std::is_convertible_v<T, std::string_view>;

    template<typename T>
    concept path_like = std::same_as<bare_t<T>, std::filesystem::path>;

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
    complex_view<T> cx(const std::complex<T> &value) {
        return {value};
    }

    template<typename T>
    struct optional_view {
        const std::optional<T> &value;
        std::string_view        empty_text = "<nullopt>";
    };

    template<typename T>
    optional_view<T> opt(const std::optional<T> &value, std::string_view empty_text = "<nullopt>") {
        return {value, empty_text};
    }

    struct path_view {
        const std::filesystem::path &value;
    };

    inline path_view path(const std::filesystem::path &value) {
        return {value};
    }

    template<typename Range>
    struct range_view {
        const Range      &range;
        std::string_view  prefix    = "[";
        std::string_view separator = ", ";
        std::string_view  suffix    = "]";
    };

    template<typename Range>
    range_view<Range> joined(const Range &range, std::string_view separator = ", ") {
        return {range, "", separator, ""};
    }

    template<typename Range>
    range_view<Range> listed(const Range &range, std::string_view separator = ", ", std::string_view prefix = "[",
                             std::string_view suffix = "]") {
        return {range, prefix, separator, suffix};
    }

    template<typename T>
    auto adapt_for_format(const T &value);

    template<typename T>
    std::string format_one(const T &value, std::string_view spec) {
        return fmt::format(fmt::runtime(make_pattern(spec)), adapt_for_format(value));
    }

    template<typename T>
    auto adapt_for_format(const T &value) {
        if constexpr(is_std_complex_v<T>) {
            return cx(value);
        } else if constexpr(path_like<T>) {
            return path(value);
        } else if constexpr(is_std_optional_v<T>) {
            return opt(value);
        } else if constexpr(range_like<T>) {
            return listed(value);
        } else {
            return value;
        }
    }
}

template<typename T>
struct fmt::formatter<testfmt::complex_view<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const testfmt::complex_view<T> &value, format_context &ctx) const {
        auto real = testfmt::format_one(value.value.real(), spec_);
        auto imag = testfmt::format_one(value.value.imag(), spec_);
        return fmt::format_to(ctx.out(), "({},{})", real, imag);
    }
};

template<>
struct fmt::formatter<testfmt::path_view> : fmt::formatter<std::string_view> {
    auto format(const testfmt::path_view &value, format_context &ctx) const {
        auto text = value.value.string();
        return fmt::formatter<std::string_view>::format(text, ctx);
    }
};

template<typename T>
struct fmt::formatter<testfmt::optional_view<T>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const testfmt::optional_view<T> &value, format_context &ctx) const {
        if(!value.value) return fmt::format_to(ctx.out(), "{}", value.empty_text);
        return fmt::format_to(ctx.out(), "{}", testfmt::format_one(*value.value, spec_));
    }
};

template<typename Range>
struct fmt::formatter<testfmt::range_view<Range>> {
    std::string spec_;

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin();
        while(it != ctx.end() && *it != '}') spec_.push_back(*it++);
        return it;
    }

    auto format(const testfmt::range_view<Range> &value, format_context &ctx) const {
        auto out = ctx.out();
        out      = fmt::format_to(out, "{}", value.prefix);
        auto elem_spec = testfmt::element_spec(spec_);
        bool first = true;
        for(const auto &elem : value.range) {
            if(!first) out = fmt::format_to(out, "{}", value.separator);
            first = false;
            out   = fmt::format_to(out, "{}", testfmt::format_one(elem, elem_spec));
        }
        out = fmt::format_to(out, "{}", value.suffix);
        return out;
    }
};

int main() {
    const auto z         = std::complex<double>(1.25, -2.5);
    const auto p         = std::filesystem::path("/tmp/results/output.h5");
    const auto od        = std::optional<double>(3.14159);
    const auto os        = std::optional<std::string>("state_real");
    const auto onone     = std::optional<double>{};
    const auto names     = std::vector<std::string>{"alpha", "beta", "gamma"};
    const auto idxs      = std::vector<size_t>{2, 4, 8};
    const auto coeffs    = std::array<double, 3>{0.25, 0.5, 0.75};
    const auto h5paths   = std::vector<std::filesystem::path>{"/tmp/a.h5", "/tmp/b.h5"};
    const auto opt_ids   = std::optional<std::vector<size_t>>(idxs);
    const auto opt_paths = std::optional<std::vector<std::filesystem::path>>(h5paths);
    const auto phases    = std::vector<std::complex<double>>{{1.0, 0.5}, {-2.0, 3.0}};

    const auto z_str         = fmt::format("{:.2f}", testfmt::cx(z));
    const auto p_str         = fmt::format("{}", testfmt::path(p));
    const auto od_str        = fmt::format("{:.3f}", testfmt::opt(od));
    const auto os_str        = fmt::format("{}", testfmt::opt(os));
    const auto onone_str     = fmt::format("{}", testfmt::opt(onone));
    const auto names_str     = fmt::format("{}", testfmt::joined(names, " | "));
    const auto idxs_str      = fmt::format("{}", testfmt::joined(idxs, ", "));
    const auto coeffs_str    = fmt::format("{:.2f}", testfmt::joined(coeffs, ", "));
    const auto coeffs_rspec  = fmt::format("{::+9.2f}", testfmt::joined(coeffs, ", "));
    const auto path_list_str = fmt::format("{}", testfmt::listed(h5paths));
    const auto opt_ids_str   = fmt::format("{}", testfmt::opt(opt_ids));
    const auto opt_path_str  = fmt::format("{}", testfmt::opt(opt_paths));
    const auto phases_str    = fmt::format("{:.1f}", testfmt::listed(phases));
    const auto phases_rspec  = fmt::format("{::+5.1f}", testfmt::listed(phases));

    assert(z_str == "(1.25,-2.50)");
    assert(p_str == "/tmp/results/output.h5");
    assert(od_str == "3.142");
    assert(os_str == "state_real");
    assert(onone_str == "<nullopt>");
    assert(names_str == "alpha | beta | gamma");
    assert(idxs_str == "2, 4, 8");
    assert(coeffs_str == "0.25, 0.50, 0.75");
    assert(coeffs_rspec == "    +0.25,     +0.50,     +0.75");
    assert(path_list_str == "[/tmp/a.h5, /tmp/b.h5]");
    assert(opt_ids_str == "[2, 4, 8]");
    assert(opt_path_str == "[/tmp/a.h5, /tmp/b.h5]");
    assert(phases_str == "[(1.0,0.5), (-2.0,3.0)]");
    assert(phases_rspec == "[( +1.0, +0.5), ( -2.0, +3.0)]");

    fmt::print("complex   : {}\n", z_str);
    fmt::print("path      : {}\n", p_str);
    fmt::print("opt value : {}\n", od_str);
    fmt::print("opt nil   : {}\n", onone_str);
    fmt::print("join s    : {}\n", names_str);
    fmt::print("join n    : {}\n", coeffs_str);
    fmt::print("join n 2  : {}\n", coeffs_rspec);
    fmt::print("path list : {}\n", path_list_str);
    fmt::print("opt ids   : {}\n", opt_ids_str);
    fmt::print("opt paths : {}\n", opt_path_str);
    fmt::print("complexes : {}\n", phases_str);
    fmt::print("complexes2: {}\n", phases_rspec);
    return 0;
}
