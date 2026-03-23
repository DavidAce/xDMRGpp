#pragma once

#include "enum_choices_generated.h"
#include "settings.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <CLI/CLI.hpp>
#include <complex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace test::tomlpp {
    struct CliExit {
        int exit_code = 0;
    };

    namespace detail {
        template<typename T>
        struct is_std_vector : std::false_type {};

        template<typename T, typename Alloc>
        struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

        template<typename T>
        struct is_std_array : std::false_type {};

        template<typename T, std::size_t N>
        struct is_std_array<std::array<T, N>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_std_array_v = is_std_array<T>::value;

        template<typename T>
        struct is_std_complex : std::false_type {};

        template<typename T>
        struct is_std_complex<std::complex<T>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

        inline std::string trim(std::string_view text) {
            auto first = text.find_first_not_of(" \t");
            if(first == std::string_view::npos) return {};
            auto last = text.find_last_not_of(" \t");
            return std::string(text.substr(first, last - first + 1));
        }

        inline std::string long_cli_name(std::string_view names) {
            std::size_t start = 0;
            while(start < names.size()) {
                auto end   = names.find(',', start);
                auto token = trim(names.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
                if(token.rfind("--", 0) == 0) return token;
                if(end == std::string_view::npos) break;
                start = end + 1;
            }
            return {};
        }

        inline std::string lower_copy(std::string_view text) {
            std::string out(text);
            for(auto &ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            return out;
        }

        inline std::vector<std::string> split_pipe(std::string_view text) {
            std::vector<std::string> parts;
            std::size_t              start = 0;
            while(start <= text.size()) {
                auto end = text.find('|', start);
                auto len = end == std::string_view::npos ? std::string_view::npos : end - start;
                parts.emplace_back(trim(text.substr(start, len)));
                if(end == std::string_view::npos) break;
                start = end + 1;
            }
            return parts;
        }

        template<typename Range>
        inline std::string join_strings(const Range &items, std::string_view sep) {
            std::string out;
            std::size_t idx = 0;
            for(const auto &item : items) {
                if(idx > 0) out += sep;
                out += std::string_view(item);
                idx++;
            }
            return out;
        }

        template<typename Enum, typename = void>
        struct has_enum_info : std::false_type {};

        template<typename Enum>
        struct has_enum_info<Enum, std::void_t<decltype(generated::EnumInfo<Enum>::choices)>> : std::true_type {};

        template<typename Enum>
        inline constexpr bool has_enum_info_v = has_enum_info<Enum>::value;

        template<typename Enum>
        const auto &enum_choices() {
            static_assert(has_enum_info_v<Enum>, "Missing generated enum metadata for CLI11 help");
            return generated::EnumInfo<Enum>::choices;
        }

        template<typename Enum>
        std::string enum_type_name() {
            return "ENUM{" + join_strings(enum_choices<Enum>(), ",") + "}";
        }

        template<typename Enum>
        std::string canonicalize_enum_input(std::string_view input) {
            constexpr bool is_bitflag = generated::EnumInfo<Enum>::is_bitflag;
            const auto    &choices    = enum_choices<Enum>();
            const auto     allowed    = "{" + join_strings(choices, ",") + "}";
            auto           parts      = split_pipe(input);

            if(parts.empty()) throw std::runtime_error(std::string(input) + " not in " + allowed);
            if(!is_bitflag && parts.size() != 1) throw std::runtime_error(std::string(input) + " not in " + allowed);

            std::vector<std::string> canonical;
            canonical.reserve(parts.size());
            for(const auto &part : parts) {
                if(part.empty()) throw std::runtime_error(std::string(input) + " not in " + allowed);
                auto part_lower = lower_copy(part);
                auto match      = std::find_if(choices.begin(), choices.end(), [&](const auto &choice) { return lower_copy(choice) == part_lower; });
                if(match == choices.end()) throw std::runtime_error(std::string(input) + " not in " + allowed);
                canonical.emplace_back(std::string(*match));
            }
            return join_strings(canonical, "|");
        }

        template<typename Enum>
        CLI::Validator enum_validator() {
            CLI::Validator validator;
            validator.operation([](std::string &input) {
                try {
                    input = canonicalize_enum_input<Enum>(input);
                    return std::string{};
                } catch(const std::exception &ex) { return std::string(ex.what()); }
            });
            return validator;
        }
    }

    template<typename Spec>
    void bind_option(CLI::App &app, const Spec &spec) {
        using value_type = std::remove_reference_t<decltype(*spec.value)>;

        if constexpr(std::is_same_v<value_type, bool>) {
            auto names = std::string(spec.cli);
            if(auto long_name = detail::long_cli_name(spec.cli); not long_name.empty()) {
                names += ",!--no-";
                names += long_name.substr(2);
            }
            app.add_flag(names, *spec.value, std::string(spec.doc));
        } else if constexpr(std::is_enum_v<value_type>) {
            auto *opt = app.add_option_function<std::string>(
                std::string(spec.cli),
                [value = spec.value](const std::string &arg) { *value = sv2enum<value_type>(detail::canonicalize_enum_input<value_type>(arg)); },
                std::string(spec.doc));
            opt->type_name(detail::enum_type_name<value_type>());
            opt->check(detail::enum_validator<value_type>());
        } else if constexpr(detail::is_std_array_v<value_type>) {
            using elem_type              = typename value_type::value_type;
            constexpr auto expected_size = std::tuple_size_v<value_type>;
            auto          *opt           = app.add_option_function<std::vector<elem_type>>(
                std::string(spec.cli),
                [value = spec.value](const std::vector<elem_type> &items) {
                    if(items.size() != expected_size) throw std::runtime_error("Incorrect array size for CLI option");
                    for(std::size_t idx = 0; idx < expected_size; ++idx) (*value)[idx] = items[idx];
                },
                std::string(spec.doc));
            opt->expected(static_cast<int>(expected_size));
            opt->delimiter(',');
        } else if constexpr(detail::is_std_complex_v<value_type>) {
            using elem_type = typename value_type::value_type;
            auto *opt       = app.add_option_function<std::vector<elem_type>>(
                std::string(spec.cli),
                [value = spec.value](const std::vector<elem_type> &items) {
                    if(items.size() != 2) throw std::runtime_error("Complex CLI option requires exactly 2 entries");
                    *value = value_type{items[0], items[1]};
                },
                std::string(spec.doc));
            opt->expected(2);
            opt->delimiter(',');
        } else {
            auto *opt = app.add_option(std::string(spec.cli), *spec.value, std::string(spec.doc));
            if constexpr(detail::is_std_vector_v<value_type>) opt->delimiter(',');
            (void) opt;
        }
    }
}
