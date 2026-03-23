#include "config_io.h"
#include "debug/exceptions.h"
#include "setting_specs_generated.h"
#include <array>
#include <complex>
#include <fmt/core.h>
#include <string>
#include <toml.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {
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

    template<typename T>
    constexpr auto type_name() {
        if constexpr(std::is_same_v<T, std::string>)
            return "string";
        else if constexpr(std::is_same_v<T, bool>)
            return "bool";
        else if constexpr(std::is_enum_v<T>)
            return "enum";
        else if constexpr(is_std_vector_v<T>)
            return "array";
        else if constexpr(is_std_array_v<T>)
            return "fixed-size array";
        else if constexpr(is_std_complex_v<T>)
            return "complex";
        else if constexpr(std::is_integral_v<T>)
            return "integer";
        else if constexpr(std::is_floating_point_v<T>)
            return "floating point";
        else
            return "value";
    }

    template<typename T, typename Node>
    T read_node(const Node &node, std::string_view path);

    template<typename T, typename Node>
    T read_scalar(const Node &node, std::string_view path) {
        if(auto value = node.template value<T>(); value.has_value()) return value.value();
        throw except::runtime_error("Expected {} at [{}]", type_name<T>(), path);
    }

    template<typename Enum, typename Node>
    Enum read_enum(const Node &node, std::string_view path) {
        if(auto value = node.template value<std::string>(); value.has_value()) return sv2enum<Enum>(value.value());
        throw except::runtime_error("Expected string enum at [{}]", path);
    }

    template<typename T, typename Node>
    T read_array_like(const Node &node, std::string_view path) {
        auto *array = node.as_array();
        if(array == nullptr) throw except::runtime_error("Expected array at [{}]", path);

        if constexpr(is_std_vector_v<T>) {
            using value_type = typename T::value_type;
            T result;
            result.reserve(array->size());
            for(std::size_t idx = 0; idx < array->size(); ++idx) {
                auto item_path = fmt::format("{}[{}]", path, idx);
                result.emplace_back(read_node<value_type>((*array)[idx], item_path));
            }
            return result;
        } else {
            using value_type             = typename T::value_type;
            constexpr auto expected_size = std::tuple_size_v<T>;
            if(array->size() != expected_size) throw except::runtime_error("Expected [{}] to have {} entries, got {}", path, expected_size, array->size());
            T result{};
            for(std::size_t idx = 0; idx < expected_size; ++idx) {
                auto item_path = fmt::format("{}[{}]", path, idx);
                result[idx]    = read_node<value_type>((*array)[idx], item_path);
            }
            return result;
        }
    }

    template<typename T, typename Node>
    T read_complex(const Node &node, std::string_view path) {
        using value_type = typename T::value_type;
        if(auto *array = node.as_array()) {
            if(array->size() != 2) throw except::runtime_error("Expected complex [{}] to have 2 entries, got {}", path, array->size());
            return T{
                read_node<value_type>((*array)[0], fmt::format("{}[0]", path)),
                read_node<value_type>((*array)[1], fmt::format("{}[1]", path)),
            };
        }
        if(auto *table = node.as_table()) {
            auto real = table->at_path("re");
            auto imag = table->at_path("im");
            if(!real || !imag) throw except::runtime_error("Expected complex table [{}] to define both [re] and [im]", path);
            return T{
                read_node<value_type>(real, fmt::format("{}.re", path)),
                read_node<value_type>(imag, fmt::format("{}.im", path)),
            };
        }
        throw except::runtime_error("Expected [{}] to be a complex array or table", path);
    }

    template<typename T, typename Node>
    T read_node(const Node &node, std::string_view path) {
        if constexpr(std::is_same_v<T, std::string>) {
            return read_scalar<std::string>(node, path);
        } else if constexpr(std::is_same_v<T, bool>) {
            return read_scalar<bool>(node, path);
        } else if constexpr(std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            return read_scalar<T>(node, path);
        } else if constexpr(std::is_floating_point_v<T>) {
            return read_scalar<T>(node, path);
        } else if constexpr(std::is_enum_v<T>) {
            return read_enum<T>(node, path);
        } else if constexpr(is_std_vector_v<T> || is_std_array_v<T>) {
            return read_array_like<T>(node, path);
        } else if constexpr(is_std_complex_v<T>) {
            return read_complex<T>(node, path);
        } else {
            static_assert(!sizeof(T), "Unsupported TOML conversion type");
        }
    }

    template<typename T>
    void load_if_present(const toml::table &table, std::string_view path, T &ref) {
        if(auto node = table.at_path(path)) ref = read_node<T>(node, path);
    }

}

void settings::load(std::string_view path) {
    toml::table table;
    try {
        table = toml::parse_file(std::string(path));
    } catch(const std::exception &ex) { throw except::runtime_error("Failed to parse [{}]: {}", path, ex.what()); }

    test::tomlpp::generated::for_each_setting([&](const auto &spec) { load_if_present(table, spec.toml_path, *spec.value); });
}
