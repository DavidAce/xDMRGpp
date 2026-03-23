#pragma once

#include <array>
#include <cctype>
#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace test::enum_support {

template<typename E>
struct enum_entry {
    E                value;
    std::string_view name;
    std::string_view doc;
    bool             canonical = true;
};

template<typename E>
struct enum_traits;

template<typename E>
concept reflected_enum = std::is_enum_v<E> && requires {
    enum_traits<E>::entries;
    enum_traits<E>::doc;
    enum_traits<E>::is_bitflag;
};

template<reflected_enum E>
constexpr auto entries() noexcept -> std::span<const enum_entry<E>> {
    return enum_traits<E>::entries;
}

template<reflected_enum E>
inline constexpr bool is_bitflag_enum_v = enum_traits<E>::is_bitflag;

constexpr bool is_space(char ch) noexcept {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
}

constexpr std::string_view trim(std::string_view text) noexcept {
    while(!text.empty() && is_space(text.front())) text.remove_prefix(1);
    while(!text.empty() && is_space(text.back())) text.remove_suffix(1);
    return text;
}

template<reflected_enum E>
constexpr std::string_view enum_doc() noexcept {
    return enum_traits<E>::doc;
}

template<reflected_enum E>
constexpr std::string_view to_string(E value) noexcept {
    for(const auto &entry : entries<E>()) {
        if(entry.value == value) return entry.name;
    }
    return {};
}

template<reflected_enum E>
constexpr std::string_view doc_of(E value) noexcept {
    for(const auto &entry : entries<E>()) {
        if(entry.value == value) return entry.doc;
    }
    return {};
}

template<reflected_enum E>
constexpr auto from_string(std::string_view name) noexcept -> std::optional<E> {
    for(const auto &entry : entries<E>()) {
        if(entry.name == name) return entry.value;
    }
    return std::nullopt;
}

template<reflected_enum E>
std::vector<std::string_view> names(bool canonical_only = false) {
    std::vector<std::string_view> result;
    result.reserve(entries<E>().size());
    for(const auto &entry : entries<E>()) {
        if(canonical_only && !entry.canonical) continue;
        result.emplace_back(entry.name);
    }
    return result;
}

template<typename E>
requires is_bitflag_enum_v<E>
constexpr E operator|(E lhs, E rhs) noexcept {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template<typename E>
requires is_bitflag_enum_v<E>
constexpr E operator&(E lhs, E rhs) noexcept {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template<typename E>
requires is_bitflag_enum_v<E>
constexpr E &operator|=(E &lhs, E rhs) noexcept {
    lhs = lhs | rhs;
    return lhs;
}

template<typename E>
requires is_bitflag_enum_v<E>
constexpr bool has_flag(E value, E flag) noexcept {
    using U = std::underlying_type_t<E>;
    const auto bits = static_cast<U>(flag);
    if(bits == 0) return static_cast<U>(value) == 0;
    return (static_cast<U>(value) & bits) == bits;
}

inline auto split_tokens(std::string_view text, char separator = '|') -> std::vector<std::string_view> {
    text = trim(text);
    if(text.empty()) return {};

    std::vector<std::string_view> tokens;
    std::size_t                   begin = 0;
    while(begin <= text.size()) {
        auto end = text.find(separator, begin);
        if(end == std::string_view::npos) end = text.size();
        tokens.emplace_back(trim(text.substr(begin, end - begin)));
        if(end == text.size()) break;
        begin = end + 1;
    }
    return tokens;
}

template<reflected_enum E>
requires is_bitflag_enum_v<E>
auto parse_flags(std::string_view text) -> std::optional<E> {
    auto tokens = split_tokens(text);
    if(tokens.empty()) return std::nullopt;

    auto value = static_cast<E>(0);
    for(auto token : tokens) {
        if(token.empty()) return std::nullopt;
        auto parsed = from_string<E>(token);
        if(!parsed.has_value()) return std::nullopt;
        value |= parsed.value();
    }
    return value;
}

template<reflected_enum E>
requires is_bitflag_enum_v<E>
auto format_flags(E value) -> std::string {
    using U = std::underlying_type_t<E>;

    if(static_cast<U>(value) == 0) {
        if(auto exact = to_string(value); !exact.empty()) return std::string(exact);
        return "0";
    }

    std::string out;
    auto        remaining = static_cast<U>(value);
    for(const auto &entry : entries<E>()) {
        const auto bits = static_cast<U>(entry.value);
        if(!entry.canonical || bits == 0) continue;
        if((remaining & bits) == bits) {
            if(!out.empty()) out += '|';
            out += entry.name;
            remaining &= ~bits;
        }
    }

    if(out.empty()) {
        if(auto exact = to_string(value); !exact.empty()) return std::string(exact);
        using UU = std::make_unsigned_t<U>;
        return std::to_string(static_cast<unsigned long long>(static_cast<UU>(static_cast<U>(value))));
    }

    if(remaining != 0) {
        out += "|UNKNOWN(";
        using UU = std::make_unsigned_t<U>;
        out += std::to_string(static_cast<unsigned long long>(static_cast<UU>(remaining)));
        out += ')';
    }

    return out;
}

} // namespace test::enum_support
