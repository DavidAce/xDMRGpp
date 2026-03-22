#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

template<typename T>
concept enum_is_bitflag_v = requires(T value) {
    { T::allow_bitops };
};

template<typename E>
requires enum_is_bitflag_v<E>
constexpr auto operator|(E lhs, E rhs) noexcept -> decltype(E::allow_bitops) {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template<typename E>
requires enum_is_bitflag_v<E>
constexpr auto operator&(E lhs, E rhs) noexcept -> decltype(E::allow_bitops) {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template<typename E>
requires enum_is_bitflag_v<E>
constexpr auto operator|=(E &lhs, E rhs) noexcept -> decltype(E::allow_bitops) {
    lhs = lhs | rhs;
    return lhs;
}

template<typename E>
inline bool has_flag(E target, E check) noexcept {
    using U = std::underlying_type_t<E>;
    return (static_cast<U>(target) & static_cast<U>(check)) == static_cast<U>(check);
}

template<typename E>
inline bool has_flag(std::optional<E> target, E check) noexcept {
    if(!target.has_value()) return false;
    return has_flag(target.value(), check);
}

template<typename E, typename... Args>
requires std::conjunction_v<std::is_same<E, Args>...>
bool has_any_flags(E target, Args &&...check) {
    using U = std::underlying_type_t<E>;
    return (((static_cast<U>(target) & static_cast<U>(check)) == static_cast<U>(check)) || ...);
}

template<typename E, typename... Args>
requires std::conjunction_v<std::is_same<E, Args>...>
bool has_none_of_flags(E target, Args &&...check) {
    using U = std::underlying_type_t<E>;
    return !(((static_cast<U>(target) & static_cast<U>(check)) == static_cast<U>(check)) || ...);
}

template<typename E, typename... Args>
requires std::conjunction_v<std::is_same<E, Args>...>
bool has_all_flags(E target, Args &&...check) {
    using U = std::underlying_type_t<E>;
    return (((static_cast<U>(target) & static_cast<U>(check)) == static_cast<U>(check)) && ...);
}

template<typename E1, typename E2>
inline bool have_common(E1 lhs, E2 rhs) noexcept {
    using U1 = std::underlying_type_t<E1>;
    using U2 = std::underlying_type_t<E2>;
    static_assert(std::is_same_v<U1, U2>);
    return (static_cast<U1>(lhs) & static_cast<U2>(rhs)) != 0;
}

template<typename T>
std::string_view enum2sv(T item) noexcept = delete;

template<typename T>
T sv2enum(std::string_view item) = delete;

template<typename T>
std::string flag2str(const T &item) noexcept = delete;

template<typename T>
std::vector<std::string_view> enum2sv(const std::vector<T> &items) noexcept {
    auto result = std::vector<std::string_view>();
    result.reserve(items.size());
    for(const auto &item : items) result.emplace_back(enum2sv(item));
    return result;
}

template<typename T, auto num>
using enumarray_t = std::array<std::pair<std::string, T>, num>;

template<typename T, typename... Args>
auto mapStr2Enum(Args... names) {
    constexpr auto num     = sizeof...(names);
    auto           pairgen = [](const std::string &name) -> std::pair<std::string, T> { return {name, sv2enum<T>(name)}; };
    return enumarray_t<T, num>{pairgen(names)...};
}

template<typename T, typename... Args>
auto mapEnum2Str(Args... enums) {
    constexpr auto num     = sizeof...(enums);
    auto           pairgen = [](const T &e) -> std::pair<std::string, T> { return {std::string(enum2sv(e)), e}; };
    return enumarray_t<T, num>{pairgen(enums)...};
}
