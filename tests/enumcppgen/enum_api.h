#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

namespace test::enumcppgen_demo {

template<typename E>
struct enable_bitops : std::false_type {};

template<typename E>
inline constexpr bool enable_bitops_v = enable_bitops<E>::value;

template<typename E>
requires enable_bitops_v<E>
constexpr E operator|(E lhs, E rhs) noexcept {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template<typename E>
requires enable_bitops_v<E>
constexpr E &operator|=(E &lhs, E rhs) noexcept {
    lhs = lhs | rhs;
    return lhs;
}

template<typename E>
requires enable_bitops_v<E>
constexpr bool has_flag(E value, E flag) noexcept {
    using U = std::underlying_type_t<E>;
    const auto bits = static_cast<U>(flag);
    if(bits == 0) return static_cast<U>(value) == 0;
    return (static_cast<U>(value) & bits) == bits;
}

template<typename E>
std::string_view enum2sv(E value) noexcept = delete;

template<typename E>
E sv2enum(std::string_view text) = delete;

template<typename E>
std::string flag2sv(E value) = delete;

}
