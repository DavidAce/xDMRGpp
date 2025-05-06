#pragma once
#include <stdexcept>
#include <string_view>

namespace tid {
    enum level : int {
        disabled = -2,
        parent   = -1,
        normal   = 0,
        higher   = 1,
        highest  = 2,
    };

    constexpr std::string_view level2sv(level l) noexcept {
        switch(l) {
            case disabled: return "disabled";
            case parent: return "parent";
            case normal: return "normal";
            case higher: return "higher";
            case highest: return "highest";
            default: return "unknown tid::level";
        }
    }
    constexpr level sv2level(std::string_view l) {
        if(l == "disabled") return tid::level::disabled;
        if(l == "parent") return tid::level::parent;
        if(l == "normal") return tid::level::normal;
        if(l == "higher") return tid::level::higher;
        if(l == "highest") return tid::level::highest;
        throw std::runtime_error("Given item is not a tid::level enum");
    }
}

template<typename T>
constexpr std::string_view enum2sv(const T item) noexcept;

template<typename T>
constexpr auto sv2enum(std::string_view item);

template<>
constexpr std::string_view enum2sv(const tid::level l) noexcept {
    return tid::level2sv(l);
}

template<>
constexpr auto sv2enum<tid::level>(std::string_view l) {
    return tid::sv2level(l);
}
