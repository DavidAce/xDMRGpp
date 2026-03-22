#pragma once
#include "config/enum_utils.h"

/*! Reasons why an algorithm run stopped */
enum class AlgorithmStop : int {
    SUCCESS,   /*!< The target condition was reached successfully */
    SATURATED, /*!< The algorithm stopped after its convergence metrics saturated */
    MAX_ITERS, /*!< The maximum iteration count was reached */
    NONE       /*!< No stop reason has been recorded */
};

template<> std::string_view enum2sv(AlgorithmStop item) noexcept;
template<> AlgorithmStop    sv2enum<AlgorithmStop>(std::string_view item);
