#pragma once
#include "config/enum_utils.h"

/*! Bitflags that decide when a local optimization should be attempted again */
enum class OptWhen : int {
    NEVER              = 0,  /*!< Never retry based on the previous exit status */
    PREV_FAIL_GRADIENT = 1,  /*!< Retry if the previous optimization failed the gradient test */
    PREV_FAIL_RESIDUAL = 2,  /*!< Retry if the previous optimization failed the residual test */
    PREV_FAIL_OVERLAP  = 4,  /*!< Retry if the previous optimization failed the overlap test */
    PREV_FAIL_NOCHANGE = 8,  /*!< Retry if the previous optimization made no meaningful change */
    PREV_FAIL_WORSENED = 16, /*!< Retry if the previous optimization worsened the objective */
    PREV_FAIL_ERROR    = 32, /*!< Retry if the previous optimization ended with an error */
    ALWAYS             = 64, /*!< Always retry regardless of the previous exit status */
    allow_bitops             /*!< Internal sentinel that marks this enum as a bitflag */
};
template<> std::string_view enum2sv(OptWhen item) noexcept;
template<> OptWhen          sv2enum<OptWhen>(std::string_view item);
template<> std::string      flag2str(const OptWhen &item) noexcept;
