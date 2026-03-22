#pragma once
#include "config/enum_utils.h"

/*! Bitflags describing the outcome of a local optimization step */
enum class OptExit : int {
    SUCCESS       = 0,  /*!< The optimization succeeded */
    FAIL_GRADIENT = 1,  /*!< The gradient-based convergence test failed */
    FAIL_RESIDUAL = 2,  /*!< The residual-based convergence test failed */
    FAIL_OVERLAP  = 4,  /*!< The overlap with the target state was too small */
    FAIL_NOCHANGE = 8,  /*!< The update changed the state too little */
    FAIL_WORSENED = 16, /*!< The update worsened the objective */
    FAIL_ERROR    = 32, /*!< The optimizer aborted due to an error */
    NONE          = 64, /*!< No optimization exit status has been recorded yet */
    allow_bitops        /*!< Internal sentinel that marks this enum as a bitflag */
};
template<> std::string_view enum2sv(OptExit item) noexcept;
template<> OptExit          sv2enum<OptExit>(std::string_view item);
template<> std::string      flag2str(const OptExit &item) noexcept;
