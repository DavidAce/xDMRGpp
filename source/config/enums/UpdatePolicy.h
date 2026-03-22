#pragma once

#include "config/enum_utils.h"

/*! Bitflags describing when adaptive limits should be updated */
enum class UpdatePolicy {
    NEVER     = 0,   /*!< Never update */
    WARMUP    = 1,   /*!< Update during warmup */
    HALFSWEEP = 2,   /*!< Update every iteration */
    FULLSWEEP = 4,   /*!< Update every second iteration (left to right + right to left sweep) */
    TRUNCATED = 8,   /*!< Update whenever the state is truncated */
    SAT_EVAR  = 16,  /*!< Update when the energy variance has saturated */
    SAT_ALGO  = 32,  /*!< Update when the algorithm is saturated */
    STK_ALGO  = 64,  /*!< Update when the algorithm is stuck */
    DYNAMIC   = 128, /*!< Increase or decrease based on the energy-variance progress rate */
    allow_bitops     /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view enum2sv(UpdatePolicy item) noexcept;
template<> UpdatePolicy     sv2enum<UpdatePolicy>(std::string_view item);
template<> std::string      flag2str(const UpdatePolicy &item) noexcept;
