#pragma once

#include "config/enum_utils.h"

/*! Reason for reinitializing or randomizing the state */
enum class ResetReason {
    INIT,        /*!< Initial state construction at the start of a run */
    FIND_WINDOW, /*!< Reset while searching for an energy window */
    SATURATED,   /*!< Reset after the algorithm saturated */
    NEW_STATE,   /*!< Reset to start from a newly selected state */
    BOND_UPDATE  /*!< Reset after increasing the bond-dimension budget */
};

template<> std::string_view enum2sv(ResetReason item) noexcept;
template<> ResetReason      sv2enum<ResetReason>(std::string_view item);
