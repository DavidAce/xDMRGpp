#pragma once

#include "config/enum_utils.h"

/*! Which previous stop conditions qualify a stored state for resume */
enum class ResumePolicy {
    IF_MAX_ITERS,    /*!< Resume only states that stopped at the iteration limit */
    IF_SATURATED,    /*!< Resume only states that stopped due to saturation */
    IF_UNSUCCESSFUL, /*!< Resume only states that did not stop successfully */
    IF_SUCCESSFUL,   /*!< Resume only states that stopped successfully */
    ALWAYS           /*!< Resume any available state */
};

template<> std::string_view enum2sv(ResumePolicy item) noexcept;
template<> ResumePolicy     sv2enum<ResumePolicy>(std::string_view item);
