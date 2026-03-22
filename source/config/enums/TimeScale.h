#pragma once

#include "config/enum_utils.h"

/*! Spacing used for target time points in time evolution */
enum class TimeScale {
    LINSPACED, /*!< Linearly spaced time points */
    LOGSPACED  /*!< Logarithmically spaced time points */
};

template<> std::string_view enum2sv(TimeScale item) noexcept;
template<> TimeScale        sv2enum<TimeScale>(std::string_view item);
