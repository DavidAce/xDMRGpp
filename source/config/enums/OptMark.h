#pragma once
#include "config/enum_utils.h"

/*! Binary pass or fail mark for optimization diagnostics */
enum class OptMark {
    PASS, /*!< The check passed */
    FAIL  /*!< The check failed */
};
template<> std::string_view enum2sv(OptMark item) noexcept;
template<> OptMark          sv2enum<OptMark>(std::string_view item);
