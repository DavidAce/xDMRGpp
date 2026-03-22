#pragma once
#include "config/enum_utils.h"

/*! Averaging rule used when combining disorder realizations or correlations */
enum class MeanType {
    ARITHMETIC, /*!< Use the arithmetic mean */
    GEOMETRIC   /*!< Use the geometric mean */
};
template<> std::string_view enum2sv(MeanType item) noexcept;
template<> MeanType         sv2enum<MeanType>(std::string_view item);
