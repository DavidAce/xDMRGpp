#pragma once

#include "config/enum_utils.h"

/*! Used to calculate the information lattice */

enum class Precision {
    SINGLE,   /*!< Single-precision floating point */
    DOUBLE,   /*!< Double-precision floating point */
    QUADRUPLE /*!< Quadruple-precision floating point */
};

template<> std::string_view enum2sv(Precision item) noexcept;
template<> Precision        sv2enum<Precision>(std::string_view item);
