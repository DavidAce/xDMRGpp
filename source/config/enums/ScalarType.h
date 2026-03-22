#pragma once

#include "config/enum_utils.h"

/*! Sets the storage type for the state and model */

enum class ScalarType {
    FP32,  /*!< Single-precision real scalar type */
    FP64,  /*!< Double-precision real scalar type */
    FP128, /*!< Quadruple-precision real scalar type */
    CX32,  /*!< Single-precision complex scalar type */
    CX64,  /*!< Double-precision complex scalar type */
    CX128  /*!< Quadruple-precision complex scalar type */
};

template<> std::string_view enum2sv(ScalarType item) noexcept;
template<> ScalarType       sv2enum<ScalarType>(std::string_view item);
