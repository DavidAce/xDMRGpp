#pragma once
#include "config/enum_utils.h"

/*! Arithmetic type used inside the local optimizer */
enum class OptType {
    FP32,  /*!< Single-precision real arithmetic */
    FP64,  /*!< Double-precision real arithmetic */
    FP128, /*!< Quadruple-precision real arithmetic */
    CX32,  /*!< Single-precision complex arithmetic */
    CX64,  /*!< Double-precision complex arithmetic */
    CX128  /*!< Quadruple-precision complex arithmetic */
};
template<> std::string_view enum2sv(OptType item) noexcept;
template<> OptType          sv2enum<OptType>(std::string_view item);
