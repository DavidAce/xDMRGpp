#pragma once
#include "config/enum_utils.h"

/*! Unary operation applied to an entire gate circuit */
enum class CircuitOp {
    NONE, /*!< Apply the circuit as stored */
    ADJ,  /*!< Apply the adjoint of the circuit */
    TRN   /*!< Apply the transpose of the circuit */
};

template<> std::string_view enum2sv(CircuitOp item) noexcept;
template<> CircuitOp        sv2enum<CircuitOp>(std::string_view item);
