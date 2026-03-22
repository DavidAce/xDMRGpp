#pragma once
#include "config/enum_utils.h"

/*! Unary operation applied to an individual gate tensor */
enum class GateOp {
    NONE, /*!< Apply the gate as stored */
    CNJ,  /*!< Apply the elementwise conjugate of the gate */
    ADJ,  /*!< Apply the adjoint of the gate */
    TRN   /*!< Apply the transpose of the gate */
};

template<> std::string_view enum2sv(GateOp item) noexcept;
template<> GateOp           sv2enum<GateOp>(std::string_view item);
