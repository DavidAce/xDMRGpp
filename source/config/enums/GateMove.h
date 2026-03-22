#pragma once
#include "config/enum_utils.h"

/*! Whether applying gates may move the MPS center position */
enum class GateMove {
    OFF, /*!< Keep the current center position fixed */
    ON,  /*!< Move the center position to follow the applied gate */
    AUTO /*!< Choose automatically based on the gate sequence */
};

template<> std::string_view enum2sv(GateMove item) noexcept;
template<> GateMove         sv2enum<GateMove>(std::string_view item);
