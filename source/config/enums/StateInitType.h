#pragma once

#include "config/enum_utils.h"

/*! Amplitude type used when preparing the initial state */
enum class StateInitType {
    REAL, /*!< Use real amplitudes */
    CPLX  /*!< Use complex amplitudes */
};

template<> std::string_view enum2sv(StateInitType item) noexcept;
template<> StateInitType    sv2enum<StateInitType>(std::string_view item);
