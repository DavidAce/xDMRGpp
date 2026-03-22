#pragma once
#include "config/enum_utils.h"

/*! Whether returned MPO tensors include boundary edge tensors */
enum class MposWithEdges {
    OFF, /*!< Return MPO tensors without boundary edges */
    ON   /*!< Return MPO tensors with boundary edges */
};
template<> std::string_view enum2sv(MposWithEdges item) noexcept;
template<> MposWithEdges    sv2enum<MposWithEdges>(std::string_view item);
