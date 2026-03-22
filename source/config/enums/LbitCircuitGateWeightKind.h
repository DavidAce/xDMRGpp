#pragma once
#include "config/enum_utils.h"

/*! The type of weights w_i used in each 2-site gate u_i of the random unitary circuit for our l-bit model: u_i = exp(-i f w_i M_i) */

enum class LbitCircuitGateWeightKind : int {
    IDENTITY, /*!< w_i = 1 (i.e. disables weights) */
    EXPDECAY  /*!< w_i = exp(-2|h[i] - h[i+1]|), where h[i] are on-site fields of the l-bit Hamiltonian */
};
template<> std::string_view          enum2sv(LbitCircuitGateWeightKind item) noexcept;
template<> LbitCircuitGateWeightKind sv2enum<LbitCircuitGateWeightKind>(std::string_view item);
