#pragma once
#include "config/enum_utils.h"

/*! \brief The type of Hermitian matrix M_i used in the 2-site gates u_i of the random unitary circuit for our l-bit model: u_i = exp(-i f w_i M_i)
 */

enum class LbitCircuitGateMatrixKind : int {
    MATRIX_V1, /*!< Below, θᵢ, RE(c), and IM(c) are gaussian N(0,1) and λ is a constant \verbatim M_i = θ₀/4 (1 + σz[i] + σz[i+1] + λ σz[i] σz[i+1]) + θ₁/4 (1 +
                  σz[i] - σz[i+1] - λ σz[i] σz[i+1]) + θ₂/4 (1 - σz[i] + σz[i+1] - λ σz[i] σz[i+1]) + θ₃/4 (1 - σz[i] - σz[i+1] + λ σz[i] σz[i+1]) + c
                  σ+[i]σ-[i+1] + c^* σ-[i]σ+[i+1] \endverbatim */
    MATRIX_V2, /*!< Below, θᵢ, RE(c), and IM(c) are gaussian N(0,1) and λ is a constant \verbatim M_i = θ₀/2 σz[i] + θ₁/2 σz[i+1] + θ₂/2 λ σz[i]σz[i+1] + c
                  σ+[i]σ-[i+1] + c^* σ-[i]σ+[i+1] \endverbatim */
    MATRIX_V3  /*!< Below, θᵢ, RE(c), and IM(c) are gaussian N(0,1) and λ is a constant \verbatim M_i = θ₀/2 σz[i] + θ₁/2 σz[i+1] + λ σz[i]σz[i+1] + c
                  σ+[i]σ-[i+1] + c^* σ-[i]σ+[i+1] \endverbatim */
};
template<> std::string_view          enum2sv(LbitCircuitGateMatrixKind item) noexcept;
template<> LbitCircuitGateMatrixKind sv2enum<LbitCircuitGateMatrixKind>(std::string_view item);
