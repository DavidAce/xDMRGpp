#pragma once
#include "config/enum_utils.h"

/*! Select the bond expansion policies.
    While this is most useful for single-site DMRG, it can help in multisite DMRG as well due to
    the added noise (POSTOPT_1SITE in particular), which helps the algorithm escape local minima.

    The POSTOPT_1SITE is identical to the DMRG3S method up to the choice of mixing factor:
        - the expansion occurs after the optimization step, before moving
        - the current site is enriched as  Ψ' =  (P⁰ + P¹ + P²)Ψ , where:
              - P⁰ = α₀ = sqrt(1-α₁²-α₂²),
              - P¹ = α₁H¹Ψ, α₁ = |(H¹ - <H¹>)Ψ| (residual norm wrt H¹)
              - P² = α₂H²Ψ, α₂ = |(H² - <H²>)Ψ| (residual norm wrt H²)
              - H¹,H²,Ψ denote the local "effective" parts of the MPS/MPO corresponding to the current site.
        - the next site ahead is zero-padded to match the new dimensions.
        - this works well because the residuals vanish as we approach an eigenstate.
    For PREOPT_NSITE_REAR and PREOPT_NSITE_FORE:
        - the expansion occurs just before the main DMRG optimization step.
        - the expansion involves [active sites] plus sites behind or ahead.
        - at least two sites are used, the upper limit depends on dmrg_blocksize
        - on these sites, we find α that minimizes f(Ψ'), where Ψ' = (α₀ + α₁ H¹ + α₂ H²)Ψ, and f is
          the relevant objective function (energy, variance or <H>/<H²>).
        - note that no zero-padding is used here.
 */

enum class BondExpansionPolicy : int {
    NONE               = 0,                                      /*!< No bond expansion (strictly for multisite DMRG, but not recommended anyway) */
    DMRG3S             = 1 << 0,                                 /*!< Single-site expansion of 1 bond ahead, after optimization (DMRG3S) */
    POSTOPT_1SITE      = 1 << 1,                                 /*!< Single-site expansion of 1 bond ahead, after optimization */
    PREOPT_1SITE       = 1 << 2,                                 /*!< Single-site expansion of 1 bond ahead, before optimization */
    POSTOPT_RDMP_1SITE = 1 << 4,                                 /*!< Single-site expansion after optimization (reduced density matrix perturbation) */
    PREOPT_RDMP_1SITE  = 1 << 3,                                 /*!< Single-site expansion before optimization (reduced density matrix perturbation) */
    PREOPT_NSITE_REAR  = 1 << 5,                                 /*!< (Experimental) Multisite expansion of [active sites] plus 1 sites behind. */
    PREOPT_NSITE_FORE  = 1 << 6,                                 /*!< (Experimental) Multisite expansion of [active sites] plus 1 sites ahead. */
    H1                 = 1 << 7,                                 /*!< Enable bond expansion using H¹ */
    H2                 = 1 << 8,                                 /*!< Enable bond expansion using H² */
    DEFAULT            = PREOPT_1SITE | POSTOPT_1SITE | H1 | H2, /*!< Default bond-expansion recipe for single-site DMRG */
    allow_bitops                                                 /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view    enum2sv(BondExpansionPolicy item) noexcept;
template<> BondExpansionPolicy sv2enum<BondExpansionPolicy>(std::string_view item);
template<> std::string         flag2str(const BondExpansionPolicy &item) noexcept;
