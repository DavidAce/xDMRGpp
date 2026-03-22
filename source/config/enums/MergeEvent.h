#pragma once
#include "config/enum_utils.h"

/*! \brief Why an mps merge event is invoked
 * Currently, this is used to check if an optimized multisite_mps was passed, in which case we need to store the truncation errors (otherwise they are
 * discarded)
 */

enum class MergeEvent {
    MOVE, /*!< Moved the center position */
    NORM, /*!< Normalized the state */
    SWAP, /*!< Swapped sites (e.g. in fLBIT) */
    GATE, /*!< Applied a gate, e.g. during time evolution */
    OPT,  /*!< Optimized some sites, e.g. during the main DMRG update */
    EXP   /*!< Subspace expansion (aka perturbation, noise, enrichment) to increase the bond dimension */
};
template<> std::string_view enum2sv(MergeEvent item) noexcept;
template<> MergeEvent       sv2enum<MergeEvent>(std::string_view item);
