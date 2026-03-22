#pragma once
#include "config/enum_utils.h"

/*! Choose the target eigenpair */

enum class OptRitz {
    NONE, /*!< No eigenpair is targeted (e.g. time evolution) */
    LR,   /*!< Largest Real eigenvalue */
    LM,   /*!< Largest Absolute eigenvalue */
    SR,   /*!< Smallest Real eigenvalue */
    SM,   /*!< Smallest magnitude eigenvalue. MPO² Energy shift == 0. Use this to find an eigenstate with energy closest to 0) */
    IS,   /*!< Initial State energy. Energy shift == Initial state energy. Targets an eigenstate with energy near that of the initial state */
    TE    /*!< Target Energy density in normalized units [0,1]. Energy shift == settings::xdmrg::energy_density_target * (EMIN+EMAX) + EMIN. */
};
template<> std::string_view enum2sv(OptRitz item) noexcept;
template<> OptRitz          sv2enum<OptRitz>(std::string_view item);
