#pragma once

#include "config/enum_utils.h"

/*! The reason that we are invoking a storage call */

enum class StorageEvent : int {
    NONE        = 0,    /*!< No event */
    INIT        = 1,    /*!< the initial state was defined */
    MODEL       = 2,    /*!< the model was defined */
    EMIN        = 4,    /*!< the ground state was found (e.g. before xDMRG) */
    EMAX        = 8,    /*!< the highest energy eigenstate was found (e.g. before xDMRG) */
    PROJECTION  = 16,   /*!< a projection to a spin parity sector was made */
    BOND_UPDATE = 32,   /*!< the bond dimension limit was updated */
    TRNC_UPDATE = 64,   /*!< the truncation error threshold for SVD was updated */
    RBDS_STEP   = 128,  /*!< reverse bond dimension scaling step was made */
    RTES_STEP   = 256,  /*!< reverse truncation error scaling step was made */
    ITERATION   = 512,  /*!< an iteration finished */
    FINISHED    = 1024, /*!< a simulation has finished */
    allow_bitops        /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view enum2sv(StorageEvent item) noexcept;
template<> StorageEvent     sv2enum<StorageEvent>(std::string_view item);
template<> std::string      flag2str(const StorageEvent &item) noexcept;
