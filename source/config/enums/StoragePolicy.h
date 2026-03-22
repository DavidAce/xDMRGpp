#pragma once

#include "config/enum_utils.h"

/*! Determines in which cases we calculate and store to file
 *  Note that many flags can be set simultaneously
 */

enum class StoragePolicy : int {
    NONE    = 0,    /*!< Never store */
    INIT    = 1,    /*!< Store only once during initialization e.g. model (usually in preprocessing) */
    ITER    = 2,    /*!< Store after every iteration */
    EMIN    = 4,    /*!< Store after finding the ground state (e.g. before xDMRG) */
    EMAX    = 8,    /*!< Store after finding the highest energy eigenstate (e.g. before xDMRG) */
    PROJ    = 16,   /*!< Store after projections */
    BOND    = 32,   /*!< Store after bond updates */
    TRNC    = 64,   /*!< Store after truncation error limit updates */
    FAILURE = 128,  /*!< Store only if the simulation did not succeed (usually for debugging) */
    SUCCESS = 256,  /*!< Store only if the simulation succeeded */
    FINISH  = 512,  /*!< Store when the simulation has finished (regardless of failure or success) */
    ALWAYS  = 1024, /*!< Store every chance you get */
    REPLACE = 2048, /*!< Keep only the last event (i.e. replace previous events when possible) */
    RBDS    = 4096, /*!< Store rbds steps */
    RTES    = 8192, /*!< Store rtes steps */
    allow_bitops    /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view enum2sv(StoragePolicy item) noexcept;
template<> StoragePolicy    sv2enum<StoragePolicy>(std::string_view item);
template<> std::string      flag2str(const StoragePolicy &item) noexcept;
