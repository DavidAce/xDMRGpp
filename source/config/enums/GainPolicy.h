#pragma once
#include "config/enum_utils.h"

/*! \brief Policy that determines when a quantity should increase or improve
 *  For example, when the number of eigs iterations should increase, or when
 *  the mps block size should increase, or jacobi preconditioner block size
 */

enum class GainPolicy : int {
    NEVER     = 0,  /*!< Never increase eigs iterations */
    HALFSWEEP = 1,  /*!< every halfsweep */
    FULLSWEEP = 2,  /*!< every full sweep */
    SAT_EVAR  = 4,  /*!< when the energy variance has saturated */
    SAT_ALGO  = 8,  /*!< when the algorithm has saturated */
    STK_ALGO  = 16, /*!< when the algorithm is stuck */
    FIN_BOND  = 32, /*!< only when the bond dimension has reached its maximum */
    FIN_TRNC  = 64, /*!< only when the truncation error has reached its minimum */
    allow_bitops    /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view enum2sv(GainPolicy item) noexcept;
template<> GainPolicy       sv2enum<GainPolicy>(std::string_view item);
template<> std::string      flag2str(const GainPolicy &item) noexcept;
