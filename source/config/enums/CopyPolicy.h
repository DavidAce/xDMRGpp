#pragma once
#include "config/enum_utils.h"

/*! Policy for copying output from a temporary HDF5 file to its final destination */
enum class CopyPolicy {
    FORCE, /*!< Always copy from the temporary file */
    TRY,   /*!< Copy opportunistically, skipping duplicate or unscheduled copies */
    OFF    /*!< Do not copy from the temporary file */
};

template<> std::string_view enum2sv(CopyPolicy item) noexcept;
template<> CopyPolicy       sv2enum<CopyPolicy>(std::string_view item);
