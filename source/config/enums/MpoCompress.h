#pragma once
#include "config/enum_utils.h"

/*! Compression strategy applied when materializing MPO tensors */
enum class MpoCompress {
    NONE, /*!< Do not compress */
    SVD,  /*!< Use SVD on each mpo */
    DPL,  /*!< Deparallelization: removes parallel columns/rows from each mpo */
    AUTO  /*!< Select based on global setting */
};
template<> std::string_view enum2sv(MpoCompress item) noexcept;
template<> MpoCompress      sv2enum<MpoCompress>(std::string_view item);
