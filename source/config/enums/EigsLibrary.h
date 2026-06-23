#pragma once
#include "config/enum_utils.h"

/*! Backend library used for iterative few-eigenpair solvers */
enum class EigsLibrary {
    ARPACK,  /*!< Use ARPACK */
    SPECTRA, /*!< Use Spectra */
    PRIMME,  /*!< Use PRIMME */
    EIGSMPO, /*!< Use the internal MPO eigensolver */
    GRIT     /*!< Use GRIT */
};
template<> std::string_view enum2sv(EigsLibrary item) noexcept;
template<> EigsLibrary      sv2enum<EigsLibrary>(std::string_view item);
