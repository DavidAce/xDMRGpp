#pragma once
#include "config/enum_utils.h"

/*! Linear-algebra solver used for the local optimization */
enum class OptSolver {
    EIG, /*!< Use an exact solver */
    EIGS /*!< Use an iterative solver (e.g. PRIMME or Arpack) */
};
template<> std::string_view enum2sv(OptSolver item) noexcept;
template<> OptSolver        sv2enum<OptSolver>(std::string_view item);
