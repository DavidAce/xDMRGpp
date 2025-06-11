#pragma once
#include "config/enums.h"
#include <cstddef>
enum class BondExpansionOrder { PREOPT, POSTOPT };
struct BondExpansionConfig {
    BondExpansionOrder  order;
    BondExpansionPolicy policy;
    size_t              maxiter       = 1ul;
    size_t              blocksize     = 1ul;
    size_t              nkrylov       = 10ul;
    float               bond_factor   = 1.0f;
    long                bond_limit    = -1l;
    double              trnc_limit    = 0;
    double              mixing_factor = 1e-3;
    OptAlgo             optAlgo;
    OptRitz             optRitz;
};