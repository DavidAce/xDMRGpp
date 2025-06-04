#pragma once
#include "config/enums.h"
#include <cstddef>
enum class BondExpansionOrder { PREOPT, POSTOPT };
struct BondExpansionConfig {
    BondExpansionOrder  order;
    BondExpansionPolicy policy;
    size_t              maxiter   = 1ul;
    size_t              blocksize = 1ul;
    size_t              nkrylov   = 10ul;
    float               factor    = 1.0f;
    long                bondlim   = -1l;
    double              trnclim   = 0;
    float               minalpha  = 1e-15f;
    float               maxalpha  = 1e-3f;
    OptAlgo             optAlgo;
    OptRitz             optRitz;
};