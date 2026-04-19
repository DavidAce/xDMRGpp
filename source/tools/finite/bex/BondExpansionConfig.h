#pragma once
#include <cstddef>
enum class BondExpansionOrder { PREOPT, POSTOPT };
enum class BondExpansionPolicy;
enum class OptAlgo;
enum class OptRitz;
enum class OptType;
struct BondExpansionConfig {
    BondExpansionOrder  order;
    BondExpansionPolicy policy;
    size_t              maxiter       = 1ul;
    size_t              blocksize     = 1ul;
    size_t              nkrylov       = 10ul;
    float               bond_factor   = 1.0f;
    long                bond_lim      = -1l;
    double              trnc_lim      = 0;
    double              mixing_factor = 1e-3;
    OptAlgo             optAlgo;
    OptRitz             optRitz;
    OptType             optType;
};
