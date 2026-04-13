#pragma once
#include <cstddef>

template<typename Scalar> class TensorsFinite;
struct BondExpansionConfig;
namespace tools::finite::env::internal {

    template<typename Scalar>
    void run_expansion_term_mixer(TensorsFinite<Scalar> &tensors, size_t posP, size_t pos0, const BondExpansionConfig &bcfg);

}
