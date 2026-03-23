#pragma once

template<typename Scalar> class TensorsFinite;
struct BondExpansionConfig;
namespace tools::finite::env::internal {

    template<typename Scalar>
    void run_expansion_term_mixer(TensorsFinite<Scalar> &tensors, long posP, long pos0, const BondExpansionConfig &bcfg);

}
