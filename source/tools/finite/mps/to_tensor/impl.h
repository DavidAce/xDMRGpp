#include "math/tenx.h"
// -- (textra first)
#include "../../mps.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"

using tools::finite::mps::RealScalar;

template<typename CalcType, typename Scalar>
Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor(const std::vector<std::unique_ptr<MpsSite<Scalar>>> &mps_sites, std::string_view name) {
    if constexpr(sfinae::is_std_complex_v<CalcType>) {
        bool all_real = std::all_of(mps_sites.begin(), mps_sites.end(), [](const auto &mps) -> bool { return mps->is_real(); });
        if(all_real) return mps2tensor<RealScalar<Scalar>>(mps_sites, name);
    }
    using R       = RealScalar<CalcType>;
    auto spindims = std::vector<long>();
    auto bonddims = std::vector<long>();
    for(auto &mps : mps_sites) {
        spindims.emplace_back(mps->spin_dim());
        bonddims.emplace_back(num::prod(spindims) * mps->get_chiR());
    }
    long  memsize = bonddims.empty() ? 0 : *std::max_element(bonddims.begin(), bonddims.end());
    auto  statev  = Eigen::Tensor<CalcType, 1>(memsize);
    auto  off1    = std::array<long, 1>{0};
    auto  ext1    = std::array<long, 1>{1};
    auto  ext2    = std::array<long, 2>{1, 1};
    auto &threads = tenx::threads::get();
    statev.slice(off1, ext1).setConstant(R{1});
    // For each site that we contract, the state vector grows by mps->spin_dim()
    // If 4 spin1/2 have been contracted, the state vector could have size 16x7 if the last chi was 7.
    // Then contracting the next site, with dimensions 2x7x9 will get you a 16x2x9 tensor.
    // Lastly, one should reshape it back into a 32 x 9 state vector

    for(auto &mps : mps_sites) {
        auto temp = Eigen::Tensor<CalcType, 2>(statev.slice(off1, ext1).reshape(ext2)); // Make a temporary copy of the state vector
        ext1      = {mps->spin_dim() * temp.dimension(0) * mps->get_chiR()};
        ext2      = {mps->spin_dim() * temp.dimension(0), mps->get_chiR()};
        if(ext1[0] > memsize) throw except::logic_error("mps2tensor [{}]: size of ext1[0] > memsize", name);
        statev.slice(off1, ext1).device(*threads->dev) = temp.contract(mps->template get_M_as<CalcType>(), tenx::idx({1}, {1})).reshape(ext1);
    }
    // Finally, we view a slice of known size 2^L
    statev    = statev.slice(off1, ext1);
    R norm    = tenx::norm(statev);
    R normTol = std::numeric_limits<R>::epsilon() * settings::precision::max_norm_slack;
    R normErr = std::abs(norm - R{1});
    if(normErr > normTol) { tools::log->warn("mps2tensor [{}]: Norm far from unity: {:.5e}", name, fp(normErr)); }
    return tenx::asScalarType<Scalar>(statev);
}

template<typename CalcType, typename Scalar>
Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor(const StateFinite<Scalar> &state) {
    return mps2tensor<CalcType>(state.mps_sites, state.get_name());
}
