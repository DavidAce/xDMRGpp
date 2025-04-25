
#include "../measure.h"
#include "config/settings.h"
#include "debug/info.h"
#include "io/fmt_custom.h"
#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/common/split.h"
#include "tools/finite/mps.h"

using tools::finite::measure::RealScalar;

template<typename Scalar>
size_t tools::finite::measure::length(const TensorsFinite<Scalar> &tensors) {
    return tensors.get_length();
}
template size_t tools::finite::measure::length(const TensorsFinite<fp32> &tensors);
template size_t tools::finite::measure::length(const TensorsFinite<fp64> &tensors);
template size_t tools::finite::measure::length(const TensorsFinite<fp128> &tensors);
template size_t tools::finite::measure::length(const TensorsFinite<cx32> &tensors);
template size_t tools::finite::measure::length(const TensorsFinite<cx64> &tensors);
template size_t tools::finite::measure::length(const TensorsFinite<cx128> &tensors);

template<typename Scalar>
size_t tools::finite::measure::length(const StateFinite<Scalar> &state) {
    return state.get_length();
}
template size_t tools::finite::measure::length(const StateFinite<fp32> &tensors);
template size_t tools::finite::measure::length(const StateFinite<fp64> &tensors);
template size_t tools::finite::measure::length(const StateFinite<fp128> &tensors);
template size_t tools::finite::measure::length(const StateFinite<cx32> &tensors);
template size_t tools::finite::measure::length(const StateFinite<cx64> &tensors);
template size_t tools::finite::measure::length(const StateFinite<cx128> &tensors);

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::norm(const StateFinite<Scalar> &state, bool full) {
    if(state.measurements.norm) return state.measurements.norm.value();
    cx64 norm;
    auto t_norm = tid::tic_scope("norm", tid::level::highest);
    if(not full) {
        // We know the all sites are normalized. We can check that the current position is normalized
        const auto  pos = std::clamp(state.template get_position<long>(), 0l, state.template get_length<long>());
        const auto &mps = state.get_mps_site(pos);
        tools::log->trace("Measuring norm using site {} with dimensions {}", pos, mps.dimensions());
        norm = tools::common::contraction::contract_mps_norm(mps.get_M());
    } else {
        tools::log->trace("Measuring norm on full chain");
        Eigen::Tensor<Scalar, 2> chain;
        Eigen::Tensor<Scalar, 2> temp;
        bool                   first   = true;
        auto                  &threads = tenx::threads::get();

        for(const auto &mps : state.mps_sites) {
            const auto &M = mps->get_M();
            if(first) {
                chain = tools::common::contraction::contract_mps_partial<std::array{0l, 1l}>(M);
                first = false;
                continue;
            }
            temp.resize(tenx::array2{M.dimension(2), M.dimension(2)});
            temp.device(*threads->dev) = chain.contract(M, tenx::idx({0}, {1})).contract(M.conjugate(), tenx::idx({0, 1}, {1, 0}));

            chain = std::move(temp);
        }
        norm = tenx::MatrixMap(chain).trace();
    }
    if(std::abs(norm - 1.0) > settings::precision::max_norm_error) tools::log->debug("norm: far from unity: {:.16f}{:+.16f}i", norm.real(), norm.imag());
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}
template fp32  tools::finite::measure::norm(const StateFinite<fp32> &state, bool full);
template fp64  tools::finite::measure::norm(const StateFinite<fp64> &state, bool full);
template fp128 tools::finite::measure::norm(const StateFinite<fp128> &state, bool full);
template fp32  tools::finite::measure::norm(const StateFinite<cx32> &state, bool full);
template fp64  tools::finite::measure::norm(const StateFinite<cx64> &state, bool full);
template fp128 tools::finite::measure::norm(const StateFinite<cx128> &state, bool full);