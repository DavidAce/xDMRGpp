#pragma once
#include "config/settings.h"
#include "debug/info.h"
#include "io/fmt_custom.h"
#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "norm.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"

using tools::finite::measure::RealScalar;

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::norm(const StateFinite<Scalar> &state, bool full) {
    if(state.measurements.norm) return state.measurements.norm.value();
    Scalar norm;
    auto   t_norm = tid::tic_scope("norm", tid::level::highest);
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
        bool                     first   = true;
        auto                    &threads = tenx::threads::get();

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
    auto normTol = std::numeric_limits<RealScalar<Scalar>>::epsilon() * settings::precision::max_norm_slack;
    auto normErr = std::abs(norm - RealScalar<Scalar>{1});

    if(normErr > normTol) tools::log->debug("norm: far from unity: {:.5e}", fp(normErr));
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}
