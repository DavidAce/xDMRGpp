#pragma once
#include "../entanglement_entropy.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
using tools::finite::measure::RealScalar;

enum class EntropyScale { LN, LOG2 };
template<EntropyScale es, typename Scalar>
[[nodiscard]] inline RealScalar<Scalar> von_neumann_entropy_from_schmidt(const Eigen::Tensor<Scalar, 1> &L) {
    using Real = RealScalar<Scalar>;

    // Eigen::Tensor uses Index for size(), but long is fine as loop counter too
    const Eigen::Index n = L.size();
    if(n <= 0) return Real{0};

    Real sum = Real{0};

#pragma omp parallel for reduction(+ : sum)
    for(Eigen::Index i = 0; i < n; ++i) {
        const Real p = std::abs(L(i) * L(i));

        // Skip exact zeros and any negative junk
        if(p <= Real{0}) continue;

        // Accumulate -p * log(p)
        if constexpr(es == EntropyScale::LN) {
            sum += -p * std::log(p);
        } else {
            sum += -p * std::log2(p);
        }
    }

    // Entropy should be nonnegative; guard small negative due to roundoff
    sum = std::max(Real{0}, sum);
    return sum;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::entanglement_entropy(const Eigen::Tensor<Scalar, 1> &L) {
    auto t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    return von_neumann_entropy_from_schmidt<EntropyScale::LN>(L);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::entanglement_entropy_current(const StateFinite<Scalar> &state) {
    if(state.measurements.entanglement_entropy_current) return state.measurements.entanglement_entropy_current.value();
    if(state.has_center_point()) {
        state.measurements.entanglement_entropy_current = entanglement_entropy(state.current_bond());
    } else {
        tools::log->trace("entanglement_entropy_current: state has no center point!");
        state.measurements.entanglement_entropy_current = 0;
    }
    return state.measurements.entanglement_entropy_current.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::entanglement_entropy_midchain(const StateFinite<Scalar> &state) {
    if(state.measurements.entanglement_entropy_midchain) return state.measurements.entanglement_entropy_midchain.value();
    state.measurements.entanglement_entropy_midchain = entanglement_entropy(state.get_midchain_bond());
    return state.measurements.entanglement_entropy_midchain.value();
}

template<typename Scalar>
std::vector<RealScalar<Scalar>> tools::finite::measure::entanglement_entropies(const StateFinite<Scalar> &state) {
    if(state.measurements.entanglement_entropies) return state.measurements.entanglement_entropies.value();
    auto                            t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    std::vector<RealScalar<Scalar>> entanglement_entropies;
    entanglement_entropies.reserve(state.get_length() + 1);
    if(not state.has_center_point()) entanglement_entropies.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        entanglement_entropies.emplace_back(entanglement_entropy(mps->get_L()));
        if(mps->isCenter()) {
            entanglement_entropies.emplace_back(entanglement_entropy(mps->get_LC()));
            state.measurements.entanglement_entropy_current = entanglement_entropies.back();
        }
    }
    if(entanglement_entropies.size() != state.get_length() + 1) throw except::logic_error("entanglement_entropies.size() should be length+1");
    if(entanglement_entropies.front() != 0) throw except::logic_error("First entropy should be 0. Got: {:.16f}", fp(entanglement_entropies.front()));
    if(entanglement_entropies.back() != 0) throw except::logic_error("Last entropy should be 0. Got: {:.16f}", fp(entanglement_entropies.back()));
    state.measurements.entanglement_entropy_midchain = entanglement_entropies[state.template get_length<size_t>() / 2];
    state.measurements.entanglement_entropies        = entanglement_entropies;
    return state.measurements.entanglement_entropies.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::entanglement_entropy_log2(const StateFinite<Scalar> &state, size_t nsites /* sites to the left of the partition */) {
    auto t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    if(nsites == 0ul) return 0.0;
    auto pos_pl1 = static_cast<size_t>(state.template get_position<long>() + 1l);
    auto pos_tgt = 0ul;
    if(pos_pl1 < nsites) {
        // Get the L of the B-site at pos == nsites-1
        pos_tgt = nsites - 1ul;
    } else if(pos_pl1 == nsites) {
        // Get the LC of the AC site at pos == nsites-1
        pos_tgt = nsites - 1ul;
    } else {
        // Get the L of the A (or AC) -site at pos == nsites
        pos_tgt = nsites;
    }
    const auto &mps = state.get_mps_site(pos_tgt);
    if(mps.isCenter()) {
        auto &LC = mps.get_LC();
        return von_neumann_entropy_from_schmidt<EntropyScale::LOG2>(LC);
    } else {
        auto &L = mps.get_L();
        return von_neumann_entropy_from_schmidt<EntropyScale::LOG2>(L);
    }
}

template<typename Scalar>
std::vector<RealScalar<Scalar>> tools::finite::measure::entanglement_entropies_log2(const StateFinite<Scalar> &state) {
    auto                            t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    std::vector<RealScalar<Scalar>> entanglement_entropies;
    entanglement_entropies.reserve(state.get_length() + 1);
    if(not state.has_center_point()) entanglement_entropies.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        auto              &L  = mps->get_L();
        RealScalar<Scalar> SE = von_neumann_entropy_from_schmidt<EntropyScale::LOG2>(L);
        entanglement_entropies.emplace_back(SE);
        if(mps->isCenter()) {
            auto &LC = mps->get_LC();
            SE       = von_neumann_entropy_from_schmidt<EntropyScale::LOG2>(LC);
            entanglement_entropies.emplace_back(SE);
        }
    }
    if(entanglement_entropies.size() != state.get_length() + 1) throw except::logic_error("entanglement_entropies.size() should be length+1");
    if(entanglement_entropies.front() != 0) throw except::logic_error("First entropy should be 0. Got: {:.16f}", fp(entanglement_entropies.front()));
    if(entanglement_entropies.back() != 0) throw except::logic_error("Last entropy should be 0. Got: {:.16f}", fp(entanglement_entropies.back()));
    return entanglement_entropies;
}

template<typename Scalar>
std::vector<RealScalar<Scalar>> tools::finite::measure::renyi_entropies(const StateFinite<Scalar> &state, double q) {
    auto inf = std::numeric_limits<double>::infinity();
    if(q == 1.0) return entanglement_entropies(state);
    if(q == 2.0 and state.measurements.renyi_2) return state.measurements.renyi_2.value();
    if(q == 3.0 and state.measurements.renyi_3) return state.measurements.renyi_3.value();
    if(q == 4.0 and state.measurements.renyi_4) return state.measurements.renyi_4.value();
    if(q == inf and state.measurements.renyi_inf) return state.measurements.renyi_inf.value();
    auto t_ren = tid::tic_scope("renyi_entropy", tid::level::highest);
    using Real = RealScalar<Scalar>;
    std::vector<RealScalar<Scalar>> renyi_q;
    renyi_q.reserve(state.get_length() + 1);
    if(not state.has_center_point()) renyi_q.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        const auto &L = mps->get_L();
        if(L.size() == 0) throw except::logic_error("renyi_entropies: empty bond L");
        Eigen::Tensor<Scalar, 0> RE;
        if(q == inf) {
            const auto Lmax = Eigen::Tensor<Real, 0>(L.abs().maximum());
            const auto l0   = std::max<Real>(Lmax(0), std::numeric_limits<Real>::min());
            RE(0)           = RealScalar<Scalar>(-2.0) * std::log(l0);
        } else
            RE = RealScalar<Scalar>(1.0 / (1.0 - q)) * L.pow(Real(2.0 * q)).sum().log();
        renyi_q.emplace_back(std::abs(RE(0)));
        if(mps->isCenter()) {
            const auto &LC = mps->get_LC();
            if(LC.size() == 0) throw except::logic_error("renyi_entropies: empty bond LC");
            if(q == inf) {
                const auto LCmax = Eigen::Tensor<Real, 0>(LC.abs().maximum());
                const auto lc0   = std::max<Real>(LCmax(0), std::numeric_limits<Real>::min());
                RE(0)            = Real(-2.0) * std::log(lc0);
            } else
                RE = RealScalar<Scalar>(1.0 / (1.0 - q)) * LC.pow(Real(2.0 * q)).sum().log();
            renyi_q.emplace_back(std::abs(RE(0)));
        }
    }
    if(renyi_q.size() != state.get_length() + 1) throw except::logic_error("renyi_q.size() should be length+1");
    if(q == 2.0) {
        state.measurements.renyi_2 = renyi_q;
        return state.measurements.renyi_2.value();
    }
    if(q == 3.0) {
        state.measurements.renyi_3 = renyi_q;
        return state.measurements.renyi_3.value();
    }
    if(q == 4.0) {
        state.measurements.renyi_4 = renyi_q;
        return state.measurements.renyi_4.value();
    }
    if(q == inf) {
        state.measurements.renyi_inf = renyi_q;
        return state.measurements.renyi_inf.value();
    }
    return renyi_q;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::renyi_entropy_midchain(const StateFinite<Scalar> &state, double q) {
    auto inf = std::numeric_limits<double>::infinity();
    if(q == 1.0) return entanglement_entropy_midchain(state);
    auto                     t_ren = tid::tic_scope("renyi_entropy_midchain", tid::level::highest);
    auto                    &LC    = state.get_midchain_bond();
    Eigen::Tensor<Scalar, 0> renyi_q;
    if(q == inf)
        renyi_q(0) = RealScalar<Scalar>(-2.0) * std::log(LC(0));
    else
        renyi_q = RealScalar<Scalar>(1.0 / (1.0 - q)) * LC.pow(RealScalar<Scalar>(2.0 * q)).sum().log();
    return std::abs(renyi_q(0));
}
