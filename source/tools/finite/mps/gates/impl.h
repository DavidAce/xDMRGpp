
#pragma once
#include "math/tenx.h"
// -- (textra first)
#include "../../mps.h"
#include "config/enums.h"
#include "config/settings.h"
#include "general/iter.h"
#include "math/num.h"
#include "math/svd.h"
#include "qm/gate.h"
#include "qm/mpo.h"
#include "qm/spin.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"

namespace settings {
    inline constexpr bool debug_gates   = false;
    inline constexpr bool verbose_gates = false;
}

template<typename GateType>
std::vector<size_t> tools::finite::mps::generate_gate_sequence(long state_len, long state_pos, const std::vector<GateType> &gates, CircuitOp cop,
                                                               bool range_long_to_short) {
    // Generate a list of staggered indices, without assuming that gates are sorted in any way
    // Consider a sequence of short-range gates such as [0,1], [1,2], [2,3], then this function is used to generate a new sequence without overlaps:
    //
    //  * short range 2-site gates:
    //         * layer 0: [0,1],[2,3],[4,5]... and so on,
    //         * layer 1: [1,2],[3,4],[5,6]..., i.e. even sites first, then odd sites,
    //  * short range 3-site gates:
    //         * layer 0: [0,1,2], [3,4,5]...
    //         * layer 1: [1,2,3], [4,5,6]...,
    //         * layer 2: [2,3,4], [5,6,7], and so on.
    //
    // To avoid having to move/swap all the way back between layers, one can flip odd layers to generate a zigzag pattern.
    // Then, when applying the adjoint operation, both the layer and gates within that layer are flipped
    //
    // Now consider a sequence of long-range gates such as [0,1], [0,2], ... [0,L-1], [1,2], [1,3]...., [1,L-1]
    // To generate a performant sequence of gates with respect to swaps, it's important to note which gates commute.
    // In this implementation we make the assumption that gates [i,j] and [i,k] always commute (i != j != k).
    // Thus, we get the following sequence
    //  * long range 2-site gates:
    //         * layer 0: [0,1],[0,2],[0,3]... then [2,3],[2,4],[2,5]... then [4,5],[4,6][4,7]... and so on
    //         * layer 1: [1,2],[1,3],[1,4]... then [3,4],[3,5],[3,6]... then [5,6],[5,7][5,8]... and so on
    //
    // This way we get as many reverse swap cancellations as possible while maintaining some support
    // for non-commutativity on short-range interactions
    //
    // For performance reasons we may want to apply gates from long to short range:
    // * flipped long range 2-site gates:
    //        * layer 0: ... [0,3],[0,2],[0,1] then ... [2,5],[2,4],[2,3] then ... [4,7],[4,6][4,5] and so on
    //        * layer 1: ... [1,4],[1,3],[1,2] then ... [3,6],[3,5],[3,4] then ... [5,8],[5,7][5,6] and so on
    //
    // Performance note:
    // If the state is at position L-1, and the list generated has to start from 0, then L-1 moves have
    // to be done before even starting. Additionally, if unlucky, we have to move/swap L-1 times again to return
    // to the original position.
    //    if(state.get_direction() < 0 and past_middle) state.flip_direction(); // Turn direction away from middle
    //    if(state.get_direction() > 0 and not past_middle) state.flip_direction(); // Turn direction away from middle
    auto                                          t_gen = tid::tic_scope("gen_gate_seq", tid::level::highest);
    std::vector<std::vector<std::vector<size_t>>> layers; // Layer --> group --> idx
    std::vector<size_t>                           idx = num::range<size_t>(0, gates.size());
    while(not idx.empty()) {
        std::vector<std::vector<size_t>> layer; // Group --> idx
        std::vector<size_t>              idx_used;
        size_t                           at = gates.at(idx.front()).pos.front(); // Left most leg of the gate at idx[0].
        for(size_t i = 0; i < idx.size(); i++) {
            const auto &gate_i = gates.at(idx.at(i)); // Gate at idx[i]
            if(gate_i.pos.front() >= at) {
                auto                back_i = gate_i.pos.back(); // Right-most leg of gate idx[i]
                std::vector<size_t> group_i;                    // List of idx with gates starting at the same position (i.e. same "at")
                for(size_t j = i; j < idx.size(); j++) {        // Look ahead
                    // In this part we accept a gate if gate j is further ahead than i, without overlap.
                    const auto &gate_j  = gates.at(idx.at(j)); // gate_i == gate_j on first iteration of j, so we always accept
                    auto        front_j = gate_j.pos.front();
                    auto        back_j  = gate_j.pos.back();
                    if(front_j == at and back_j >= back_i) {
                        group_i.emplace_back(idx.at(j));
                        idx_used.emplace_back(idx.at(j));
                    }
                }
                // Append the group in reverse order to the current layer
                if(range_long_to_short) std::reverse(group_i.begin(), group_i.end());
                layer.emplace_back(group_i);
                at = back_i + 1; // Move the position forward to a gate that doesn't overlap with the gate at idx[0]
            }
        }
        // Append the layer
        layers.emplace_back(layer);

        // Remove the used elements from idx by value
        for(const auto &i : iter::reverse(idx_used)) idx.erase(std::remove(idx.begin(), idx.end(), i), idx.end());
    }

    // Reverse every other layer to get a zigzag pattern
    // If the state is past the middle, reverse layers 0,2,4... otherwise 1,3,5...
    size_t past_middle = state_pos > state_len / 2 ? 0 : 1;
    for(const auto &[i, layer] : iter::enumerate(layers)) {
        if(num::mod<size_t>(i, 2) == past_middle) std::reverse(layer.begin(), layer.end());
    }

    // To apply inverse we reverse the layers
    // Note that we don't need to reverse groups because we assumed that these commute
    bool reverse = cop != CircuitOp::NONE;
    if(reverse) {
        for(auto &layer : layers) std::reverse(layer.begin(), layer.end());
        std::reverse(layers.begin(), layers.end());
    }

    // Move the layers into a long sequence of positions
    std::vector<size_t> gate_sequence;
    for(auto &layer : layers)
        for(auto &group : layer) gate_sequence.insert(gate_sequence.end(), std::make_move_iterator(group.begin()), std::make_move_iterator(group.end()));

    if(gate_sequence.size() != gates.size()) throw except::logic_error("gate_sequence.size() {} != gates.size() {}", gate_sequence.size(), gates.size());
    return gate_sequence;
}


template<typename Scalar>
void tools::finite::mps::apply_gate(StateFinite<Scalar> &state, const qm::Gate &gate, GateOp gop, GateMove gm, std::optional<svd::config> svd_cfg) {
    if(gate.pos.back() >= state.get_length()) throw except::logic_error("The last position of gate is out of bounds: {}", gate.pos);
    if(gm == GateMove::AUTO) gm = GateMove::ON;

    if constexpr(!sfinae::is_std_complex_v<Scalar>) {
        if(!tenx::isReal(gate.op))
            throw except::runtime_error("apply_gate<{}> requires real-valued gates, but the provided gate has an imaginary part", sfinae::type_name<Scalar>());
    }

    auto old_posC = state.template get_position<long>();
    auto old_svds = svd::solver::get_count();

    if constexpr(settings::verbose_gates)
        tools::log->trace("apply_gate: status  pos {} | op {} | gm {} | center {} | svds {} | labels {}", gate.pos, enum2sv(gop), enum2sv(gm), old_posC,
                          old_svds, state.get_labels());
    if(gm == GateMove::ON) {
        // When gm == GateMove::ON, applying swap operators automatically moves the center position onto posL. This can
        // happen IFF the center position is already on one of posL-1, posL or posR.
        // To get this process going we need to move the center into one of these positions first.
        // Note 1: posL and posR here refer to the first swap in the gate if it exists. Otherwise, they are front/back of gate.pos.
        // Note 2: both the swap and apply operations will set the current center on the left-most site of the gate.
        auto tgt_gate = gate.pos;
        auto tgt_posC = std::clamp<long>(old_posC, safe_cast<long>(tgt_gate.front()) - 1l, safe_cast<long>(tgt_gate.back()));
        auto num_step = move_center_point_to_pos_dir(state, tgt_posC, 1, svd_cfg);
        if constexpr(settings::verbose_gates)
            if(num_step > 0)
                tools::log->trace("apply_gate: moved   pos {} | op {} | gm {} | center {} -> {} | tgt {} | steps {} | svds {}", gate.pos, enum2sv(gop),
                                  enum2sv(gm), old_posC, tgt_posC, tgt_gate, num_step, svd::solver::get_count());
        old_posC = tgt_posC; // Update the current position
    }

    // Generate the mps object on which to apply the gate
    auto mps = state.template get_multisite_mps<Scalar>(gate.pos);
    /* Apply the gate
        1 --- [mps] --- 2
                |
                0
                1
                |
               [G]
                |
                0

     */
    Eigen::Tensor<Scalar, 3> tmp(mps.dimensions());
    {
        auto t_apply = tid::tic_token("apply", tid::level::highest);
        //        auto &threads = tenx::threads::get();
        //        const auto &gateop  = gate.unaryOp(gop); // Apply any pending modifier like transpose, adjoint or conjugation
        //        tmp.resize(std::array<long, 3>{gateop.dimension(1), multisite_mps.dimension(1), multisite_mps.dimension(2)});
        //        tmp.device(*threads->dev) = gateop.contract(multisite_mps, tenx::idx({1}, {0}));
        assert(gate.op.dimension(0) == gate.op.dimension(1)); // This assumption makes it ok to take the adjoint and transpose below without resizing tmp!
        auto gatemap = tenx::asScalarType<Scalar>(tenx::MatrixMap(gate.op));
        auto mps_map = tenx::MatrixMap(mps, mps.dimension(0), mps.dimension(1) * mps.dimension(2));
        auto tmp_map = tenx::MatrixMap(tmp, tmp.dimension(0), tmp.dimension(1) * tmp.dimension(2));
        switch(gop) { // These will call gemm (hopefully from blas
            case GateOp::NONE: tmp_map = gatemap * mps_map; break;
            case GateOp::CNJ: tmp_map = gatemap.conjugate() * mps_map; break;
            case GateOp::ADJ: tmp_map = gatemap.adjoint() * mps_map; break;
            case GateOp::TRN: tmp_map = gatemap.transpose() * mps_map; break;
            default: throw except::runtime_error("Unhandled switch case for GateOp: {}", enum2sv(gop));
        }
    }
    if constexpr(settings::verbose_gates)
        tools::log->trace("apply_gate: applied pos {} | op {} | gm {} | svds {} | bond {} | trnc {:.3e}", gate.pos, enum2sv(gop), enum2sv(gm),
                          svd::solver::get_count(), state.get_mps_site(gate.pos.front()).get_chiR(), state.get_truncation_error(gate.pos.front()));

    auto new_posC = gm == GateMove::ON ? safe_cast<long>(gate.pos.front()) : old_posC;
    tools::finite::mps::merge_multisite_mps(state, tmp, gate.pos, new_posC, MergeEvent::GATE, svd_cfg, LogPolicy::SILENT);
    if constexpr(settings::verbose_gates)
        tools::log->trace("apply_gate: merged  pos {} | op {} | gm {} | center {} -> {} | svds {} | bond {} | trnc {:.3e}", gate.pos, enum2sv(gop), enum2sv(gm),
                          old_posC, new_posC, svd::solver::get_count(), state.get_mps_site(gate.pos.front()).get_chiR(),
                          state.get_truncation_error(gate.pos.front()));
}

template<typename Scalar>
void tools::finite::mps::apply_gates(StateFinite<Scalar> &state, const std::vector<Eigen::Tensor<Scalar, 2>> &nsite_tensors, size_t gate_size, CircuitOp cop,
                                     bool moveback, GateMove gm, std::optional<svd::config> svd_cfg) {
    // Pack the two-site operators into a vector of qm::Gates
    // using RealScalar = decltype(std::real(std::declval<Scalar>()));
    // using CplxScalar = std::complex<RealScalar>;
    std::vector<qm::Gate> gates;
    gates.reserve(nsite_tensors.size());
    for(const auto &[idx, op] : iter::enumerate(nsite_tensors)) {
        auto pos = num::range<size_t>(idx, idx + gate_size, 1);
        auto dim = std::vector<long>(pos.size(), 2);
        gates.emplace_back(qm::Gate(tenx::asScalarType<cx64>(nsite_tensors[idx]), pos, dim));
    }
    apply_gates(state, gates, cop, moveback, gm, svd_cfg);
}


template<typename Scalar>
void tools::finite::mps::apply_gates(StateFinite<Scalar> &state, const std::vector<qm::Gate> &gates, CircuitOp cop, bool moveback, GateMove gm,
                                     std::optional<svd::config> svd_cfg) {
    auto t_apply_gates = tid::tic_scope("apply_gates", tid::level::higher);

    if(gates.empty()) return;
    auto svd_count     = svd::solver::get_count();
    auto gate_sequence = generate_gate_sequence(state.template get_length<long>(), state.template get_position<long>(), gates, cop);
    if constexpr(settings::verbose_gates)
        tools::log->trace("apply_gates: current pos {} dir {} | gate_sequence {}", state.template get_position<long>(), state.get_direction(), gate_sequence);

    state.clear_cache();
    GateOp gop = GateOp::NONE;
    switch(cop) {
        case CircuitOp::NONE: gop = GateOp::NONE; break;
        case CircuitOp::ADJ: gop = GateOp::ADJ; break;
        case CircuitOp::TRN: gop = GateOp::TRN; break;
    }
    for(const auto &idx : gate_sequence) apply_gate(state, gates.at(idx), gop, gm, svd_cfg);

    if(moveback) move_center_point_to_pos_dir(state, 0, 1, svd_cfg);
    svd_count = svd::solver::get_count() - svd_count;
    if constexpr(settings::verbose_gates)
        tools::log->debug("apply_gates: applied {} gates | svds {} | time {:.4f}", gates.size(), svd_count, t_apply_gates->get_last_interval());
}

template<typename Scalar>
void tools::finite::mps::apply_circuit(StateFinite<Scalar> &state, const std::vector<std::vector<qm::Gate>> &circuit, CircuitOp cop, bool moveback, GateMove gm,
                                       std::optional<svd::config> svd_cfg) {
    //    tools::log->debug("Applying circuit | adjoint: {} | gm {}", adjoint, enum2sv(gm));
    switch(cop) {
        case CircuitOp::ADJ:
        case CircuitOp::TRN: {
            for(const auto &gates : iter::reverse(circuit)) apply_gates(state, gates, cop, moveback, gm, svd_cfg);
            break;
        }
        case CircuitOp::NONE: {
            for(const auto &gates : circuit) apply_gates(state, gates, cop, moveback, gm, svd_cfg);
            break;
        }
    }
}


template<typename T>
std::vector<T> get_site_idx(const std::vector<size_t> &sites, const std::vector<size_t> &pos) {
    std::vector<T> idx;
    if(sites.empty()) {
        for(const auto &p : pos) idx.emplace_back(static_cast<T>(p));
    } else {
        for(const auto &p : pos) {
            auto p_it = std::find(sites.begin(), sites.end(), p);
            if(p_it == sites.end()) throw except::logic_error("state position of gate pos {} not found in {}", p, sites);
            idx.emplace_back(static_cast<T>(std::distance(sites.begin(), p_it)));
        }
    }
    return idx;
}

template<typename Scalar>
void tools::finite::mps::swap_sites(StateFinite<Scalar> &state, size_t posL, size_t posR, std::vector<size_t> &sites, GateMove gm,
                                    std::optional<svd::config> svd_cfg) {
    /* The swap operation takes two neighboring sites
     *
     * (1)chiL---[ mpsL ]---chiC(2)    chiC(1)---[ mpsR ]---(2)chiR
     *              |                               |
     *           (0)dL                            (0)dR
     *
     *
     * and joins them in multisite standard form, i.e., in the order:
     *
     *      1) physical indices di (any number of them, contracted from left to right)
     *      2) left bond index chiL
     *      3) right bond index chiR
     *
     * (1)chiL ---[tensor]--- (2)chiR
     *               |
     *          (0)d*d*d...
     *
     * For the swap operation, we reshape this into a rank-4, splitting up the physical dimensions
     *
     * (2)chiL---[  tensor  ]---(3)chiR
     *            |        |
     *        (0)dL      (1)dR
     *
     * Then the swap operator is applied
     *
     * (2)chiL---[  tensor  ]---(3)chiR
     *            |        |
     *             \      /
     *               \  /
     *                /\      <---- Swap
     *              /    \
     *            /        \
     *            |        |
     *         (0)dR     (1)dL
     *
     *
     * For a schmidt decomposition we need to shuffle this object into standard form for schmidt decomposition
     *
     *   (1)chiL---[  tensor  ]---(3)chiR
     *              |        |
     *          (0)dL      (2)dR
     *
     * Then a schmidt decomposition returns

     * (1)chiL---[  U  ]---chiC(2) chiC(0)---[S]---chiC(1)  chiC(1)---[ V ]---(2)chiR
     *              |                                                   |
     *           (0)dR                                                (0)dL
     *
     *
     * In practice the swap operation can be performed as a reshape + shuffle
     *
     * (1)chiL ---[tensor]--- (2)chiR     reshape      (2)chiL---[  tensor  ]---(3)chiR    shuffle          (1)chiL---[  tensor  ]---(3)chiR
     *               |                      ---->                 |        |               ----->                      |        |
     *          (0)d*d*d...                                    (0)dL      (1)dR                                      (0)dR      (2)dL
     *
     *
     *  Center position:
     *  Before swapping, one of the sites is supposed to be an "AC" site:
     *      Swap: posL is AC, posR is B
     *      Rwap: posL is A, posR is AC
     *
     *  After swapping, the AC-site moves:
     *      Swap: posL becomes A, posR becomes AC
     *      Rwap: posL becomes AC, posR becomes A
     *
     *  This is handled by the merge operation, which takes the "new" center position as argument
     *      Swap: center_position = posR
     *      Rwap: center_position = posL
     *
     */
    auto t_swap = tid::tic_scope("swap", tid::level::highest);
    if(posR != posL + 1) throw except::logic_error("Expected posR == posL+1. Got posL {} and posR {}", posL, posR);
    if(posR != std::clamp(posR, 0ul, state.template get_length<size_t>() - 1ul))
        throw except::logic_error("Expected posR in [0,{}]. Got {}", state.get_length() - 1, posR);
    if(posL != std::clamp(posL, 0ul, state.template get_length<size_t>() - 1ul))
        throw except::logic_error("Expected posL in [0,{}]. Got {}", state.get_length() - 1, posL);

    if(gm == GateMove::AUTO) gm = GateMove::ON;
    auto           old_svd            = svd::solver::get_count();
    auto           old_pos            = state.template get_position<long>();
    auto           dimL               = state.get_mps_site(posL).dimensions();
    auto           dimR               = state.get_mps_site(posR).dimensions();
    auto           dL                 = dimL[0];
    auto           dR                 = dimR[0];
    auto           chiL               = dimL[1];
    auto           chiR               = dimR[2];
    auto           rsh4               = tenx::array4{dL, dR, chiL, chiR};
    constexpr auto shf4               = tenx::array4{1, 0, 2, 3};
    auto           rsh3               = tenx::array3{dR * dL, chiL, chiR};
    auto           swapped_mps        = Eigen::Tensor<Scalar, 3>(rsh3);
    auto          &threads            = tenx::threads::get();
    swapped_mps.device(*threads->dev) = state.template get_multisite_mps<Scalar>({posL, posR})
                                            .reshape(rsh4)
                                            .shuffle(shf4)  // swap
                                            .reshape(rsh3); // prepare for merge
    auto new_pos = old_pos;
    if(gm == GateMove::ON)
        new_pos = safe_cast<long>(posL); // The benefit of GateMove::ON is to prefer "AC-B" splits that require a single SVD as often as possible
                                         //    double lastS_old = 0;
                                         //    long   bondd_old = 0;
                                         //    auto   label_old = state.get_mps_site(posL).get_label();
                                         //    if(label_old == "A") {
                                         //        auto &mpsR = state.get_mps_site(posR);
                                         //        bondd_old  = mpsR.get_L().size();
                                         //        lastS_old  = mpsR.get_L().coeff(bondd_old - 1).real();
                                         //    }
                                         //    if(label_old == "AC") {
                                         //        auto &mpsL = state.get_mps_site(posL);
                                         //        bondd_old  = mpsL.get_LC().size();
                                         //        lastS_old  = mpsL.get_LC().coeff(bondd_old - 1).real();
                                         //    }
                                         //    if(label_old == "B") {
                                         //        auto &mpsL = state.get_mps_site(posL);
                                         //        bondd_old  = mpsL.get_L().size();
                                         //        lastS_old  = mpsL.get_L().coeff(bondd_old - 1).real();
                                         //    }

    // This SVD shouldn't modify the current mps, so no truncation here (no svd config)
    //    merge_multisite_mps(state, swapped_mps, {posL, posR}, new_pos, svd_cfg, LogPolicy::SILENT);
    merge_multisite_mps(state, swapped_mps, {posL, posR}, new_pos, MergeEvent::SWAP, svd_cfg, LogPolicy::SILENT);

    //    double lastS_new = 0;
    //    long   bondd_new = 0;
    //    auto   label_new = state.get_mps_site(posL).get_label();
    //    if(label_new == "A") {
    //        auto &mpsR = state.get_mps_site(posR);
    //        bondd_new  = mpsR.get_L().size();
    //        lastS_new  = mpsR.get_L().coeff(bondd_new - 1).real();
    //    }
    //    if(label_new == "AC") {
    //        auto &mpsL = state.get_mps_site(posL);
    //        bondd_new  = mpsL.get_LC().size();
    //        lastS_new  = mpsL.get_LC().coeff(bondd_new - 1).real();
    //    }
    //    if(label_new == "B") {
    //        auto &mpsL = state.get_mps_site(posL);
    //        bondd_new  = mpsL.get_L().size();
    //        lastS_new  = mpsL.get_L().coeff(bondd_new - 1).real();
    //    }
    //
    //    tools::log->info("swap pos [{}, {}] gm {} | center {} -> {} | bond dim {} -> {} | last singular value {:.3e} -> {:.3e}", posL, posR, enum2sv(gm),
    //    old_pos,
    //                     new_pos, bondd_old, bondd_new, lastS_old, lastS_new);

    std::swap(sites[posL], sites[posR]);
    // Sanity check
    if constexpr(settings::verbose_gates) {
        tools::log->trace("swap_sites     : swapped pos [{}, {}] | idx {} | gm {} | center {} -> {} | svds {} -> {} | labels {} | sites {}", posL, posR,
                          get_site_idx<size_t>(sites, {posL, posR}), enum2sv(gm), old_pos, new_pos, old_svd, svd::solver::get_count(), state.get_labels(),
                          sites);
    }
    if constexpr(settings::debug_gates)
        for(const auto &mps : state.mps_sites) mps->assert_normalized();
}

template<typename Scalar>
void tools::finite::mps::apply_swap_gate(StateFinite<Scalar> &state, const qm::SwapGate &gate, GateOp gop, std::vector<size_t> &sites, GateMove gm,
                                         std::optional<svd::config> svd_cfg) {
    if(gate.pos.back() >= state.get_length()) throw except::logic_error("The last position of gate is out of bounds: {}", gate.pos);
    if constexpr(!sfinae::is_std_complex_v<Scalar>) {
        if(!tenx::isReal(gate.op))
            throw except::runtime_error("apply_swap_gate<{}> requires real-valued gates, but the provided gate has an imaginary part",
                                        sfinae::type_name<Scalar>());
    }

    auto old_posC = state.template get_position<long>();
    auto old_svds = svd::solver::get_count();
    auto pos_idxs = get_site_idx<size_t>(sites, gate.pos); // Site on the state where the gate positions are currently located

    // pos_idx has the indices pointing to where the gate positions are located on the chain right now, after many swaps.
    // Example:
    //      sites    == [0,2,3,1,4,5]
    //      gate.pos == [1,4]
    //      pos_idx  == [3,4] <--- valid if diff(pos_idx) == 1, before a gate is applied
    // Note that the point of swap gates is to apply gates on nearest neighbors. Therefore, pos_idx should always
    // be a vector of elements with increments by 1 for the gate to be applicable.

    if constexpr(settings::verbose_gates)
        tools::log->trace("apply_swap_gate: status  pos {} | idx {} | gm {} | center {} | svds {} | labels {} | sites {}", gate.pos, pos_idxs, enum2sv(gm),
                          old_posC, old_svds, state.get_labels(), sites);

    if(gm == GateMove::ON) {
        // When gm == GateMove::ON, applying swap operators automatically moves the center position onto posL. This can
        // happen IFF the center position is already on one of posL-1, posL or posR.
        // To get this process going we need to move the center into one of these positions first.
        // Note 1: posL and posR here refer to the first swap in the gate if it exists. Otherwise, they are front/back of gate.pos.
        // Note 2: both the swap and apply operations will set the current center on the left-most site of the gate.
        auto tgt_gate = gate.swaps.empty() ? gate.pos : std::vector<size_t>{gate.swaps[0].posL, gate.swaps[0].posR};
        auto tgt_posC = std::clamp<long>(old_posC, safe_cast<long>(tgt_gate.front()) - 1l, safe_cast<long>(tgt_gate.back()));
        auto num_step = move_center_point_to_pos_dir(state, tgt_posC, 1, svd_cfg);
        if constexpr(settings::verbose_gates)
            if(num_step > 0)
                tools::log->trace("apply_swap_gate: moved  pos {} | gm {} | center {} -> {} | tgt {} | steps {} | svds {}", gate.pos, enum2sv(gm), old_posC,
                                  tgt_posC, tgt_gate, num_step, svd::solver::get_count());
        old_posC = tgt_posC; // Update the current position
    }

    // with i<j, start by applying all the swap operators S(i,j), which move site i to j-1
    for(const auto &s : gate.swaps) swap_sites(state, s.posL, s.posR, sites, gm);

    // Refresh the site indices of the gate after having swapped a lot.
    pos_idxs = get_site_idx<size_t>(sites, gate.pos);

    // Check that idx elements increase by one
    if(pos_idxs.size() >= 2) {
        for(size_t i = 1; i < pos_idxs.size(); i++) {
            if(pos_idxs[i] != pos_idxs[i - 1] + 1) throw except::logic_error("Invalid pos idx {} | gate.pos {} | sites {}", pos_idxs, gate.pos, sites);
        }
    }

    // Generate the mps object on which to apply the gate
    auto mps = state.template get_multisite_mps<Scalar>(pos_idxs);

    // Apply the gate
    Eigen::Tensor<Scalar, 3> tmp(mps.dimensions());
    {
        auto t_apply = tid::tic_token("apply", tid::level::highest);
        //        const auto &gateop  = gate.unaryOp(gop); // Apply any pending modifier like transpose, adjoint or conjugation
        //        auto       &threads = tenx::threads::get();
        //        temp.resize(std::array<long, 3>{gateop.dimension(1), multisite_mps.dimension(1), multisite_mps.dimension(2)});
        //        temp.device(*threads->dev) = gateop.contract(multisite_mps, tenx::idx({1}, {0}));

        assert(gate.op.dimension(0) == gate.op.dimension(1)); // This assumption makes it ok to take the adjoint and transpose below without resizing tmp!
        auto gatemap = tenx::asScalarType<Scalar>(tenx::MatrixMap(gate.op));
        auto mps_map = tenx::MatrixMap(mps, mps.dimension(0), mps.dimension(1) * mps.dimension(2));
        auto tmp_map = tenx::MatrixMap(tmp, tmp.dimension(0), tmp.dimension(1) * tmp.dimension(2));
        switch(gop) { // These will call gemm (hopefully from blas)
            case GateOp::NONE: tmp_map = gatemap * mps_map; break;
            case GateOp::CNJ: tmp_map = gatemap.conjugate() * mps_map; break;
            case GateOp::ADJ: tmp_map = gatemap.adjoint() * mps_map; break;
            case GateOp::TRN: tmp_map = gatemap.transpose() * mps_map; break;
            default: throw except::runtime_error("Unhandled switch case for GateOp: {}", enum2sv(gop));
        }
    }
    if constexpr(settings::verbose_gates)
        tools::log->trace("apply_swap_gate: applied pos {} | idx {} | gm {} | sites {} | svds {}", gate.pos, pos_idxs, enum2sv(gm), sites,
                          svd::solver::get_count());

    // It's best to do an AC-B type of SVD split, so we put the center position on the left-most site when GateMove::ON
    long new_posC = gm == GateMove::ON ? safe_cast<long>(pos_idxs.front()) : old_posC;

    tools::finite::mps::merge_multisite_mps(state, tmp, pos_idxs, new_posC, MergeEvent::GATE, svd_cfg, LogPolicy::SILENT);
    if constexpr(settings::verbose_gates)
        tools::log->trace("apply_swap_gate: merged  pos {} | idx {} | gm {} | sites {} | svds {}", gate.pos, pos_idxs, enum2sv(gm), sites,
                          svd::solver::get_count());

    // Now swap site j-1 in reverse back to i
    for(const auto &r : gate.rwaps) swap_sites(state, r.posL, r.posR, sites, gm);

    // Sanity check
    if constexpr(settings::verbose_gates) {
        tools::log->trace("apply_swap_gate: return  pos {} | gm {} | center {} -> {} | svds {} | labels {} | sites {}", gate.pos, enum2sv(gm), old_posC,
                          new_posC, svd::solver::get_count(), state.get_labels(), sites);
    }
}


template<typename Scalar>
void tools::finite::mps::apply_swap_gates(StateFinite<Scalar> &state, std::vector<qm::SwapGate> &gates, CircuitOp cop, GateMove gm,
                                          std::optional<svd::config> svd_cfg) {
    auto t_swapgate = tid::tic_scope("apply_swap_gates", tid::level::higher);
    if(gates.empty()) return;
    state.clear_cache(); // So that multisite_mps does not use cache
    // Sanity check
    if constexpr(settings::debug_gates) {
        for(const auto &mps : state.mps_sites) mps->assert_normalized();
    }
    if constexpr(settings::verbose_gates) tools::log->trace("before applying swap gates: labels {}", state.get_labels());

    if(gm == GateMove::AUTO) {
        // It only pays to move center point if we are actually using swaps
        // If there are any swaps or rwaps to be made, then switch on moving of center sites.
        gm = GateMove::OFF;
        for(const auto &gate : gates) {
            if(not gate.swaps.empty() or not gate.rwaps.empty()) {
                gm = GateMove::ON;
                break;
            }
        }
    }

    auto                    sites      = num::range<size_t>(0ul, state.template get_length<size_t>(), 1ul);
    [[maybe_unused]] auto   svds_count = svd::solver::get_count();
    [[maybe_unused]] size_t skip_count = 0;
    [[maybe_unused]] size_t swap_count = 0;
    [[maybe_unused]] size_t rwap_count = 0;

    auto   gate_sequence = generate_gate_sequence(state.template get_length<long>(), state.template get_position<long>(), gates, cop);
    GateOp gop           = GateOp::NONE;
    switch(cop) {
        case CircuitOp::NONE: gop = GateOp::NONE; break;
        case CircuitOp::ADJ: gop = GateOp::ADJ; break;
        case CircuitOp::TRN: gop = GateOp::TRN; break;
    }

    for(const auto &[i, gate_idx] : iter::enumerate(gate_sequence)) {
        auto &gate = gates.at(gate_idx);
        if(i + 1 < gate_sequence.size()) skip_count += gate.cancel_rwaps(gates[gate_sequence[i + 1]].swaps);
        swap_count += gate.swaps.size();
        rwap_count += gate.rwaps.size();
        apply_swap_gate(state, gate, gop, sites, gm, svd_cfg);
    }

    move_center_point_to_pos_dir(state, 0, 1, svd_cfg);

    if constexpr(settings::verbose_gates) {
        svds_count = svd::solver::get_count() - svds_count;
        tools::log->debug("apply_swap_gates: applied {} gates | swaps {} | rwaps {} | total {} | skips {} | svds {} | time {:.4f}", gates.size(), swap_count,
                          rwap_count, swap_count + rwap_count, skip_count, svds_count, t_swapgate->get_last_interval());
    }
}


template<typename Scalar>
void tools::finite::mps::apply_swap_gates(StateFinite<Scalar> &state, const std::vector<qm::SwapGate> &gates, CircuitOp cop, GateMove gm,
                                          std::optional<svd::config> svd_cfg) {
    auto t_swapgate = tid::tic_scope("apply_swap_gates", tid::level::higher);
    if(gates.empty()) return;
    state.clear_cache(); // So that multisite_mps does not use cache
    // Sanity check
    if constexpr(settings::debug_gates) {
        for(const auto &mps : state.mps_sites) mps->assert_normalized();
    }
    if constexpr(settings::verbose_gates) tools::log->trace("before applying swap gates: labels {}", state.get_labels());

    if(gm == GateMove::AUTO) {
        // It only pays to move center point if we are actually using swaps
        // If there are any swaps or rwaps to be made, then switch on moving of center sites.
        gm = GateMove::OFF;
        for(const auto &gate : gates) {
            if(not gate.swaps.empty() or not gate.rwaps.empty()) {
                gm = GateMove::ON;
                break;
            }
        }
    }

    auto                    sites      = num::range<size_t>(0ul, state.template get_length<size_t>(), 1ul);
    [[maybe_unused]] auto   svds_count = svd::solver::get_count();
    [[maybe_unused]] size_t skip_count = 0;
    [[maybe_unused]] size_t swap_count = 0;
    [[maybe_unused]] size_t rwap_count = 0;

    auto   gate_sequence = generate_gate_sequence(state.template get_length<long>(), state.template get_position<long>(), gates, cop);
    GateOp gop           = GateOp::NONE;
    switch(cop) {
        case CircuitOp::NONE: gop = GateOp::NONE; break;
        case CircuitOp::ADJ: gop = GateOp::ADJ; break;
        case CircuitOp::TRN: gop = GateOp::TRN; break;
    }

    auto gates_internal = gates; // Make a local copy because gates is const

    for(const auto &[i, gate_idx] : iter::enumerate(gate_sequence)) {
        auto &gate = gates_internal.at(gate_idx);
        if(i + 1 < gate_sequence.size()) skip_count += gate.cancel_rwaps(gates_internal[gate_sequence[i + 1]].swaps);
        swap_count += gate.swaps.size();
        rwap_count += gate.rwaps.size();
        apply_swap_gate(state, gate, gop, sites, gm, svd_cfg);
    }

    move_center_point_to_pos_dir(state, 0, 1, svd_cfg);

    if constexpr(settings::verbose_gates) {
        svds_count = svd::solver::get_count() - svds_count;
        tools::log->debug("apply_swap_gates: applied {} gates | swaps {} | rwaps {} | total {} | skips {} | svds {} | time {:.4f}", gates.size(), swap_count,
                          rwap_count, swap_count + rwap_count, skip_count, svds_count, t_swapgate->get_last_interval());
    }
}
