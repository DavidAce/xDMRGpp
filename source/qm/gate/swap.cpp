//
// Created by david on 2021-09-10.
//
#include "../gate.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/num.h"
#include "tools/common/log.h"
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace settings {
    inline constexpr bool debug_swap   = false;
    inline constexpr bool verbose_swap = false;
}

qm::Swap::Swap(size_t posL, size_t posR) : posL(posL), posR(posR) {
    if(posL + 1 != posR) throw except::logic_error("Swap: Expected posL+1 == posR. Got posL {} and posR {}", posL, posR);
}
qm::Rwap::Rwap(size_t posL, size_t posR) : posL(posL), posR(posR) {
    if(posL + 1 != posR) throw except::logic_error("Rwap: Expected posL+1 == posR. Got posL {} and posR {}", posL, posR);
}

bool qm::Swap::operator==(const Rwap &rwap) const { return posL == rwap.posL and posR == rwap.posR; }

bool qm::Rwap::operator==(const Swap &swap) const { return posL == swap.posL and posR == swap.posR; }

qm::SwapGate qm::SwapGate::exp(cx64 alpha) const {
    if(op_t.size() != 0)
        return qm::SwapGate(op_t, pos, dim, alpha);
    else
        return qm::SwapGate(op, pos, dim, alpha);
}
qm::SwapGate qm::SwapGate::exp(cx128 alpha) const {
    if(op_t.size() != 0)
        return qm::SwapGate(op_t, pos, dim, alpha);
    else
        return qm::SwapGate(op, pos, dim, alpha);
}

void qm::SwapGate::generate_swap_sequences() {
    // Letting i < j, a swap sequence defines the swap operators S(i,i+1) necessary to move distant sites at i,j until they are neighbors at j-1, j,
    // at which point the gate can be applied as a local 2site operator.
    // Similarly, a "rwap" sequence defines the swap operators S(i,i+1) necesary to move local sites j-1,j back to their original position i,j

    if(pos.empty()) return;
    if(pos.size() == 1ul) return;                            // No swaps needed
    if(pos.size() == 2ul and pos[0] + 1ul == pos[1]) return; // No swaps needed
    if(pos.size() >= 3) {
        for(size_t p = 0; p < pos.size() - 1ul; p++)
            if(pos[p] + 1ul != pos[p + 1]) throw except::logic_error("SwapGate: 3body gates are only supported for adjacent sites: pos {}", pos);
        return;
    }
    for(const auto &p : num::range<size_t>(pos.front(), pos.back())) {
        if(p + 1ul == pos.back()) break;
        swaps.emplace_back(p, p + 1ul);
        rwaps.emplace_front(p, p + 1ul);
    }
}

size_t qm::SwapGate::cancel_swaps(std::deque<Rwap> &other_rwaps) {
    // exploit that S(i,j)² == 1 to cancel swaps against rwaps
    size_t count = 0;
    while(true) {
        if(other_rwaps.empty()) return count;
        if(swaps.empty()) return count;
        if(other_rwaps.back() == swaps.front()) {
            if constexpr(settings::verbose_swap) tools::log->trace("Cancel swap S({},{})", swaps.front().posL, swaps.front().posR);
            other_rwaps.pop_back();
            swaps.pop_front();
            count++;
        } else {
            return count;
        }
    }
}
size_t qm::SwapGate::cancel_rwaps(std::deque<Swap> &other_swaps) {
    // exploit that S(i,j)² == 1 to cancel swaps against rwaps
    size_t count = 0;
    while(true) {
        if(other_swaps.empty()) return count;
        if(rwaps.empty()) return count;
        if(other_swaps.front() == rwaps.back()) {
            if constexpr(settings::verbose_swap) tools::log->trace("Cancel rwap S({},{})", rwaps.back().posL, rwaps.back().posR);
            other_swaps.pop_front();
            rwaps.pop_back();
            count++;
        } else {
            return count;
        }
    }
}
