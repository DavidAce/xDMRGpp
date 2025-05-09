#pragma once
#include "number_entropy.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "dimensions.h"
#include "general/iter.h"
#include "math/cast.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/mps.h"
#include <bitset>
#include <fmt/ranges.h>
#include <utility>

using tools::finite::measure::RealScalar;

namespace settings {
    inline constexpr bool debug_numen   = false;
    inline constexpr bool debug_cache   = false;
    inline constexpr bool verbose_numen = false;
    inline constexpr bool verbose_cache = false;
}

enum class From { A, B };

template<From from, auto N, bool on = settings::debug_numen>
[[nodiscard]] inline std::string bits_to_string(const std::bitset<N> &b, [[maybe_unused]] long state_size, long num_bits) {
    if constexpr(on) {
        if(num_bits < 0l or std::cmp_greater_equal(num_bits, N)) {
            tools::log->warn("num_bits should be in range [0,{}]. Got {} | bits {}", N, num_bits, b.to_string());
            return "SEE WARNING";
        }

        // Extract the relevant bits
        auto bs = b.to_string().substr(safe_cast<std::string::size_type>(N) - safe_cast<std::string::size_type>(num_bits));

        if constexpr(from == From::B) {
            // Print in the original order, from N - state_size + num_bits until the end
            return std::string{bs.begin(), bs.end()};
        }
        if constexpr(from == From::A) {
            // Print in reverse order,
            return std::string{bs.rbegin(), bs.rend()};
        }
    } else
        return {};
}

template<auto N>
[[nodiscard]] inline bool bits_are_equal(const std::bitset<N> &b1, const std::bitset<N> &b2, long num_bits) {
    // Example. Let this amplitude be a and the other is o.
    // Consider two bitsets of 8 bits, and we are checking num_bits == 3:
    // Then bits 0,1 and 2 need to be identical
    //     b1 = 00001011
    //     b2 = 00000011
    // Shift bits to the left edge with <<
    //     b1 << (8 - num_bits) = 01100000
    //     b2 << (8 - num_bits) = 01100000
    // Compare with the ^ operator and check that all are zero with .none()
    //     r = (( a << (8 - num_bits) ) ^ (o << (8 - num_bits))) = 00000000
    //     return r.none()
    if constexpr(settings::debug or settings::debug_numen)
        if(num_bits < 0) throw except::runtime_error("num_bits must be in range [0,{}] | Got: {}", N, num_bits);
    const auto num = safe_cast<size_t>(num_bits);
    return ((b1 << (N - num)) ^ (b2 << (N - num))).none();
}

template<typename Scalar, From from, auto N = 64>
struct Amplitude {
    long                     state_size;        // System size
    std::bitset<N>           bits;              // Bits that select spins on each MPS site
    long                     pos = -1l;         // Keeps track of the latest mps pos that has been contracted into ampl
    Eigen::Tensor<Scalar, 1> ampl;              // Accumulates the MPS tensors
    bool                     cache_hit = false; // True if eval was avoided due to cache hit
    Amplitude(long state_size_, const std::bitset<64> &bits_, const Eigen::Tensor<Scalar, 1> &ampl_) : state_size(state_size_), bits(bits_), ampl(ampl_) {
        // In the beginning, no mps site has been contracted into ampl, so pos must be outside the chain 0...L
        if constexpr(from == From::A) pos = -1l;
        if constexpr(from == From::B) pos = state_size;
    }

    public:
    [[nodiscard]] std::string to_string() const {
        if constexpr(from == From::A)
            return bits_to_string<from>(bits, state_size, pos + 1);
        else
            return bits_to_string<from>(bits, state_size, state_size - pos);
    }
    [[nodiscard]] std::string to_string(long num_bits) const { return bits_to_string<from>(bits, state_size, num_bits); }

    [[nodiscard]] bool equal_bitwise(const Amplitude &other, long num_bits) const { return bits_are_equal(bits, other.bits, num_bits); }
    [[nodiscard]] bool equal_bitwise(const std::bitset<N> &other, long num_bits) const { return bits_are_equal(bits, other, num_bits); }
    template<typename CacheT>
    [[nodiscard]] size_t get_idx_from_unsorted_cache(CacheT &cache, long num_bits) {
        // Let's say the we have a bitset 11011100 and we are looking
        // for a cache for tgt_pos = 6. Then we look for
        // 1101110
        // in the cache.
        long num_bits_cache = -1; // How many bits to compare
        for(const auto &[i, c] : iter::enumerate(cache)) {
            if constexpr(from == From::A) num_bits_cache = c.pos + 1;
            if constexpr(from == From::B) num_bits_cache = c.state_size - c.pos;
            if(num_bits_cache != num_bits) continue;
            if(bits_are_equal(this->bits, c.bits, num_bits)) return i;
        }
        return -1ul;
    }
    template<typename CacheT>
    [[nodiscard]] size_t get_root_idx_from_unsorted_cache(CacheT &cache, long tgt_pos) {
        // Let's say the we have a bitset 11011100 and we are looking
        // for a cache to make tgt_pos = 6. Then we look for
        // 1101110, then (i.e. already computed)
        // 110111, then
        // 11011, and so on
        // in the cache.
        if constexpr(from == From::A) {
            for(long p = tgt_pos; p >= 0; --p) {
                long num_bits = p + 1;
                for(const auto &[i, c] : iter::enumerate(cache)) {
                    if(c.pos != p) continue;
                    bool eq = bits_are_equal(this->bits, c.bits, num_bits);
                    //                    if constexpr(settings::debug_cache) {
                    //                        auto tb = this->to_string(num_bits);
                    //                        auto cb = c.to_string(num_bits);
                    //                        tools::log->trace("comparison idx {} | a {} == c {} {} | pos {}", i, tb, cb, eq, p);
                    //                    }
                    if(eq) return i;
                }
            }
        }
        if constexpr(from == From::B) {
            for(long p = tgt_pos; p < state_size; ++p) {
                long num_bits = state_size - p;
                for(const auto &[i, c] : iter::enumerate(cache)) {
                    if(c.pos != p) continue;
                    bool eq = bits_are_equal(this->bits, c.bits, num_bits);
                    //                    if constexpr(settings::debug_cache) {
                    //                        auto tb = this->to_string(num_bits);
                    //                        auto cb = c.to_string(num_bits);
                    //                        tools::log->trace("comparison idx {} | a {} == c {} {} | pos {}", i, tb, cb, eq, p);
                    //                    }
                    if(eq) return i;
                }
            }
        }
        return -1ul;
    }

    void eval(const StateFinite<Scalar> &state, long tgt_pos, std::vector<Amplitude> &cache) {
        if constexpr(from == From::A) {
            if(pos < tgt_pos) {
                auto t_evalA = tid::tic_scope("evalA", tid::level::highest);
                // There are missing mps in the amplitude. Let's add them
                // First, check if it is already available in the amplitudes database
                // We can look it up by transforming the bits between 0 to bit_site to an integer
                // and use that as the index on amplitudes
                long state_pos = state.template get_position<long>();
                if(tgt_pos > state_pos) throw except::logic_error("eval_from_A: expected mps_site ({}) <= state_pos ({})", tgt_pos, state_pos);

                size_t cidx = get_root_idx_from_unsorted_cache(cache, tgt_pos);
                if(cidx != -1ul) {
                    auto &c   = cache[cidx];
                    pos       = c.pos;
                    ampl      = c.ampl;
                    cache_hit = true;
                    if constexpr(settings::verbose_numen or settings::verbose_cache) {
                        tools::log->trace("from A: cache hit: found [pos {:>2} | n {:>2} | bits {}] target [pos {:>2} | n {:>2} | bits {}]", c.pos,
                                          c.bits.count(), c.to_string(), tgt_pos, bits.count(), to_string(tgt_pos + 1));
                    }
                } else if(ampl.size() == 0) {
                    // Initialize
                    ampl.resize(1);
                    ampl.setConstant(1.0);
                    pos       = -1l;
                    cache_hit = false;
                    if constexpr(settings::verbose_numen or settings::verbose_cache)
                        tools::log->trace("from A: cache miss: could not find bits to build target [pos {:>2} | n {:>2} | bits {}]", tgt_pos, bits.count(),
                                          to_string(tgt_pos + 1));
                }

                auto  t_con   = tid::tic_scope("contract", tid::level::highest);
                auto &threads = tenx::threads::get();

                Eigen::Tensor<Scalar, 1> temp;
                // Contract the missing mps up to, but not including, the last mps at mps_site
                for(const auto &mps : state.mps_sites) {
                    long mps_pos = mps->template get_position<long>();
                    if(pos >= mps_pos) continue; // Fast-forward to the missing sites
                    if(tgt_pos < mps_pos) break; // Contract up to the mps at tgt_pos
                    if(ampl.size() != mps->get_chiL())
                        throw except::runtime_error("eval() failed for site {}: mismatch in ampl({}) with size = {} and mps({}) with chiL = {} | bits {}",
                                                    tgt_pos, pos, ampl.size(), mps_pos, mps->get_chiL(), to_string());

                    long                size = mps->get_chiR();
                    std::array<long, 3> off  = {bits[safe_cast<size_t>(mps_pos)], 0, 0}; // This selects which bit gets appended to ampl
                    std::array<long, 3> ext  = {1, mps->get_chiL(), mps->get_chiR()};
                    // ampl never has a trailing Lambda
                    if constexpr(settings::verbose_numen)
                        tools::log->trace("from A: contraction: pos {:>2} | tgt {:>2} | bits {} -> {}", mps_pos, tgt_pos, to_string(), to_string(mps_pos + 1));

                    temp.resize(size);
                    temp.device(*threads->dev) = ampl.contract(mps->get_M_bare().slice(off, ext), tenx::idx({0}, {1})).reshape(std::array<long, 1>{size});
                    ampl                       = std::move(temp); // Update the current amplitude
                    pos                        = mps_pos;         // Update the current site

                    // Add to cache
                    cidx = get_idx_from_unsorted_cache(cache, pos + 1);
                    if(cidx == -1ul) {
                        // Append
                        if constexpr(settings::debug_cache) tools::log->trace("from A: appended to cache: {}", to_string(pos + 1));
                        cache.emplace_back(*this);
                    } else
                        throw except::runtime_error("from A: ampl already in cache at idx {}: {}", cidx, cache[cidx].to_string());
                }
                if(pos != tgt_pos) throw except::logic_error("pos ({}) != tgt_pos ({})", pos, tgt_pos);
            }
        }
        if constexpr(from == From::B) {
            // Start by calculating the mps sites that should be included in the amplitude
            // Remember that in eval_from_B we calculate the amplitude starting from the right-end of the chain
            if(pos > tgt_pos) {
                auto t_evalB = tid::tic_scope("evalB", tid::level::highest);

                // There are missing mps in the amplitude. Let's add them
                // First, check if it is already available in the amplitudes database
                // We can look it up by transforming the bits between 0 to bit_site to an integer
                // and use that as the index on amplitudes
                long state_pos = state.template get_position<long>();
                if(tgt_pos <= state_pos) throw except::logic_error("eval_from_B: expected mps_site ({}) > state_pos ({})", tgt_pos, state_pos);

                size_t cidx = get_root_idx_from_unsorted_cache(cache, tgt_pos);
                if(cidx != -1ul) {
                    auto &c   = cache[cidx];
                    pos       = c.pos;
                    ampl      = c.ampl;
                    cache_hit = true;
                    if constexpr(settings::verbose_numen or settings::verbose_cache) {
                        tools::log->trace("from B: cache hit: found [pos {:>2} | n {:>2} | bits {}] target [pos {:>2} | n {:>2} | bits {}]", c.pos,
                                          c.bits.count(), c.to_string(), tgt_pos, bits.count(), to_string(state_size - tgt_pos));
                    }
                } else if(ampl.size() == 0) {
                    // Initialize
                    ampl.resize(1);
                    ampl.setConstant(1.0);
                    pos       = state_size;
                    cache_hit = false;
                    if constexpr(settings::verbose_numen or settings::verbose_cache) {
                        tools::log->trace("from B: cache miss: could not find bits to build target [pos {:>2} | n {:>2} | bits {}]", tgt_pos, bits.count(),
                                          to_string(state_size - tgt_pos));
                        for(const auto &[i, c] : iter::enumerate(cache)) tools::log->trace("  {}: {}", i, c.to_string());
                    }
                }

                auto                     t_con   = tid::tic_scope("contract", tid::level::highest);
                auto                    &threads = tenx::threads::get();
                Eigen::Tensor<Scalar, 1> temp;
                // Contract the missing mps
                for(const auto &mps : iter::reverse(state.mps_sites)) {
                    long mps_pos = mps->template get_position<long>();
                    if(pos <= mps_pos) continue; // Fast-forward to the missing sites
                    if(tgt_pos > mps_pos) break; // Contract up to the mps at mps_pos
                    if(ampl.size() != mps->get_chiR())
                        throw except::runtime_error("eval() failed for site {}: mismatch in ampl({}) with size = {} and mps({}) with chiR = {} | bits {}",
                                                    tgt_pos, pos, ampl.size(), mps_pos, mps->get_chiL(), to_string());

                    long                mps_rpos = state_size - 1 - mps_pos;
                    long                size     = mps->get_chiL();
                    std::array<long, 3> off      = {bits[safe_cast<size_t>(mps_rpos)], 0, 0}; // This selects which bit gets prepended to ampl
                    std::array<long, 3> ext      = {1, mps->get_chiL(), mps->get_chiR()};

                    // ampl never has a trailing Lambda
                    if constexpr(settings::verbose_numen)
                        tools::log->trace("from B: contraction: pos  {:>2} | tgt {:>2} | bits {} <- {}", mps_pos, tgt_pos, to_string(state_size - mps_pos),
                                          to_string());
                    temp.resize(size);
                    temp.device(*threads->dev) = mps->get_M_bare().slice(off, ext).contract(ampl, tenx::idx({2}, {0})).reshape(std::array<long, 1>{size});
                    ampl                       = std::move(temp); // Update the current amplitude
                    pos                        = mps_pos;         // Update the current site

                    // Add to cache
                    cidx = get_idx_from_unsorted_cache(cache, state_size - pos);
                    if(cidx == -1ul) {
                        // Append
                        if constexpr(settings::debug_cache) tools::log->trace("from B: added to cache: {}", to_string());
                        cache.emplace_back(*this);
                    } else
                        throw except::runtime_error("from B: ampl already in cache at idx {}: {}", cidx, cache[cidx].to_string());
                }
                if(pos != tgt_pos) throw except::logic_error("site ({}) != mps_site ({})", pos, tgt_pos);
            }
        }
    }
};

inline size_t nextGreaterWithSameSetBit(size_t n) {
    // http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    //     size_t t = n | (n - 1); // t gets v's least significant 0 bits set to 1
    //  Next set to 1 the most significant bit to change,
    //  set to 0 the least significant ones, and add the necessary 1 bits.
    //     return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(n) + 1));

    // The __builtin_ctz(v) GNU C compiler intrinsic for x86 CPUs returns the number of trailing zeros. If you are using Microsoft compilers for x86, the
    // intrinsic is _BitScanForward. These both emit a bsf instruction, but equivalents may be available for other architectures. If not, then consider using
    // one of the methods for counting the consecutive zero bits mentioned earlier.
    //
    // Here is another version that tends to be slower because of its division operator, but it does not require counting the trailing zeros.

    if(n == 0ul) return -1ul;
    size_t t = (n | (n - 1)) + 1;
    return t | ((((t & -t) / (n & -n)) >> 1) - 1);
}

// function to find the position of rightmost set bit. Returns -1 if there are no set bits
inline size_t getFirstSetBitPos(size_t n) { return safe_cast<size_t>((std::log2(n & -n) + 1) - 1); }

// function to find the next greater integer
inline size_t nextGreaterWithOneMoreSetBit(size_t n) {
    // position of rightmost unset bit of n by passing ~n as argument
    size_t pos = getFirstSetBitPos(~n);
    // if n consists of unset bits, then set the rightmost unset bit
    if(pos > -1ul) return (1 << pos) | n;
    // n does not consists of unset bits
    return ((n << 1) + 1);
}

inline std::vector<size_t> get_numbers_with_hamming_weight_n(size_t n, long num_bits) {
    std::vector<size_t> numbers;
    auto                max_num = std::uint64_t(1) << num_bits;
    for(size_t num = 0; num < max_num; ++num) {
        if(std::bitset<64>(num).count() == n) numbers.emplace_back(num);
    }
    return numbers;
}

inline std::vector<std::vector<size_t>> get_popcount_partitions(size_t num_bits) {
    // We make a vector of vectors where each inner vector has numbers with equal popcount,
    // and popcount is the number of set bits.
    size_t num_part = 0;
    for(size_t i = 0; i <= num_bits; i++) num_part++;
    std::vector<std::vector<size_t>> partitions(num_part);
    auto                             max_num = 1ul << num_bits;
    for(size_t num = 0; num < max_num; ++num) {
        if constexpr(settings::debug or settings::debug_numen)
            partitions.at(std::bitset<64>(num).count()).emplace_back(num);
        else
            partitions[std::bitset<64>(num).count()].emplace_back(num);
    }
    return partitions;
}

inline std::vector<size_t> get_random_roundrobin_popcount_vector(size_t num_bits) {
    auto popcount_partitions = get_popcount_partitions(num_bits);
    // Shuffle each partition
    for(auto &p : popcount_partitions) std::shuffle(p.begin(), p.end(), rnd::internal::rng64);

    // Make a new list where we pick the tails of each partition in round-robin fashion
    std::vector<size_t> rrp;
    size_t              size = 1ul << num_bits;
    rrp.reserve(size);
    while(true) {
        for(auto &p : popcount_partitions) {
            if(not p.empty()) {
                rrp.emplace_back(p.back());
                p.pop_back();
            }
        }
        bool empty = std::all_of(popcount_partitions.begin(), popcount_partitions.end(), [](const auto &p) -> bool { return p.empty(); });
        if(empty) break;
    }
    return rrp;
}

template<typename AmplitudesT>
size_t amplitude_next_idx_round_robin(size_t aidx, size_t &nbit, size_t nmax, std::vector<bool> &nflg, std::vector<size_t> &namp,
                                      const AmplitudesT &amplitudes) {
    if(namp.size() != amplitudes.size()) throw except::logic_error("Size mistmatch: namp {} != amplitudes {}", namp.size(), amplitudes.size());
    if(nflg.size() != nmax + 1) throw except::logic_error("nflg.size() != nmax+1");
    auto t_next = tid::tic_token("next", tid::level::highest);
    // aidx is the current index in abit
    // nbit is the number of bits at this index, i.e. nbit = abit[aidx].
    if(nbit == namp[aidx]) {
        namp[aidx] = -1ul; // Mark as used
        return aidx;       // The current index hasn't been checked, so return it
    }

    auto next_nbit = [&](size_t n) -> std::pair<size_t, bool> {
        if(nflg.size() != nmax + 1) throw except::logic_error("nflg.size() != nmax+1");
        bool found_all_nbit = std::all_of(nflg.begin(), nflg.end(), [](bool b) -> bool { return b; });
        if(found_all_nbit) return {-1ul, found_all_nbit};
        while(true) {
            n = num::pbc<size_t>(n + 1, nmax + 1); // Advance
            if(not nflg[n]) break;                 // Keep if it's still valid
        }
        return {n, found_all_nbit};
    };

    // The goal now is to find the next index aidx in abit which has nbit+1
    size_t                  counter        = 0;
    [[maybe_unused]] size_t wcount         = 0;
    bool                    found_all_nbit = false;
    std::tie(nbit, found_all_nbit)         = next_nbit(nbit);

    while(not found_all_nbit) {
        wcount++;
        if(nbit == namp[aidx]) {
            namp[aidx] = -1ul; // Mark as used
            counter    = 0;    // Reset counter when we find a match
                               //            tools::log->info("aidx {} | nbit {} | wcount: {}",aidx, nbit, wcount);
            return aidx;       // Return aidx. This index points to the next amplitude that has nbit increased by one.
        }

        if(namp[aidx] == -1ul and nbit == std::bitset<64>(aidx).count()) {
            // aidx has the correct nbit but this particular aidx is already taken.
            // We can fast-forward to the next aidx with the same nbit and try that
            auto aidx_new = nextGreaterWithSameSetBit(aidx);
            if(aidx_new < namp.size()) {
                counter += aidx_new - aidx;
                aidx = aidx_new;
                continue; // Try this aidx instead
            }
        }
        aidx = num::pbc<size_t>(aidx + 1, namp.size()); // Cycle through the whole list of amplitudes

        if(counter++ >= namp.size()) {
            // We haven't been able to find a matching index, which means that all indices with the current nbit are taken.
            // Simply go to the next nbit and reset the counter
            counter                        = 0;
            nflg[nbit]                     = true; // Flag this nbit as used up
            auto nbit_old                  = nbit;
            std::tie(nbit, found_all_nbit) = next_nbit(nbit);
            if(not found_all_nbit and nbit > nbit_old) {
                // Fastforward to the next greater with one nbit increased appropriately
                auto aidx_new = aidx;
                for(size_t i = 0; i < nbit - nbit_old; i++) aidx_new = nextGreaterWithOneMoreSetBit(aidx_new);
                if(aidx_new < namp.size()) aidx = aidx_new; // Take it if it's in range. Otherwise ignore.
            }
        }
    }
    return -1ul;
}

enum class Side { LEFT, RIGHT };

template<typename Scalar, Side side, typename AmplitudesT, typename CacheT>
std::vector<RealScalar<Scalar>> compute_probability_rrp(const StateFinite<Scalar> &state, long tgt_pos, AmplitudesT &amplitudes, CacheT &cache) {
    // Here we compute the probability of finding

    auto                            t_prob    = tid::tic_scope("probability", tid::level::highest);
    auto                            state_pos = state.template get_position<long>();
    auto                            state_len = state.template get_length<long>();
    std::vector<RealScalar<Scalar>> probability(safe_cast<size_t>(state_len + 1), 0.0);
    auto                            probability_sum = RealScalar<Scalar>{0};
    // Figure out which schmidt values to use
    auto                                 t_figout = tid::tic_scope("figout", tid::level::highest);
    Eigen::Tensor<RealScalar<Scalar>, 1> schmidt_values;
    if(tgt_pos < state_pos)
        schmidt_values = state.get_mps_site(tgt_pos + 1).get_L().abs(); // A-site
    else if(tgt_pos == state_pos)
        schmidt_values = state.get_mps_site(tgt_pos).get_LC().abs(); // AC-site
    else {
        const auto &mps_left = state.get_mps_site(tgt_pos - 1);
        schmidt_values       = mps_left.isCenter() ? mps_left.get_LC().abs() : mps_left.get_L().abs(); // B-site
    }
    t_figout.toc();

    // Create optional slots for each schmidt value
    auto t_slots                     = tid::tic_scope("slots", tid::level::highest);
    auto amplitude_cutoff            = RealScalar<Scalar>{1e-4f};
    using MatrixReal                 = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal                 = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, 1>;
    MatrixReal      ampacc_sq_matrix = MatrixReal::Zero(schmidt_values.size(), state_len + 1);
    Eigen::VectorXi schmidt_taken    = Eigen::VectorXi::Zero(schmidt_values.size());
    VectorReal      schmidt_squared  = tenx::VectorMap(schmidt_values).cwiseAbs2();
    auto            min_alpha        = 0l;
    auto            max_alpha        = schmidt_values.size();

    for(auto &&[a_idx, a] : iter::enumerate(amplitudes)) {
        auto n  = a.bits.count();
        long nl = safe_cast<long>(n); // Number of bits in amplitude a, as <long>
        // Evaluate the amplitude vector (this updates its internal position)
        a.eval(state, tgt_pos, cache);
        if(a.ampl.size() != schmidt_values.size()) {
            tools::log->dump_backtrace();
            throw except::logic_error("Mismatching size ampl {} != {}", a.ampl.size(), schmidt_values.size());
        }
        // Add amplitudes to the nth column
        auto avec = tenx::VectorMap(a.ampl);
        ampacc_sq_matrix.col(nl) += avec.conjugate().cwiseProduct(avec).cwiseAbs();
        //        tools::log->info("ampacc_sq: \n{}\n", linalg::matrix::to_string(ampacc_sq_matrix,8));

        // Check if any amplitude element gives the signal to add probability
        for(long alpha = min_alpha; alpha < max_alpha; alpha++) {
            if(schmidt_taken(alpha) == 1) continue;
            auto asq = ampacc_sq_matrix(alpha, nl); // The a²[alpha] value tells us to pick the corresponding λ²[alpha] when nonzero.
            auto ssq = schmidt_squared[alpha];      // The value λ² is added to probability if the amplitude is greater than cutoff.
            // Check that the probability would not grow too large, in case we are erroneously considering an amplitude
            // This is important when we work with a small cutoff, where sometimes numerical noise is mistaken for a signal.
            bool accept = asq > amplitude_cutoff and probability_sum + ssq <= RealScalar<Scalar>{1} + RealScalar<Scalar>{1e-8f};
            if(accept) {
                if constexpr(side == Side::LEFT) {
                    if(state_pos < tgt_pos) {
                        // The convention is that p_i(n) is the probability of having n particles to the LEFT of tgt_pos i.
                        // If we are calculating from the right (using B's) we have to convert to the left convention.
                        // In that case, ssq is a probability for having n particles to the right of tgt_pos.
                        // If we have N particles in total, this is the same as the probability of having N-n particles to the left.
                        // Since we calculate from the right side whenever tgt_pos > state_pos, we can take the complementary number of particles
                        auto nc = state.popcount - n;
                        probability.at(nc) += ssq;
                        probability_sum += ssq;
                    } else {
                        probability[n] += ssq;
                        probability_sum += ssq;
                    }
                } else if constexpr(side == Side::RIGHT) {
                    if(tgt_pos < state_pos) {
                        // The convention is that p_i(n) is the probability of having n particles to the RIGHT of tgt_pos i.
                        // If we are calculating from the left (using A's) we have to convert to the right convention.
                        // In that case, ssq is a probability for having n particles to the left of tgt_pos.
                        // If we have N particles in total, this is the same as the probability of having N-n particles to the right.
                        // Since we calculate from the left side whenever tgt_pos < state_pos, we can take the complementary number of particles
                        auto nc = state.popcount - n;
                        probability.at(nc) += ssq;
                        probability_sum += ssq;
                    } else {
                        probability[n] += ssq;
                        probability_sum += ssq;
                    }
                }

                //                probability[n] += ssq;
                //                probability_sum += ssq;
                schmidt_taken(alpha) = 1;
                if(schmidt_taken.isOnes()) break;
            }
            if constexpr(settings::verbose_numen) {
                std::string_view accept_str = accept ? "accept" : "";
                std::string_view cacheh_str = a.cache_hit ? "cache" : "";
                tools::log->info("pos {:>2} | n {:>2} | bits {} | 1-P {:10.3e} | a({:>4})² {:9.3e} (cut {:8.2e}) | λ({:>4})² "
                                 "{:9.3e} [{:6}|{:5}]",
                                 tgt_pos, n, a.to_string(), fp(1 - probability_sum), alpha, asq, fp(amplitude_cutoff), alpha, ssq, accept_str, cacheh_str);
            }
        }
        if(schmidt_taken.isOnes()) break;                                                    // All schmidt values squared have been added to probability
        while(schmidt_taken(min_alpha) == 1) min_alpha++;                                    // Advance min_alpha to skip first taken schmidt values
        while(schmidt_taken(max_alpha - 1) == 1 and max_alpha >= min_alpha + 1) max_alpha--; // Decrease max_alpha to skip the last taken schmidt values
    }
    // Sanity check on probabilities
    auto p_sum = std::accumulate(probability.begin(), probability.end(), RealScalar<Scalar>{0});
    tools::log->trace("p(n)[{:2}] = {:18.16f}", tgt_pos, fp(p_sum));
    if(std::abs(p_sum - RealScalar<Scalar>{1}) > RealScalar<Scalar>{1e-4f}) {
        tools::log->dump_backtrace();
        tools::log->info("p(n)[{:>2}] = {:18.16f}", tgt_pos, fp(p_sum));
        throw except::runtime_error("p_sum - 1.0 = {:.8e}", fp(p_sum - RealScalar<Scalar>{1}));
    }
    if(std::abs(p_sum - RealScalar<Scalar>{1}) > RealScalar<Scalar>{1e-8f}) {
        tools::log->dump_backtrace();
        tools::log->info("p(n)[{:>2}] = {:18.16f}", tgt_pos, fp(p_sum));
        tools::log->warn("p_sum - 1.0 = {:.8e}", fp(p_sum - RealScalar<Scalar>{1}));
    }
    return probability;
}

template<typename Scalar, From from>
std::vector<Amplitude<Scalar, from>> generate_amplitude_list_rrp(long state_len, long mps_pos) {
    // Generate a list of bit sequences of size prod_{i=0}^pos spin_dim_i.
    // For spin-half this is just 2^pos elements, where pos is the mps position
    // counting from the left.
    // Example: Let state_pos == 4 and state_len == 8. Then
    // num_bitseqs = prod_{i=0}^{mpo_pos} = spin_dim_0 * spin_dim_1 * spin_dim_2 = 2³ = 8,
    //      if mpo_pos == 2, "from A",  we generate
    //          000, 001, 010, 011, 100, 101, 110, 111
    //      if mpo_pos == 6 "from B" we also generate
    //          000, 001, 010, 011, 100, 101, 110, 111
    // So we generate the same regardless, but in the first case the numbers are interpreted
    // in "in reverse" so that 001 becomes 100.

    // In this rrp version, we sort the bit sequences using a random round-robin popcount vector,
    // so in the example above we could get
    //    000, 001, 011, 111, 010, 101, 100, 110
    // notice how the bits with 1 or 2 bits end up in random order
    if(state_len < 0) throw except::logic_error("Expected a non-negative state_len. Got: {}", state_len);
    if(mps_pos < 0) throw except::logic_error("Expected a non-negative mps_pos. Got: {}", mps_pos);
    if(state_len <= mps_pos) throw except::logic_error("Expected a state_len <= mps_pos. Got: state_len={}, mps_pos={}", state_len, mps_pos);
    auto t_amp    = tid::tic_scope("amplitude", tid::level::highest);
    auto num_bits = -1ul;
    if constexpr(from == From::A) { num_bits = safe_cast<size_t>(mps_pos + 1l); }
    if constexpr(from == From::B) { num_bits = safe_cast<size_t>(state_len - mps_pos); }
    auto                                 rrp = get_random_roundrobin_popcount_vector(num_bits);
    std::vector<Amplitude<Scalar, from>> amplitudes;
    amplitudes.reserve(rrp.size());
    //    long start_pos = from == From::A ? -1l : state_len;
    for(const auto &num : rrp) amplitudes.emplace_back(Amplitude<Scalar, from>{state_len, std::bitset<64>(safe_cast<unsigned long long int>(num)), {}});
    return amplitudes;
}

/*! \brief Compute the number entropies at all cuts of the 1d-chain
 *
 * In fLBIT simulations we have particle-number conservation.
 * The number of particles is invariant to both time evolution of the lbit state,
 * as well as transforming to real-basis with the unitary circuit.
 *
 * The number entropy is defined as
 *
 *      S_N = - Σ_n p(n) ln p(n)
 *
 * where p(n) is the probability of finding n particles in a subsystem l.
 * At a given bond, we calculate p(n) for the left/right subsystem
 * from schmidt values corresponding to nonzero amplitudes A(n)_𝛼:
 *      p(n) =  Σ_[𝛼|A({𝜎})_𝛼 > ϵ] (λ_α)²,
 *
 * where:
 *      * {𝜎} is a particle configuration on the 1d chain such as 11000 or 00101
 *      * n is the number of particles in {𝜎},  Σ_i 𝜎_i.  (e.g. both 11000 and 00101 would give n = 2)
 *      * A({𝜎}) is the amplitude vector identified by the configuration {𝜎}. There are 2^l per bond.
 *      * λ are the schmidt values of the bond at the bipartition. There is only one λ per bond.
 *      * 𝛼 indexes both the schmidt values λ and the amplitude vector and A({𝜎})
 *      * ϵ is a small cutoff, typically ~ 1e-14
 *      * subscript _[𝛼|A({𝜎})_𝛼 > ϵ] means that the sum only includes
 *        𝛼 for which the corresponding amplitude element is large enough, signaling that
 *        the probability of finding n particles in the subsystem is nonzero.
 *
 *
 * Example: Consider a bipartition in the middle of 8 particles.
 *          In general, an MPS describing such a subsystem has the form
 *
 *            [0]--[1]--[2]--[3]--A({𝜎})_α
 *             |    |    |    |
 *            𝜎0   𝜎1   𝜎2   𝜎3
 *
 *          where 𝜎i = [0,1] sets an lbit off/on on that site,
 *          and 𝛼 indexes both the A({𝜎}) and schmidt values λ.
 *          Observe that:
 *              * Each combination of 𝜎i's such as {𝜎}=01101 yields a unique amplitude vector A(01101)
 *              * We expect A({𝜎}) and A({𝜎'}) with the same n to have nonzero elements at the same α
 *
 *
 */
template<typename Scalar>
std::vector<RealScalar<Scalar>> tools::finite::measure::number_entropies(const StateFinite<Scalar> &state) {
    if(state.measurements.number_entropies) return state.measurements.number_entropies.value();
    if(state.get_algorithm() != AlgorithmType::fLBIT) {
        // Only fLBIT has particle-number conservation
        tools::log->warn("Called number_entropies(StateFinite) from algorithm [{}]: This is currently only valid with the [fLBIT] algorithm",
                         enum2sv(state.get_algorithm()));
        return std::vector<RealScalar<Scalar>>(state.template get_length<size_t>(), 0.0);
    }
    if(state.popcount == -1ul or state.popcount > state.get_length()) {
        tools::log->error("Canceled number entropy calculation because state.popcount == {}", state.popcount);
        return std::vector<RealScalar<Scalar>>(state.template get_length<size_t>(), 0.0);
    }

    auto t_num      = tid::tic_scope("number_entropy", tid::level::normal);
    auto state_copy = state; // Make a local copy, so we can move it to the middle without touching the original state
    tools::finite::mps::move_center_point_to_middle(state_copy);
    //    tools::finite::mps::move_center_point_to_pos(state_copy, state_copy.get_length<long>() - 1);
    //    tools::finite::mps::move_center_point_to_pos(state_copy,state_copy.get_length<long>()-1);
    auto state_pos       = state_copy.template get_position<long>();
    auto state_len       = state_copy.template get_length();
    auto state_llen      = state_copy.template get_length<long>();
    auto von_neumann_sum = [](RealScalar<Scalar> sum, const RealScalar<Scalar> p) { return p > 0 ? sum + p * std::log(p) : sum; };

    std::vector<RealScalar<Scalar>>      number_entropies(state_len + 1, 0.0); // Collects the resulting number entropies
    [[maybe_unused]] size_t              cacheA_size;
    [[maybe_unused]] size_t              cacheB_size;
    Eigen::Tensor<RealScalar<Scalar>, 2> probabilities(state_llen + 1, state_llen + 1);
    probabilities.setZero();
    // The first and last probabilities [0] and [L+1]
    tools::log->enable_backtrace(200);
    {
        std::vector<Amplitude<Scalar, From::A>> cache;
        for(const auto &mps : state_copy.mps_sites) {
            auto pos = mps->template get_position<long>();
            auto idx = safe_cast<size_t>(pos) + 1; // First [0] and last [L+1] number entropy are zero. Then mps[0] generates number entropy idx 1, and so on.
            if(pos > state_pos) break;             // Only compute up to and including AC
            if(mps->get_label() == "B") throw except::logic_error("Expected A/AC site, got B");
            auto amplitudes                     = generate_amplitude_list_rrp<Scalar, From::A>(state_llen, pos);
            auto probability                    = compute_probability_rrp<Scalar, Side::LEFT>(state_copy, pos, amplitudes, cache);
            auto number_entropy                 = -std::accumulate(probability.begin(), probability.end(), RealScalar<Scalar>{0}, von_neumann_sum);
            number_entropies[idx]               = std::abs(number_entropy);
            auto                psize           = safe_cast<long>(probability.size());
            std::array<long, 2> offset          = {0, pos + 1};
            std::array<long, 2> extent          = {psize, 1};
            probabilities.slice(offset, extent) = Eigen::TensorMap<Eigen::Tensor<RealScalar<Scalar>, 2>>(probability.data(), psize, 1);
        }
        cacheA_size = cache.size();
    }
    {
        std::vector<Amplitude<Scalar, From::B>> cache;
        for(const auto &mps : iter::reverse(state_copy.mps_sites)) { // Now compute from the right edge until the middle
            auto pos = mps->template get_position<long>();
            auto idx = safe_cast<size_t>(pos); // First [0] and last [L+1] number entropy are zero. Then mps[L] generates number entropy idx L, and so on.
            if(pos <= state_pos + 1) break;    // +1 because we don't need to compute AC again
            if(mps->get_label() != "B") throw except::logic_error("Expected B site, got {}", mps->get_label());
            auto amplitudes                     = generate_amplitude_list_rrp<Scalar, From::B>(state_llen, pos);
            auto probability                    = compute_probability_rrp<Scalar, Side::LEFT>(state_copy, pos, amplitudes, cache);
            auto number_entropy                 = -std::accumulate(probability.begin(), probability.end(), RealScalar<Scalar>{0}, von_neumann_sum);
            number_entropies[idx]               = std::abs(number_entropy);
            auto                psize           = safe_cast<long>(probability.size());
            std::array<long, 2> offset          = {0, pos};
            std::array<long, 2> extent          = {psize, 1};
            probabilities.slice(offset, extent) = Eigen::TensorMap<Eigen::Tensor<RealScalar<Scalar>, 2>>(probability.data(), psize, 1);
        }
        cacheB_size = cache.size();
    }
    tools::log->disable_backtrace();
    state.measurements.number_entropies        = number_entropies;
    state.measurements.number_entropy_midchain = number_entropies.at(state.template get_length<size_t>() / 2);
    state.measurements.number_entropy_current  = number_entropies.at(state.template get_position<size_t>() + 1);
    state.measurements.number_probabilities    = probabilities;
    tools::log->debug("Number entropies: cchA {} + cchB {} = {} | χ = {} | time {:.3e} s", cacheA_size, cacheB_size, cacheA_size + cacheB_size,
                      measure::bond_dimension_midchain(state), t_num->get_last_interval());
    // tools::log->debug("Number entropies: {::.4f}", number_entropies);
    return state.measurements.number_entropies.value();
    /* clang-format off */
    /*
    Full chain number entropy calculation
    RRP    L   χ      Cache size      Time       Speedup
           8   15     22              0.0004 s
    *      8   15     25              0.0007 s   0.57x

           16  117    269             0.025  s
    *      16  117    154             0.013  s   1.92x

           24  829    5927            6.6    s
    *      24  829    655             2.0    s   3.3x

           32  423    117953          128.9  s
    *      32  423    2325            1.9    s   68x
    */
    /* clang-format on */
}


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::number_entropy_current(const StateFinite<Scalar> &state) {
    if(state.measurements.number_entropy_current) return state.measurements.number_entropy_current.value();
    if(state.measurements.number_entropies) {
        state.measurements.number_entropy_current = state.measurements.number_entropies->at(state.template get_position<size_t>() + 1);
        return state.measurements.number_entropy_current.value();
    }
    state.measurements.number_entropies = measure::number_entropies(state);
    if(not state.measurements.number_entropies->empty()) {
        state.measurements.number_entropy_current = state.measurements.number_entropies->at(state.template get_position<size_t>() + 1);
        return state.measurements.number_entropy_current.value();
    } else
        return 0;
}


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::number_entropy_midchain(const StateFinite<Scalar> &state) {
    if(state.measurements.number_entropy_midchain) return state.measurements.number_entropy_midchain.value();
    if(state.measurements.number_entropies) {
        state.measurements.number_entropy_midchain = state.measurements.number_entropies->at(state.get_length() / 2);
        return state.measurements.number_entropy_midchain.value();
    }
    state.measurements.number_entropies = measure::number_entropies(state);
    if(not state.measurements.number_entropies->empty()) {
        state.measurements.number_entropy_midchain = state.measurements.number_entropies->at(state.get_length() / 2);
        return state.measurements.number_entropy_midchain.value();
    } else
        return 0;
}


// template<typename AmplitudesT, typename CacheT>
// std::vector<double> compute_probability(const StateFinite<Scalar> &state, long tgt_pos, AmplitudesT &amplitudes, CacheT &cache) {
//     // Here we compute the probability of finding
//
//     auto t_prob    = tid::tic_scope("probability");
//     auto state_pos = state.get_position<long>();
//     auto state_len = state.get_length<long>();
//     //    auto                tgt_rpos  = state_len - 1 - tgt_pos;
//     //    auto                prob_size = tgt_pos > state_pos ? tgt_rpos + 2 : tgt_pos + 2;
//     auto                prob_size = tgt_pos + 2;
//     std::vector<double> probability(safe_cast<size_t>(prob_size), 0.0);
//     double              probability_sum = 0.0;
//     // Figure out which schmidt values to use
//     auto                     t_figout = tid::tic_scope("figout");
//     Eigen::Tensor<double, 1> schmidt_values;
//     if(tgt_pos < state_pos)
//         schmidt_values = state.get_mps_site(tgt_pos + 1).get_L().abs(); // A-site
//     else if(tgt_pos == state_pos)
//         schmidt_values = state.get_mps_site(tgt_pos).get_LC().abs(); // AC-site
//     else {
//         const auto &mps_left = state.get_mps_site(tgt_pos - 1);
//         schmidt_values       = mps_left.isCenter() ? mps_left.get_LC().abs() : mps_left.get_L().abs(); // B-site
//     }
//     t_figout.toc();
//
//     // Create optional slots for each schmidt value
//     auto                t_slots          = tid::tic_scope("slots");
//     double              amplitude_cutoff = 1e-12;
//     std::vector<size_t> namp; // Number of bits in each amplitude
//     namp.reserve(amplitudes.size());
//     for(const auto &a : amplitudes) namp.emplace_back(a.bits.count());
//
//     auto              nmax = safe_cast<size_t>(tgt_pos) + 1ul; // Maximum number of n
//     std::vector<bool> nflg(nmax + 1, false);                     // Flags for each n to tell if they have all been found in amplitudes
//
//     Eigen::MatrixXd ampacc_sq_matrix = Eigen::MatrixXd::Zero(schmidt_values.size(), tgt_pos + 2);
//     Eigen::VectorXi schmidt_taken    = Eigen::VectorXi::Zero(schmidt_values.size());
//     Eigen::VectorXd schmidt_squared  = tenx::VectorMap(schmidt_values).cwiseAbs2();
//     auto            idx              = 0ul; // Start amplitude index
//     auto            n                = 0ul; // Number of bits in the current amplitude
//     auto            min_alpha        = 0l;
//     auto            max_alpha        = schmidt_values.size();
//
//     while(true) {
//         idx = amplitude_next_idx_round_robin(idx, n, nmax, nflg, namp, amplitudes);
//         if(idx == -1ul) // Could not find next idx. Probably all have been checked.
//             break;
//         auto &a  = amplitudes[idx];
//         long  nl = safe_cast<long>(a.bits.count()); // Number of bits in amplitude a, as <long>
//         if constexpr(settings::debug_numen)
//             if(safe_cast<long>(n) != nl) throw except::logic_error("Wrong bit number!");
//         // Evaluate the amplitude vector
//         a.eval(state, tgt_pos, cache);
//         if(a.ampl.size() != schmidt_values.size()) {
//             tools::log->dump_backtrace();
//             throw except::logic_error("Mismatching size ampl {} != {}", a.ampl.size(), schmidt_values.size());
//         }
//         // Add amplitudes to the nth column
//         auto avec = tenx::VectorMap(a.ampl);
//         ampacc_sq_matrix.col(nl) += avec.conjugate().cwiseProduct(avec).cwiseAbs();
//         //        tools::log->info("ampacc_sq: \n{}\n", linalg::matrix::to_string(ampacc_sq_matrix,8));
//
//         // Check if any amplitude element gives the signal to add probability
//         for(long alpha = min_alpha; alpha < max_alpha; alpha++) {
//             if(schmidt_taken(alpha) == 1) continue;
//             auto asq = ampacc_sq_matrix(alpha, nl); // The a²[alpha] value tells us to pick the corresponding λ²[alpha] when nonzero.
//             auto ssq = schmidt_squared[alpha];      // The value λ² is added to probability if the amplitude is greater than cutoff.
//             // Check that the probability would not grow too large, in case we are erroneously considering an amplitude
//             // This is important when we work with a small cutoff, where sometimes numerical noise is mistaken for a signal.
//             bool accept = asq > amplitude_cutoff and probability_sum + ssq <= 1.0 + 1e-8;
//             if(accept) {
//                 if(state_pos < tgt_pos) {
//                     // The convention for probabilities is for having n particles to the left of tgt_pos.
//                     // Since we calculate from the right side whenever tgt_pos > state_pos, we can take the complementary number of particles
//                     auto nc = state.popcount - n;
//                     probability.at(nc) += ssq;
//                     probability_sum += ssq;
//                 } else {
//                     probability[n] += ssq;
//                     probability_sum += ssq;
//                 }
//                 //                probability[n] += ssq;
//                 //                probability_sum += ssq;
//                 schmidt_taken(alpha) = 1;
//                 if(schmidt_taken.isOnes()) break;
//             }
//             if constexpr(settings::verbose_numen) {
//                 std::string_view accept_str = accept ? "accept" : "";
//                 std::string_view cacheh_str = a.cache_hit ? "cache" : "";
//                 tools::log->trace("pos {:>2} | n {:>2} | bits {} | idx {} | 1-P {:10.3e} | a({:>4})² {:9.3e} (cut {:8.2e}) | λ({:>4})² "
//                                   "{:9.3e} [{:6}|{:5}]",
//                                   tgt_pos, n, a.to_string(), idx, 1 - probability_sum, alpha, asq, amplitude_cutoff, alpha, ssq, accept_str, cacheh_str);
//             }
//         }
//         if(schmidt_taken.isOnes()) break;                                                    // All schmidt values squared have been added to probability
//         while(schmidt_taken(min_alpha) == 1) min_alpha++;                                    // Advance min_alpha to skip first taken schmidt values
//         while(schmidt_taken(max_alpha - 1) == 1 and max_alpha >= min_alpha + 1) max_alpha--; // Decrease max_alpha to skip the last taken schmidt values
//     }
//
//     // Sanity check on probabilities
//     auto p_sum = std::accumulate(probability.begin(), probability.end(), 0.0);
//     if(std::abs(p_sum - 1.0) > 1e-4) {
//         tools::log->dump_backtrace();
//         tools::log->info("p(n) = {::18.16f} = {:18.16f}", probability, p_sum);
//         throw except::runtime_error("p_sum - 1.0 = {:.8e}", p_sum - 1.0);
//     }
//     if(std::abs(p_sum - 1.0) > 1e-8) {
//         tools::log->dump_backtrace();
//         tools::log->warn("p(n) = {::18.16f} = {:18.16f}", probability, p_sum);
//         tools::log->warn("p_sum - 1.0 = {:.8e}", p_sum - 1.0);
//     }
//     //    tools::log->trace("p(n) = {:18.16f} = {:18.16f}", fmt::join(probability, ", "), p_sum);
//     return probability;
// }

// template<From from>
// std::vector<Amplitude<from>> generate_amplitude_list(const StateFinite<Scalar> &state, long mps_pos) {
//     // Generate a list of bit sequences of size prod_{i=0}^pos spin_dim_i.
//     // For spin-half this is just 2^pos elements, where pos is the mps position
//     // counting from the left.
//     // Example: Let state_pos == 4 and state_len == 8. Then
//     // num_bitseqs = prod_{i=0}^{mpo_pos} = spin_dim_0 * spin_dim_1 * spin_dim_2 = 2³ = 8,
//     //      if mpo_pos == 2, "from A",  we generate
//     //          000, 001, 010, 011, 100, 101, 110, 111
//     //      if mpo_pos == 6 "from B" we also generate
//     //          000, 001, 010, 011, 100, 101, 110, 111
//     // So we generate the same regardless, but in the first case the numbers are interpreted
//     // in "in reverse" so that 001 becomes 100.
//
//     auto t_amp     = tid::tic_scope("amplitude");
//     auto state_pos = state.get_position<long>();
//     auto state_len = state.get_length<long>();
//     auto spinprod  = [](long &acc, const auto &mps) {
//         if(mps->spin_dim() != 2) throw std::runtime_error("number_entropies: spin_dim() != 2 is not supported");
//         return acc * mps->spin_dim();
//     };
//
//     long num_bitseqs;
//     if(mps_pos <= state_pos) {
//         num_bitseqs = std::accumulate(state.mps_sites.begin(), state.mps_sites.begin() + mps_pos + 1, 1l, spinprod);
//     } else {
//         auto mps_rpos = state_len - 1 - mps_pos;
//         num_bitseqs   = std::accumulate(state.mps_sites.rbegin(), state.mps_sites.rbegin() + mps_rpos + 1, 1l, spinprod);
//     }
//
//     std::vector<Amplitude<from>> amplitudes;
//     amplitudes.reserve(safe_cast<size_t>(num_bitseqs));
//     long start_pos = from == From::A ? -1l : state_len;
//     for(long count = 0; count < num_bitseqs; count++)
//         amplitudes.emplace_back(Amplitude<from>{state_len, start_pos, std::bitset<64>(safe_cast<unsigned long long int>(count)), {}});
//
//     return amplitudes;
// }