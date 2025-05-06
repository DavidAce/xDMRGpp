#pragma once
#include "../../mps.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"
#include <bitset>
#include <fmt/ranges.h>


/*!
 * There are many ways to generate an initial product state based on the
 * arguments (axis ,use_eigenspinors, pattern) = (string,bool,std::string).
 * Let
 *      * axis="str"         where str is valid if it is {"x","+x","-x","y","+y","-y","z","+z","-z"},
 *                           and the sign corresponds to the sign of the global spin compoonent (parity sector).
 *                           When the sign is present, the spin on the last site is set manually to end up on the correct axis and parity sector.
 *      * use_eigenspinors = true selects on of {|up>, |down>} along axis randomly on each site,
 *                           whereas false selects a|up> + b|down> with random a,b on each site.
 *      * pattern is only enabled if it is a non-empty string. The 0/1 bits in this pattern are interpreted as [up/down].
 * Then
 *
 *       a) ("random" , ignored , ignored)    Set each spinor randomly on C2 (or R2 if type == REAL)
 *       b) (valid axis, ignored, enabled)    Interpret seed_state as bitfield "01100010110..." and interpret these as
 *                                            up(0)/down(1) eigenspinors of the pauli matrix corresponding to "axis".
 *                                            Note that the axis sign is ignored.
 *                                            Note that the bitfield is used only once. Subsequent calls with the same bitfield
 *                                            will verert to case
 *       c) (valid axis, true, disabled)      Set each |spinor> = |up> or |down> with 50% probability,
 *                                            and |up/down> are eigenspinors of the pauli matrix specified in "axis". If the global
 *                                            sign (+-) is omitted, a random sign is chosen with equal probabilities. In the x and
 *                                            z cases the global state becomes entirely real, which improves performance.
 *       d) (valid axis, false, disabled)     Set each |spinor> = a|up> + b|down>, where a and b are random,
 *                                            real and normalized weights, and |up/down> are eigenspinors of the pauli matrix
 *                                            specified in "axis". Note that the axis sign is ignored.


 * Note: we "use" the bitfield only once. Subsequent calls do not keep resetting the seed.
*/
template<typename Scalar>
void tools::finite::mps::init::random_product_state(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, bool use_eigenspinors,
                                                    std::string &pattern) {
    tools::log->debug("Setting random product state of type {} on axis {}", enum2sv(type), axis);
    state.clear_measurements();
    state.clear_cache();
    auto axis_valid = qm::spin::half::is_valid_axis(axis);
    if(axis == "random") {
        init::set_random_product_state_with_random_spinors(state, type, pattern); // a)
    } else if(not pattern.empty()) {
        init::set_product_state_on_axis_using_pattern(state, type, axis, pattern); // b)
    } else if(use_eigenspinors and axis_valid) {
        init::set_random_product_state_on_axis_using_eigenspinors(state, type, axis, pattern); // c)
    } else if(axis_valid) {
        init::set_random_product_state_on_axis(state, type, axis, pattern); // d)
    } else {
        throw except::runtime_error("Expected initial axis string (+ or -)x,y,z,i, or id (e.g., -z). Got {}", axis);
    }
}


template<typename Scalar>
void tools::finite::mps::init::set_product_state_neel_shuffled(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern) {
    tools::log->debug("Setting randomly shuffled Néel state of type {} on axis {} {}", enum2sv(type), axis,
                      pattern.empty() ? "" : fmt::format(" | from pattern: {}", pattern));
    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    using namespace qm::spin::half::tensor;
    Eigen::Tensor<cx64, 3> spin_up = get_spinor(axus, +1).reshape(tenx::array3{2, 1, 1});
    Eigen::Tensor<cx64, 3> spin_dn = get_spinor(axus, -1).reshape(tenx::array3{2, 1, 1});

    auto bitfield = tools::get_bitfield(state.get_length(), pattern);
    if(bitfield.size() != state.get_length()) {
        bitfield.resize(state.get_length(), 0);
        for(auto &&[i, b] : iter::enumerate(bitfield)) b = num::mod<size_t>(i, 2) == 0 ? '0' : '1'; // Set Neel pattern 0101010 or 10101010..
        rnd::shuffle(bitfield);                                                                     // Shuffle the sequence randomly like 101011110100...
    }
    std::string label = "A";
    for(const auto &[pos, mps_ptr] : iter::enumerate(state.mps_sites)) {
        auto &&mps = *mps_ptr;
        mps.set_mps(bitfield.at(pos) == '0' ? spin_up : spin_dn, L, 0, label);

        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    tools::log->debug("Initial state: {}", bitfield);
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = safe_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1'));
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}


template<typename Scalar>
void tools::finite::mps::init::set_product_state_neel_dislocated(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern) {
    tools::log->debug("Setting a dislocated Néel state of type {} on axis {} {}", enum2sv(type), axis,
                      pattern.empty() ? "" : fmt::format(" | from pattern: {}", pattern));

    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    using namespace qm::spin::half::tensor;
    Eigen::Tensor<cx64, 3> spin_up  = get_spinor(axus, +1).reshape(tenx::array3{2, 1, 1});
    Eigen::Tensor<cx64, 3> spin_dn  = get_spinor(axus, -1).reshape(tenx::array3{2, 1, 1});
    auto                   bitfield = tools::get_bitfield(state.get_length(), pattern);
    if(bitfield.empty() or bitfield.size() != state.get_length()) {
        bitfield.resize(state.get_length(), 0);
        // Sets pattern 010101101010
        for(auto &&[i, p] : iter::enumerate(bitfield)) {
            if(i < state.get_length() / 2)
                p = num::mod<size_t>(i, 2) == 0 ? '0' : '1'; // Set Neel pattern 010101
            else
                p = num::mod<size_t>(i, 2) == 0 ? '1' : '0'; // Set Neel pattern 101010
        }
        if(rnd::uniform_integer_box<size_t>(0, 1) == 1) {
            // Reverse with 50% probability
            std::reverse(bitfield.begin(), bitfield.end());
        }
    }

    std::string label = "A";
    for(const auto &[pos, mps_ptr] : iter::enumerate(state.mps_sites)) {
        auto &&mps = *mps_ptr;
        mps.set_mps(bitfield.at(pos) == '0' ? spin_up : spin_dn, L, 0, label);
        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = safe_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1'));
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
    tools::log->debug("Initial state: {}", bitfield);
}

template<typename Scalar>
void tools::finite::mps::init::set_product_state_domain_wall(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern) {
    tools::log->debug("Setting domain-wall initial state of type {} on axis {} {}", enum2sv(type), axis,
                      pattern.empty() ? "" : fmt::format(" | from pattern: {}", pattern));

    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    using namespace qm::spin::half::tensor;
    Eigen::Tensor<cx64, 3> spin_up = get_spinor(axus, +1).reshape(tenx::array3{2, 1, 1});
    Eigen::Tensor<cx64, 3> spin_dn = get_spinor(axus, -1).reshape(tenx::array3{2, 1, 1});

    std::string bitfield;
    for(const auto &[pos, mps_ptr] : iter::enumerate(state.mps_sites)) {
        if(pos < state.get_length() / 2)
            bitfield.push_back('0');
        else
            bitfield.push_back('1');
    }
    std::string label = "A";
    for(const auto &[pos, mps_ptr] : iter::enumerate(state.mps_sites)) {
        auto &&mps = *mps_ptr;
        mps.set_mps(bitfield.at(pos) == '0' ? spin_up : spin_dn, L, 0, label);
        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    tools::log->debug("Initial state: {}", bitfield);
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = safe_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1'));
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}


template<typename Scalar>
void tools::finite::mps::init::set_product_state_aligned(StateFinite<Scalar> &state, StateInitType type, std::string_view axis,
                                                         [[maybe_unused]] std::string &pattern) {
    tools::log->debug("Setting Néel state of type {} on axis {} {}", enum2sv(type), axis, pattern.empty() ? "" : fmt::format(" | from pattern: {}", pattern));
    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    int  sign = qm::spin::half::get_sign(axis);
    if(type == StateInitType::REAL and axis == "y") throw std::runtime_error("StateInitType REAL incompatible with state in axis [y] which impliex CPLX");
    using namespace qm::spin::half::tensor;
    Eigen::Tensor<cx64, 3> spin_up = get_spinor(axus, +1).reshape(tenx::array3{2, 1, 1});
    Eigen::Tensor<cx64, 3> spin_dn = get_spinor(axus, -1).reshape(tenx::array3{2, 1, 1});

    Eigen::Tensor<cx64, 3> spinor = tenx::TensorCast(qm::spin::half::get_spinor(axus, sign), 2, 1, 1);

    tools::log->debug("Setting product state aligned using the |{}> eigenspinor of the pauli matrix σ{} on all sites", sign, axis);
    std::string label = "A";
    std::string bitfield(state.template get_length<size_t>(), sign >= 0 ? '0' : '1');
    for(const auto &mps_ptr : state.mps_sites) {
        auto &mps = *mps_ptr;
        mps.set_mps(spinor, L, 0, label);
        // mps.set_mps(bitfield.at(0) == '0' ? spin_up : spin_dn, L, 0, label);

        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = safe_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1'));
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
    tools::log->debug("Initial state: {}", bitfield);
}


template<typename Scalar>
void tools::finite::mps::init::set_product_state_neel(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern) {
    tools::log->debug("Setting Néel state of type {} on axis {} {}", enum2sv(type), axis, pattern.empty() ? "" : fmt::format(" | from pattern: {}", pattern));

    using Real = typename StateFinite<Scalar>::RealScalar;
    using Cplx = std::complex<Real>;

    Eigen::Tensor<Real, 1> L(1);
    L.setConstant(Real{1});
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    using namespace qm::spin::half::tensor;
    Eigen::Tensor<Cplx, 3> spin_up = tenx::asScalarType<Cplx>(get_spinor(axus, +1)).reshape(tenx::array3{2, 1, 1});
    Eigen::Tensor<Cplx, 3> spin_dn = tenx::asScalarType<Cplx>(get_spinor(axus, -1)).reshape(tenx::array3{2, 1, 1});
    // Eigen::Tensor<cx64, 3> spin_up = tenx::TensorCast(qm::spin::half::get_spinor(axus, +1), 2, 1, 1);
    // Eigen::Tensor<cx64, 3> spin_dn = tenx::TensorCast(qm::spin::half::get_spinor(axus, -1), 2, 1, 1);
    auto bitfield = tools::get_bitfield(state.get_length(), pattern);
    if(bitfield.empty() or bitfield.size() != state.get_length()) {
        bitfield.resize(state.get_length(), 0);
        for(auto &&[i, p] : iter::enumerate(bitfield)) p = num::mod<size_t>(i, 2) == 0 ? '0' : '1'; // Set Neel pattern 0101010
        if(rnd::uniform_integer_box<size_t>(0, 1) == 1) {
            // Reverse with 50% probability
            std::reverse(bitfield.begin(), bitfield.end());
        }
    }

    std::string label = "A";
    for(const auto &[pos, mps_ptr] : iter::enumerate(state.mps_sites)) {
        auto &&mps = *mps_ptr;
        mps.set_mps(bitfield.at(pos) == '0' ? spin_up : spin_dn, L, 0, label);
        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = safe_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1'));
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
    tools::log->debug("Initial state: {}", bitfield);
}


template<typename Scalar>
void tools::finite::mps::init::set_random_product_state_with_random_spinors(StateFinite<Scalar> &state, StateInitType type, std::string &pattern) {
    tools::log->debug("Setting random product state with spinors in C²");
    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    std::string label = "A";
    for(auto &mps_ptr : state.mps_sites) {
        auto &mps = *mps_ptr;
        if(type == StateInitType::CPLX)
            mps.set_mps(tenx::TensorCast(Eigen::VectorXcd::Random(2).normalized(), 2, 1, 1), L, 0, label);
        else if(type == StateInitType::REAL)
            mps.set_mps(tenx::TensorCast(Eigen::VectorXd::Random(2).normalized().cast<cx64>(), 2, 1, 1), L, 0, label);
        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    pattern.clear();
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}


template<typename Scalar>
void tools::finite::mps::init::set_product_state_on_axis_using_pattern(StateFinite<Scalar> &state, StateInitType type, std::string_view axis,
                                                                       std::string &pattern) {
    /* The pattern is a string containing either an unsigned number or a binary string. Binary strings are preceded by the character 'b'.
     * For example
     *      pattern == "536" for L=16 sites is interpreted with fmt as "{:0>16b}": 0000001000011000
     *      pattern == "b0000001000011000" is interpreted as "0000001000011000"
     *
     */

    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(type == StateInitType::REAL and axus == "y") throw except::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    if(pattern.empty()) throw except::runtime_error("Initial state pattern is an empty string.");
    using namespace qm::spin::half::tensor;
    Eigen::Tensor<cx64, 3> spin_up  = get_spinor(axus, +1).reshape(tenx::array3{2, 1, 1});
    Eigen::Tensor<cx64, 3> spin_dn  = get_spinor(axus, -1).reshape(tenx::array3{2, 1, 1});
    auto                   bitfield = tools::get_bitfield(state.get_length(), pattern);

    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    std::string label = "A";
    for(auto &mps_ptr : state.mps_sites) {
        auto &mps = *mps_ptr;
        auto  pos = mps.get_position();
        mps.set_mps(bitfield.at(pos) == '0' ? spin_up : spin_dn, L, 0, label);
        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    tools::log->debug("Initial state on axis {}: {}", axis, bitfield);
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = safe_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1'));
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}


template<typename Scalar>
void tools::finite::mps::init::set_sum_of_random_product_states(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern) {
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(type == StateInitType::REAL and axus == "y") throw except::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    using namespace qm::spin::half::tensor;

    size_t                   nsum = state.template get_length<size_t>();
    std::vector<std::string> patterns(nsum);
    set_product_state_neel_shuffled(state, type, axis, patterns[0]);

    auto other = StateFinite<Scalar>(state.get_algorithm(), state.template get_length<size_t>(), state.template get_position<long>(), state.get_spin_dim());
    for(size_t n = 1; n < nsum; ++n) {
        set_product_state_neel_shuffled(other, type, axis, patterns[n]);
        tools::log->info("{}", patterns[n]);
        state = add_states(state, other);
    }
    pattern = patterns.back();

    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}


template<typename Scalar>
void tools::finite::mps::init::set_random_product_state_on_axis_using_eigenspinors(StateFinite<Scalar> &state, StateInitType type, std::string_view axis,
                                                                                   std::string &pattern) {
    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    auto axus     = qm::spin::half::get_axis_unsigned(axis);
    int  sign_tgt = qm::spin::half::get_sign(axis);
    int  sign_now = 1;
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    std::string label = "A";
    using namespace qm::spin::half::tensor;
    Eigen::Tensor<cx64, 3> spin_up = get_spinor(axus, +1).reshape(tenx::array3{2, 1, 1});
    Eigen::Tensor<cx64, 3> spin_dn = get_spinor(axus, -1).reshape(tenx::array3{2, 1, 1});

    auto bitfield = tools::get_bitfield(state.get_length(), pattern);
    if(bitfield.empty() or bitfield.size() != state.get_length()) {
        bitfield.resize(state.get_length() + 1, 0);
        for(auto &&[i, b] : iter::enumerate(bitfield)) {
            if(i == 0) {
                b = 'b';
                continue;
            } else {
                b = rnd::uniform_integer_box<size_t>(0, 1) == 0 ? '0' : '1';
                sign_now *= b == '0' ? 1 : -1;
            }
        }
        if(sign_now * sign_tgt < 0) bitfield.back() = sign_now == -1 ? 1 : 0; // Flips the last spin so that the total sign product matches sign_tgt
    }

    for(auto &mps_ptr : state.mps_sites) {
        auto &mps = *mps_ptr;
        auto  pos = mps.template get_position<size_t>();
        mps.set_mps(bitfield.at(pos) == '0' ? spin_up : spin_dn, L, 0, label);

        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }

    tools::log->debug("Set random product state on axis {} using eigenspinors of the pauli matrix σ{}: {}", axis, axus, bitfield);
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = static_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1'));
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}


template<typename Scalar>
void tools::finite::mps::init::set_random_product_state_on_axis(StateFinite<Scalar> &state, StateInitType type, std::string_view axis,
                                                                [[maybe_unused]] std::string &pattern) {
    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    tools::log->debug("Setting random product state on axis {} using linear combinations of eigenspinors a|+> + b|-> of the pauli matrix σ{}", axus, axus);
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    auto spinor_up = qm::spin::half::get_spinor(axus, +1);
    auto spinor_dn = qm::spin::half::get_spinor(axus, -1);

    std::string label = "A";
    for(auto &mps_ptr : state.mps_sites) {
        auto            &mps = *mps_ptr;
        Eigen::Vector2cd ab;
        if(type == StateInitType::REAL)
            ab = Eigen::Vector2d::Random().normalized();
        else
            ab = Eigen::Vector2cd::Random().normalized();

        Eigen::VectorXcd spinor = ab(0) * spinor_up + ab(1) * spinor_dn;
        mps.set_mps(tenx::TensorCast(spinor.normalized(), 2, 1, 1), L, 0, label);
        if(mps.isCenter()) {
            mps.set_LC(L);
            label = "B";
        }
    }
    pattern.clear();
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}

