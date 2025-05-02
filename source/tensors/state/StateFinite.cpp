#include "math/tenx.h"

// -- (textra first)
#include "config/settings.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "math/cast.h"
#include "math/linalg/tensor.h"
#include "math/num.h"
#include "math/stat.h"
#include "StateFinite.h"
#include "StateFinite.impl.h"
#include "tensors/site/mps/MpsSite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/measure/truncation.h"
#include "tools/finite/multisite.h"
#include <fmt/ranges.h>

template class StateFinite<fp32>;
template class StateFinite<fp64>;
template class StateFinite<fp128>;
template class StateFinite<cx32>;
template class StateFinite<cx64>;
template class StateFinite<cx128>;

template<typename Scalar>
StateFinite<Scalar>::StateFinite() = default; // Can't initialize lists since we don't know the model size yet

// We need to define the destructor and other special functions
// because we enclose data in unique_ptr for this pimpl idiom.
// Otherwise, unique_ptr will forcibly inline its own default deleter.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
template<typename Scalar>
StateFinite<Scalar>::~StateFinite() noexcept = default; // default dtor
template<typename Scalar>
StateFinite<Scalar>::StateFinite(StateFinite &&other) noexcept = default; // default move ctor
template<typename Scalar>
StateFinite<Scalar> &StateFinite<Scalar>::operator=(StateFinite &&other) noexcept = default; // default move assign

/* clang-format off */
template<typename Scalar> StateFinite<Scalar>::StateFinite(const StateFinite &other) noexcept :
    direction(other.direction),
    cache_fp32(other.cache_fp32),
    cache_fp64(other.cache_fp64),
    cache_fp128(other.cache_fp128),
    cache_cx32(other.cache_cx32),
    cache_cx64(other.cache_cx64),
    cache_cx128(other.cache_cx128),
    tag_normalized_sites(other.tag_normalized_sites),
    name(other.name),
    algo(other.algo),
    convrates(other.convrates),
    active_sites(other.active_sites),
    measurements(other.measurements),
    popcount(other.popcount)
{
    mps_sites.clear();
    mps_sites.reserve(other.mps_sites.size());
    for(const auto &mps_other : other.mps_sites) { mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(*mps_other)); }

}
/* clang-format on */

template<typename Scalar>
StateFinite<Scalar> &StateFinite<Scalar>::operator=(const StateFinite &other) noexcept {
    if(this == &other) return *this; // check for self-assignment

    direction            = other.direction;
    cache_fp32           = other.cache_fp32;
    cache_fp64           = other.cache_fp64;
    cache_fp128          = other.cache_fp128;
    cache_cx32           = other.cache_cx32;
    cache_cx64           = other.cache_cx64;
    cache_cx128          = other.cache_cx128;
    tag_normalized_sites = other.tag_normalized_sites;
    name                 = other.name;
    algo                 = other.algo;
    convrates            = other.convrates;
    active_sites         = other.active_sites;
    measurements         = other.measurements;
    popcount             = other.popcount;
    mps_sites.clear();
    mps_sites.reserve(other.mps_sites.size());
    for(const auto &mps_other : other.mps_sites) { mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(*mps_other)); }
    return *this;
}

template<typename Scalar>
StateFinite<Scalar>::StateFinite(AlgorithmType algo_type, size_t model_size, long position, long spin_dim) {
    initialize(algo_type, model_size, position, spin_dim);
}

template<typename Scalar>
void StateFinite<Scalar>::initialize(AlgorithmType algo_type, size_t model_size, long position, long spin_dim) {
    set_algorithm(algo_type);
    tools::log->debug("Initializing state: sites {} | position {} | spin_dim {}", model_size, position, spin_dim);
    if(model_size < 2) throw except::logic_error("Tried to initialize state with less than 2 sites");
    if(model_size > 2048) throw except::logic_error("Tried to initialize state with more than 2048 sites");
    if(position >= safe_cast<long>(model_size)) throw except::logic_error("Tried to initialize state at a position larger than the number of sites");

    mps_sites.clear();

    // Generate a simple state with all spins equal
    Eigen::Tensor<cx64, 3> M(safe_cast<long>(spin_dim), 1, 1);
    Eigen::Tensor<cx64, 1> L(1);
    M(0, 0, 0) = 0;
    M(1, 0, 0) = 1;
    L(0)       = 1;
    for(size_t site = 0; site < model_size; site++) {
        std::string label = safe_cast<long>(site) <= position ? "A" : "B";
        mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(M, L, site, 0.0, label));
        if(safe_cast<long>(site) == position) { mps_sites.back()->set_LC(L); }
    }
    if(mps_sites.size() != model_size) throw except::logic_error("Initialized state with wrong size");
    if(not get_mps_site(position).isCenter()) throw except::logic_error("Initialized state center bond at the wrong position");
    if(get_position<long>() != position) throw except::logic_error("Initialized state at the wrong position");
    tag_normalized_sites = std::vector<bool>(model_size, false);
}

template<typename Scalar>
long StateFinite<Scalar>::find_position() const {
    long pos = -1l; // default value if the position is not found
    for(const auto &mps : mps_sites)
        if(mps->isCenter()) {
            if constexpr(settings::debug)
                if(pos != -1l) throw except::logic_error("Found multiple centers: first center at {} and another at {}", pos, mps->get_position());
            pos = mps->template get_position<long>();
        }
    // If no center position was found, then all sites are "B" sites. In that case, return -1
    return pos;
}

template<typename Scalar>
void StateFinite<Scalar>::set_name(std::string_view statename) {
    name = statename;
}
template<typename Scalar>
std::string_view StateFinite<Scalar>::get_name() const {
    return name;
}

template<typename Scalar>
void StateFinite<Scalar>::set_algorithm(const AlgorithmType &algo_type) {
    algo = algo_type;
}
template<typename Scalar>
AlgorithmType StateFinite<Scalar>::get_algorithm() const {
    return algo;
}

template<typename Scalar>
void StateFinite<Scalar>::set_positions() {
    for(auto &&[pos, mps] : iter::enumerate(mps_sites)) mps->set_position(pos);
}

template<typename Scalar>
long StateFinite<Scalar>::get_largest_bond() const {
    auto bond_dimensions = tools::finite::measure::bond_dimensions(*this);
    return *max_element(std::begin(bond_dimensions), std::end(bond_dimensions));
}

template<typename Scalar>
long StateFinite<Scalar>::get_largest_bond(const std::vector<size_t> &sites) const {
    // Get the largest bond in the interior of sites
    auto bond_dimensions = tools::finite::measure::bond_dimensions(*this);
    long bond_max        = 0;
    for(const auto &i : sites) {
        if(i == sites.back() and sites.size() >= 2) continue;
        const auto &mps = get_mps_site(i);
        bond_max        = std::max(bond_max, mps.get_chiR());
    }
    return bond_max;
}

template<typename Scalar>
double StateFinite<Scalar>::get_smallest_schmidt_value() const {
    double schmidt_min = 1;
    for(const auto &mps : mps_sites) {
        const auto &L     = mps->get_L();
        double      lastL = static_cast<double>(std::real(L.coeff(L.size() - 1)));
        schmidt_min       = std::min(schmidt_min, lastL);
        if(mps->isCenter()) {
            const auto &LC     = mps->get_LC();
            double      lastLC = static_cast<double>(std::real(LC.coeff(LC.size() - 1)));
            schmidt_min        = std::min(schmidt_min, lastLC);
        }
    }
    return schmidt_min;
}

template<typename Scalar>
int StateFinite<Scalar>::get_direction() const {
    return direction;
}
template<typename Scalar>
std::vector<std::string_view> StateFinite<Scalar>::get_labels() const {
    std::vector<std::string_view> labels;
    labels.reserve(get_length());
    for(const auto &mps : mps_sites) labels.emplace_back(mps->get_label());
    return labels;
}

template<typename Scalar>
void StateFinite<Scalar>::flip_direction() {
    direction *= -1;
}

template<typename Scalar>
std::array<long, 3> StateFinite<Scalar>::dimensions_1site() const {
    auto pos = get_position<long>();
    if(pos >= 0)
        return get_mps_site(pos).dimensions();
    else
        return {0, 0, 0};
}

template<typename Scalar>
std::array<long, 3> StateFinite<Scalar>::dimensions_2site() const {
    std::array<long, 3> dimensions{};
    auto                pos  = get_position<long>();
    auto                posL = std::clamp<long>(pos, 0, get_length<long>() - 2);
    auto                posR = std::clamp<long>(pos + 1, 0, get_length<long>() - 1);
    const auto         &mpsL = get_mps_site(posL);
    const auto         &mpsR = get_mps_site(posR);
    dimensions[1]            = mpsL.get_chiL();
    dimensions[2]            = mpsR.get_chiR();
    dimensions[0]            = posL != posR ? mpsL.spin_dim() * mpsR.spin_dim() : mpsL.spin_dim();
    return dimensions;
}

template<typename Scalar>
std::array<long, 3> StateFinite<Scalar>::dimensions_nsite() const {
    return tools::finite::multisite::get_dimensions(*this, active_sites);
}

template<typename Scalar>
long StateFinite<Scalar>::size_1site() const {
    auto dims = dimensions_1site();
    return dims[0] * dims[1] * dims[2];
}

template<typename Scalar>
long StateFinite<Scalar>::size_2site() const {
    auto dims = dimensions_2site();
    return dims[0] * dims[1] * dims[2];
}

template<typename Scalar>
long StateFinite<Scalar>::size_nsite() const {
    auto dims = dimensions_nsite();
    return dims[0] * dims[1] * dims[2];
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_the_middle() const {
    return get_position<long>() + 1 == get_length<long>() / 2 and direction == 1;
}
template<typename Scalar>
bool StateFinite<Scalar>::position_is_the_middle_any_direction() const {
    return get_position<long>() + 1 == get_length<long>() / 2;
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_outward_edge_left([[maybe_unused]] size_t nsite) const {
    if(nsite == 1) {
        return get_position<long>() <= -1 and direction == -1; // i.e. all sites are B's
    } else
        return get_position<long>() == 0 and direction == -1 and get_mps_site().isCenter(); // left-most site is a an AC
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_outward_edge_right(size_t nsite) const {
    return get_position<long>() >= get_length<long>() - safe_cast<long>(nsite) and direction == 1;
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_outward_edge(size_t nsite) const {
    return position_is_outward_edge_left(nsite) or position_is_outward_edge_right(nsite);
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_inward_edge_left([[maybe_unused]] size_t nsite) const {
    return get_position<long>() == 0 and direction == 1; // i.e. first site is an AC going to the right
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_inward_edge_right(size_t nsite) const {
    return get_position<long>() >= get_length<long>() - safe_cast<long>(nsite) and direction == -1;
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_inward_edge(size_t nsite) const {
    return position_is_inward_edge_left(nsite) or position_is_inward_edge_right(nsite);
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_at(long pos) const {
    return get_position<long>() == pos;
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_at(long pos, int dir) const {
    return get_position<long>() == pos and get_direction() == dir;
}

template<typename Scalar>
bool StateFinite<Scalar>::position_is_at(long pos, int dir, bool isCenter) const {
    return get_position<long>() == pos and get_direction() == dir and (pos >= 0) == isCenter;
}

template<typename Scalar>
bool StateFinite<Scalar>::has_center_point() const {
    return get_position<long>() >= 0;
}

template<typename Scalar>
bool StateFinite<Scalar>::is_real() const {
    if(std::is_floating_point_v<Scalar>) return true;
    bool mps_real = true;
    for(const auto &mps : mps_sites) mps_real = mps_real and mps->is_real();
    return mps_real;
}

template<typename Scalar>
bool StateFinite<Scalar>::has_nan() const {
    for(const auto &mps : mps_sites)
        if(mps->has_nan()) return true;
    return false;
}

template<typename Scalar>
void StateFinite<Scalar>::assert_validity() const {
    size_t pos = 0;
    for(const auto &mps : mps_sites) {
        if(pos != mps->template get_position<size_t>())
            throw except::runtime_error("State is corrupted: position mismatch: expected position {} != mps position {}", pos, mps->get_position());
        pos++;
    }

    for(const auto &mps : mps_sites) mps->assert_validity();
    if(settings::model::model_type == ModelType::ising_sdual or settings::model::model_type == ModelType::ising_majorana) {
        for(const auto &mps : mps_sites)
            if(not mps->is_real()) {
                auto M = mps->get_M_bare();
                for(long idx = 0; idx < M.size(); ++idx) tools::log->warn("elem {}: {:.8e}", mps->get_position(), fp(M.data()[idx]));
                throw except::runtime_error("state has imaginary part at mps position {}", mps->get_position());
            }
    }
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 1> &StateFinite<Scalar>::get_bond(long posL, long posR) const {
    if(posL + 1 != posR) throw except::runtime_error("Expected posL+1 == posR, got: posL {}, posR {}", posL, posR);
    auto pos = get_position<long>();
    if(pos < posL) return get_mps_site(posL).get_L(); // B.B
    if(pos > posL) return get_mps_site(posR).get_L(); // A.A or A.AC
    return get_mps_site(posL).get_LC();               // AC.B
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 1> &StateFinite<Scalar>::get_midchain_bond() const {
    auto pos = get_position<long>();
    auto cnt = (get_length<long>() - 1) / 2;
    if(pos < cnt) return get_mps_site(cnt).get_L();
    if(pos > cnt) return get_mps_site(cnt + 1).get_L();
    return get_mps_site(cnt).get_LC();
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 1> &StateFinite<Scalar>::current_bond() const {
    return get_mps_site(get_position()).get_LC();
}

template<typename Scalar>
const MpsSite<Scalar> &StateFinite<Scalar>::get_mps_site() const {
    return get_mps_site(get_position());
}

template<typename Scalar>
MpsSite<Scalar> &StateFinite<Scalar>::get_mps_site() {
    return get_mps_site(get_position());
}

template<typename Scalar>
void StateFinite<Scalar>::set_mps(const std::vector<MpsSite<Scalar>> &mps_list) {
    for(const auto &mps_new : mps_list) {
        auto  pos     = mps_new.get_position();
        auto &mps_old = get_mps_site(pos);
        if(mps_new.has_L() and mps_new.has_M() and mps_old.get_label() == mps_new.get_label())
            mps_old = mps_new;
        else {
            mps_old.set_label(mps_new.get_label());
            if(mps_new.has_M()) mps_old.set_M(mps_new.get_M_bare());
            if(mps_new.has_L()) mps_old.set_L(mps_new.get_L());
            if(mps_new.has_LC()) mps_old.set_LC(mps_new.get_LC());
        }
    }
}

template<typename Scalar>
std::vector<std::reference_wrapper<const MpsSite<Scalar>>> StateFinite<Scalar>::get_mps(const std::vector<size_t> &sites) const {
    std::vector<std::reference_wrapper<const MpsSite<Scalar>>> mps;
    mps.reserve(sites.size());
    for(auto &site : sites) mps.emplace_back(get_mps_site(site));
    return mps;
}

template<typename Scalar>
std::vector<std::reference_wrapper<MpsSite<Scalar>>> StateFinite<Scalar>::get_mps(const std::vector<size_t> &sites) {
    std::vector<std::reference_wrapper<MpsSite<Scalar>>> mps;
    mps.reserve(sites.size());
    for(auto &site : sites) mps.emplace_back(get_mps_site(site));
    return mps;
}

template<typename Scalar>
std::vector<MpsSite<Scalar>> StateFinite<Scalar>::get_mps_copy(const std::vector<size_t> &sites) {
    std::vector<MpsSite<Scalar>> mps;
    mps.reserve(sites.size());
    for(auto &site : sites) mps.emplace_back(get_mps_site(site));
    return mps;
}

template<typename Scalar>
std::vector<std::reference_wrapper<const MpsSite<Scalar>>> StateFinite<Scalar>::get_mps_active() const {
    return get_mps(active_sites);
}

template<typename Scalar>
std::vector<std::reference_wrapper<MpsSite<Scalar>>> StateFinite<Scalar>::get_mps_active() {
    return get_mps(active_sites);
}

template<typename Scalar>
std::array<long, 3> StateFinite<Scalar>::active_dimensions() const {
    return tools::finite::multisite::get_dimensions(*this, active_sites);
}

template<typename Scalar>
long StateFinite<Scalar>::active_problem_size() const {
    return tools::finite::multisite::get_problem_size(*this, active_sites);
}

template<typename Scalar>
std::vector<long> StateFinite<Scalar>::get_bond_dims(const std::vector<size_t> &sites) const {
    // If the sites are {2,3,4,5,6} this returns the 4 bonds connecting {2,3}, {3,4}, {4,5} and {5,6}
    // If sites is just {4}, it returns the bond between {4,5} when going left or right.
    if(sites.empty()) return {};
    if(sites.size() == 1) {
        // In single-site DMRG the active site is a center "AC = L G LC" site:
        //  * Going left-to-right, the forward (right) bond is expanded, and this same bond is truncated when merging
        //  * Going right-to-left, the forward (left) bond is expanded (L), but LC is still the one truncated when merging.
        return {get_mps_site(sites.front()).get_chiR()};
    }
    if(sites.size() == 2) return {get_mps_site(sites.front()).get_chiR()};
    std::vector<long> bond_dimensions;
    for(const auto &pos : sites) {
        if(&pos == &sites.front()) continue;
        const auto &mps = get_mps_site(pos);
        bond_dimensions.push_back(mps.get_chiL());
    }
    return bond_dimensions;
}
template<typename Scalar>
std::vector<long> StateFinite<Scalar>::get_bond_dims_active() const {
    return get_bond_dims(active_sites);
}

template<typename Scalar>
std::vector<long> StateFinite<Scalar>::get_spin_dims(const std::vector<size_t> &sites) const {
    if(sites.empty()) throw except::runtime_error("No sites on which to collect spin dimensions");
    std::vector<long> dims;
    dims.reserve(sites.size());
    for(const auto &site : sites) { dims.emplace_back(get_mps_site(site).spin_dim()); }
    return dims;
}

template<typename Scalar>
std::vector<long> StateFinite<Scalar>::get_spin_dims() const {
    return get_spin_dims(active_sites);
}
template<typename Scalar>
long StateFinite<Scalar>::get_spin_dim() const {
    return get_mps_site(0).spin_dim();
}

template<typename Scalar>
std::vector<std::array<long, 3>> StateFinite<Scalar>::get_mps_dims(const std::vector<size_t> &sites) const {
    std::vector<std::array<long, 3>> dims;
    for(const auto &pos : sites) dims.emplace_back(get_mps_site(pos).dimensions());
    return dims;
}

template<typename Scalar>
std::vector<std::array<long, 3>> StateFinite<Scalar>::get_mps_dims_active() const {
    return get_mps_dims(active_sites);
}

template<typename Scalar>
std::string StateFinite<Scalar>::generate_cache_key(const std::vector<size_t> &sites, const size_t pos, std::string_view side) const {
    if(sites.empty()) return {};
    assert(pos >= sites.front());
    assert(pos <= sites.back());
    std::string key;
    auto        nelems = 1 + pos - sites.front();
    key.reserve(nelems * 8);
    key += "[";
    if(side.starts_with('l')) {
        for(const auto &i : sites) {
            if(i == sites.front()) {
                key += fmt::format("{}{}{}", mps_sites[i]->get_label() == "B" ? "LB" : "A", sites.size() == 1 ? "L" : "", i);
                if(sites.size() == 1) key += "L";
            } else if(i == sites.back()) {
                key += fmt::format("{}{}L", mps_sites[i]->get_label() == "B" ? "B" : "A", i);
            } else {
                key += fmt::format("{}{}", mps_sites[i]->get_label() == "B" ? "B" : "A", i);
            }
            if(i == pos) break;
            key += ",";
        }
    }
    if(side.starts_with('r')) {
        for(const auto &i : sites) {
            if(i < pos) continue;
            if(i == sites.front()) {
                key += fmt::format("{}{}{}", mps_sites[i]->get_label() == "B" ? "LB" : "A", sites.size() == 1 ? "L" : "", i);
            } else if(i == sites.back()) {
                key += fmt::format("{}{}L", mps_sites[i]->get_label() == "B" ? "B" : "A", i);
            } else {
                key += fmt::format("{}{}", mps_sites[i]->get_label() == "B" ? "B" : "A", i);
            }
            if(i != sites.back()) key += ",";
        }
    }
    key += "]";
    return std::string(key.begin(), key.end()); // Return only the relevant part.
}

template<typename Scalar>
double StateFinite<Scalar>::get_trf_cache_gbts() const {
    double size_fp32 = 0, size_fp64 = 0, size_fp128 = 0, size_cx32 = 0, size_cx64 = 0, size_cx128 = 0;
    for(const auto &elem : get_cache<fp32>().trf) size_fp32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<fp64>().trf) size_fp64 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx32>().trf) size_cx32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx64>().trf) size_cx64 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx128>().trf) size_cx128 += static_cast<double>(elem.second.size());

    size_fp32 *= 4.0 / std::pow(1024.0, 3.0);
    size_fp64 *= 8.0 / std::pow(1024.0, 3.0);
    size_fp64 *= 16.0 / std::pow(1024.0, 3.0);
    size_cx32 *= 8.0 / std::pow(1024.0, 3.0);
    size_cx64 *= 16.0 / std::pow(1024.0, 3.0);
    size_cx128 *= 16.0 / std::pow(1024.0, 3.0);
    return size_fp32 + size_fp64 + size_fp128 + size_cx32 + size_cx64 + size_cx128;
}

template<typename Scalar>
double StateFinite<Scalar>::get_mps_cache_gbts() const {
    double size_fp32 = 0, size_fp64 = 0, size_fp128 = 0, size_cx32 = 0, size_cx64 = 0, size_cx128 = 0;
    for(const auto &elem : get_cache<fp32>().mps) size_fp32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<fp64>().mps) size_fp64 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<fp128>().mps) size_fp128 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx32>().mps) size_cx32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx64>().mps) size_cx64 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx128>().mps) size_cx128 += static_cast<double>(elem.second.size());

    size_fp32 *= 4.0 / std::pow(1024.0, 3.0);
    size_fp64 *= 8.0 / std::pow(1024.0, 3.0);
    size_fp64 *= 16.0 / std::pow(1024.0, 3.0);
    size_cx32 *= 8.0 / std::pow(1024.0, 3.0);
    size_cx64 *= 16.0 / std::pow(1024.0, 3.0);
    size_cx128 *= 32.0 / std::pow(1024.0, 3.0);
    return size_fp32 + size_fp64 + size_fp128 + size_cx32 + size_cx64 + size_cx128;
}

template<typename Scalar>
std::array<double, 2> StateFinite<Scalar>::get_cache_sizes() const {
    return {get_mps_cache_gbts(), get_trf_cache_gbts()};
}

template<typename Scalar>
void StateFinite<Scalar>::set_truncation_error(size_t pos, double error) {
    get_mps_site(pos).set_truncation_error(error);
}
template<typename Scalar>
void StateFinite<Scalar>::set_truncation_error(double error) {
    set_truncation_error(get_position(), error);
}

template<typename Scalar>
void StateFinite<Scalar>::set_truncation_error_LC(double error) {
    auto &mps = get_mps_site(get_position());
    if(not mps.isCenter()) throw except::runtime_error("mps at current position is not a center");
    mps.set_truncation_error_LC(error);
}

template<typename Scalar>
void StateFinite<Scalar>::keep_max_truncation_errors(std::vector<double> &other_errors) {
    auto errors = get_truncation_errors();
    if(other_errors.size() != errors.size()) throw except::runtime_error("keep_max_truncation_errors: size mismatch");
    std::vector<double> max_errors(errors.size(), 0);
    for(auto &&[i, e] : iter::enumerate(max_errors)) e = std::max({errors[i], other_errors[i]});
    other_errors = max_errors;

    // Now set the maximum errors back to each site
    size_t past_center = 0;
    for(const auto &mps : mps_sites) {
        auto pos = mps->template get_position<size_t>();
        auto idx = pos + past_center;
        set_truncation_error(pos, max_errors[idx]);
        if(mps->isCenter()) {
            past_center = 1;
            idx         = pos + past_center;
            set_truncation_error_LC(max_errors[idx]);
        }
    }
}

template<typename Scalar>
double StateFinite<Scalar>::get_truncation_error(size_t pos) const {
    return get_mps_site(pos).get_truncation_error();
}

template<typename Scalar>
double StateFinite<Scalar>::get_truncation_error() const {
    auto pos = get_position<long>();
    if(pos >= 0)
        return get_mps_site(pos).get_truncation_error();
    else
        return 0;
}

template<typename Scalar>
double StateFinite<Scalar>::get_truncation_error_LC() const {
    return get_mps_site(get_position()).get_truncation_error_LC();
}
template<typename Scalar>
double StateFinite<Scalar>::get_truncation_error_midchain() const {
    auto pos = get_position<long>();
    auto cnt = (get_length<long>() - 1) / 2;
    if(pos < cnt) return get_mps_site(cnt).get_truncation_error();
    if(pos > cnt) return get_mps_site(cnt + 1).get_truncation_error();
    return get_mps_site(cnt).get_truncation_error_LC();
}

template<typename Scalar>
std::vector<fp64> StateFinite<Scalar>::get_truncation_errors() const {
    return tools::finite::measure::truncation_errors(*this);
}
template<typename Scalar>
std::vector<fp64> StateFinite<Scalar>::get_truncation_errors_active() const {
    return tools::finite::measure::truncation_errors_active(*this);
}
template<typename Scalar>
fp64 StateFinite<Scalar>::get_truncation_error_active_max() const {
    auto truncation_errors_active = get_truncation_errors_active();
    return *std::max_element(truncation_errors_active.begin(), truncation_errors_active.end());
}

template<typename Scalar>
size_t StateFinite<Scalar>::num_sites_truncated(double truncation_threshold) const {
    auto truncation_errors = get_truncation_errors();
    auto trunc_bond_count  = safe_cast<size_t>(
        std::count_if(truncation_errors.begin(), truncation_errors.end(), [truncation_threshold](auto const &val) { return val > truncation_threshold; }));
    return trunc_bond_count;
}

template<typename Scalar>
size_t StateFinite<Scalar>::num_bonds_at_limit(long bond_lim) const {
    auto bond_dimensions = tools::finite::measure::bond_dimensions(*this);
    auto bonds_at_lim =
        safe_cast<size_t>(std::count_if(bond_dimensions.begin(), bond_dimensions.end(), [bond_lim](auto const &dim) { return dim >= bond_lim; }));
    return bonds_at_lim;
}

template<typename Scalar>
bool StateFinite<Scalar>::is_at_bond_limit(long bond_lim) const {
    return num_bonds_at_limit(bond_lim) > 0;
}

template<typename Scalar>
size_t StateFinite<Scalar>::num_bonds_at_maximum(const std::vector<size_t> &sites) const {
    if(sites.empty()) return 0;
    auto L            = get_length<size_t>();
    auto bond_dims    = tools::finite::measure::bond_dimensions(*this);
    auto spin_dims    = tools::finite::measure::spin_dimensions(*this);
    auto get_bond_max = [&](auto bond_idx) {
        if(bond_idx <= L / 2) {
            return std::accumulate(spin_dims.begin(), spin_dims.begin() + bond_idx, long(1), std::multiplies<long>());
        } else {
            return std::accumulate(spin_dims.begin() + bond_idx, spin_dims.end(), long(1), std::multiplies<long>());
        }
    };
    size_t num_bonds_at_max = 0;
    for(size_t i = 1; i < sites.size(); ++i) {
        assert(i < L);
        num_bonds_at_max += bond_dims[sites[i]] >= get_bond_max(sites[i]) ? 1 : 0;
    }
    return num_bonds_at_max;
}

template<typename Scalar>
bool StateFinite<Scalar>::is_truncated(double truncation_error_limit) const {
    auto truncation_errors = get_truncation_errors();
    auto num_above_lim     = static_cast<size_t>(
        std::count_if(truncation_errors.begin(), truncation_errors.end(), [truncation_error_limit](auto const &err) { return err >= truncation_error_limit; }));
    return num_above_lim > 0;
}

template<typename Scalar>
void StateFinite<Scalar>::clear_measurements(LogPolicy logPolicy) const {
    if(logPolicy == LogPolicy::VERBOSE or (settings::debug and logPolicy == LogPolicy::DEBUG)) { tools::log->trace("Clearing state measurements"); }
    measurements = MeasurementsStateFinite<Scalar>();
}

template<typename Scalar>
void StateFinite<Scalar>::clear_cache(LogPolicy logPolicy) const {
    if(logPolicy == LogPolicy::VERBOSE or (settings::debug and logPolicy == LogPolicy::DEBUG)) { tools::log->trace("Clearing state cache"); }
    cache_fp32  = Cache<fp32>();
    cache_fp64  = Cache<fp64>();
    cache_fp128 = Cache<fp128>();
    cache_cx32  = Cache<cx32>();
    cache_cx64  = Cache<cx64>();
    cache_cx128 = Cache<cx128>();
}

template<typename Scalar>
void StateFinite<Scalar>::shrink_cache() const {
    auto shrink_while = [&](auto &cache) {
        while(cache.mps.size() > max_mps_cache_size) cache.mps.pop_front();
        while(cache.trf.size() > max_trf_cache_size) cache.trf.pop_front();
    };
    auto shrink_once = [&](auto &cache) {
        using namespace settings;
        using namespace settings::precision;
        /* clang-format off */
        cache.mps.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del mps: {}", max_cache_gbts, cache.mps.front().first);
        cache.trf.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del trf: {}", max_cache_gbts, cache.trf.front().first);
        /* clang-format on */
    };
    shrink_while(cache_fp32);
    shrink_while(cache_fp64);
    shrink_while(cache_fp128);
    shrink_while(cache_cx32);
    shrink_while(cache_cx64);
    shrink_while(cache_cx128);

    while(get_mps_cache_gbts() + get_trf_cache_gbts() > std::max(0.0, settings::precision::max_cache_gbts)) {
        shrink_once(cache_fp32);
        shrink_once(cache_fp64);
        shrink_once(cache_fp128);
        shrink_once(cache_cx32);
        shrink_once(cache_cx64);
        shrink_once(cache_cx128);
    }
}

template<typename Scalar>
void StateFinite<Scalar>::tag_active_sites_normalized(bool tag) const {
    assert(tag_normalized_sites.size() == get_length());
    for(auto &site : active_sites) tag_normalized_sites[site] = tag;
}

template<typename Scalar>
void StateFinite<Scalar>::tag_all_sites_normalized(bool tag) const {
    assert(tag_normalized_sites.size() == get_length());
    tag_normalized_sites = std::vector<bool>(get_length(), tag);
}

template<typename Scalar>
void StateFinite<Scalar>::tag_site_normalized(size_t pos, bool tag) const {
    assert(tag_normalized_sites.size() == get_length());
    tag_normalized_sites[pos] = tag;
}

template<typename Scalar>
bool StateFinite<Scalar>::is_normalized_on_all_sites(RealScalar prec) const {
    if(tag_normalized_sites.size() != get_length()) throw except::runtime_error("Cannot check normalization status on all sites, size mismatch in site list");
    // If all tags are false then we should definitely normalize:
    auto normalized_none = std::none_of(tag_normalized_sites.begin(), tag_normalized_sites.end(), [](bool v) { return v; });
    if(normalized_none) {
        tools::log->debug("{} normalized: false (none)", get_name());
        return false;
    }

    prec = std::max(prec, std::numeric_limits<RealScalar>::epsilon() * 100);
    if constexpr(settings::debug) {
        auto normalized_some = std::any_of(tag_normalized_sites.begin(), tag_normalized_sites.end(), [](bool v) { return v; });
        if(normalized_some) {
            // In debug mode we check if the tags are truthful
            for(const auto &mps : mps_sites) {
                auto pos = mps->template get_position<size_t>();
                if(not tag_normalized_sites[pos]) {
                    if(mps->is_normalized(prec)) tag_normalized_sites[pos] = true;
                }
            }
        }
    }
    auto normalized_tags = std::all_of(tag_normalized_sites.begin(), tag_normalized_sites.end(), [](bool v) { return v; });
    auto normalized_fast = true;
    auto normalized_full = true;
    auto normalized_site = true;
    auto msg             = fmt::format("tags {}", normalized_tags);
    if(normalized_tags) {
        // We don't need this check fully if the tags already told us the state is normalized
        auto norm       = tools::finite::measure::norm(*this, false);
        auto norm_error = std::abs(norm - RealScalar{1});
        normalized_fast = num::leq(norm_error, prec);
        msg += fmt::format(" | fast {} norm error: {:.3e}", normalized_fast, fp(norm_error));
    }
    if constexpr(settings::debug) {
        if(normalized_tags and normalized_fast) {
            auto norm       = tools::finite::measure::norm(*this, true);
            normalized_full = num::leq(std::abs(norm - RealScalar{1}), prec);
            msg += fmt::format(" | full {} {:.3e}", normalized_full, fp(norm));
        }
        if(normalized_tags and normalized_fast and normalized_full) {
            std::vector<long> site_list;
            for(const auto &mps : mps_sites) {
                if(not mps->is_normalized(prec)) { site_list.emplace_back(mps->template get_position<long>()); }
            }
            if(not site_list.empty()) {
                normalized_site = false;
                msg += fmt::format(" | non-normalized sites {}", site_list);
            }
        }
    }
    tools::log->debug("{} normalized: {}", get_name(), msg);
    return normalized_tags and normalized_fast and normalized_full and normalized_site;
}

template<typename Scalar>
bool StateFinite<Scalar>::is_normalized_on_any_sites() const {
    if(tag_normalized_sites.size() != get_length()) throw except::runtime_error("Cannot check normalization status on any sites, size mismatch in site list");
    return std::any_of(tag_normalized_sites.begin(), tag_normalized_sites.end(), [](bool v) { return v; });
}

template<typename Scalar>
bool StateFinite<Scalar>::is_normalized_on_active_sites() const {
    if(tag_normalized_sites.size() != get_length())
        throw except::runtime_error("Cannot check normalization status on active sites, size mismatch in site list");
    if(active_sites.empty()) return false;
    auto first_site_ptr = std::next(tag_normalized_sites.begin(), safe_cast<long>(active_sites.front()));
    auto last_site_ptr  = std::next(tag_normalized_sites.begin(), safe_cast<long>(active_sites.back()));
    return std::all_of(first_site_ptr, last_site_ptr, [](bool v) { return v; });
}

template<typename Scalar>
bool StateFinite<Scalar>::is_normalized_on_non_active_sites() const {
    if(tag_normalized_sites.size() != get_length()) throw except::runtime_error("Cannot check update status on all sites, size mismatch in site list");
    if(active_sites.empty()) return is_normalized_on_all_sites();
    for(size_t idx = 0; idx < get_length(); idx++)
        if(std::find(active_sites.begin(), active_sites.end(), idx) == active_sites.end() and not tag_normalized_sites[idx]) return false;
    return true;
}

template<typename Scalar>
std::vector<size_t> StateFinite<Scalar>::get_active_ids() const {
    std::vector<size_t> ids;
    ids.reserve(active_sites.size());
    for(const auto &pos : active_sites) ids.emplace_back(get_mps_site(pos).get_unique_id());
    return ids;
}
template<typename Scalar>
const std::vector<bool> &StateFinite<Scalar>::get_normalization_tags() const {
    return tag_normalized_sites;
}
