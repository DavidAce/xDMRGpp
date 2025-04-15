#include "math/tenx.h"

// -- (textra first)
#include "config/settings.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "math/cast.h"
#include "math/num.h"
#include "math/stat.h"
#include "StateFinite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/finite/measure.h"
#include "tools/finite/multisite.h"
#include <fmt/ranges.h>

namespace settings {
    inline constexpr bool debug_state           = false;
    inline constexpr bool debug_cache           = false;
    inline constexpr bool debug_density_matrix  = false;
    inline constexpr bool debug_transfer_matrix = false;
}

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
template<typename Scalar> StateFinite<Scalar>::StateFinite(const StateFinite &other):
    direction(other.direction),
    cache_fp32(other.cache_fp32),
    cache_fp64(other.cache_fp64),
    cache_cx32(other.cache_cx32),
    cache_cx64(other.cache_cx64),
    tag_normalized_sites(other.tag_normalized_sites),
    name(other.name),
    algo(other.algo),
    active_sites(other.active_sites),
    measurements(other.measurements),
    popcount(other.popcount)
{
    mps_sites.clear();
    mps_sites.reserve(other.mps_sites.size());
    for(const auto &mps : other.mps_sites) mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(*mps));
}
/* clang-format on */

template<typename Scalar>
StateFinite<Scalar> &StateFinite<Scalar>::operator=(const StateFinite &other) {
    // check for self-assignment
    if(this != &other) {
        direction            = other.direction;
        cache_fp32           = other.cache_fp32;
        cache_fp64           = other.cache_fp64;
        cache_cx32           = other.cache_cx32;
        cache_cx64           = other.cache_cx64;
        tag_normalized_sites = other.tag_normalized_sites;
        name                 = other.name;
        algo                 = other.algo;
        active_sites         = other.active_sites;
        measurements         = other.measurements;
        popcount             = other.popcount;
        mps_sites.clear();
        mps_sites.reserve(other.mps_sites.size());
        for(const auto &mps : other.mps_sites) mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(*mps));
    }
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
template<typename T>
StateFinite<Scalar>::Cache<T> &StateFinite<Scalar>::get_cache() const {
    static_assert(sfinae::is_any_v<T, fp32, fp64, cx32, cx64>);
    /* clang-format off */
         if constexpr(std::is_same_v<T, fp32>) { return cache_fp32; }
    else if constexpr(std::is_same_v<T, fp64>) { return cache_fp64; }
    else if constexpr(std::is_same_v<T, cx32>) { return cache_cx32; }
    else if constexpr(std::is_same_v<T, cx64>) { return cache_cx64; }
    else throw std::logic_error("unrecognized cache type T");
    /* clang-format on */
}
template StateFinite<>::Cache<fp32> &StateFinite<>::get_cache() const;
template StateFinite<>::Cache<fp64> &StateFinite<>::get_cache() const;
template StateFinite<>::Cache<cx32> &StateFinite<>::get_cache() const;
template StateFinite<>::Cache<cx64> &StateFinite<>::get_cache() const;

// template<typename T>
// StateFinite<Scalar>::Cache<T> &StateFinite<Scalar>::get_cache() {
//     return const_cast<Cache &>(std::as_const(*this).get_cache<T>());
// }
// template StateFinite<Scalar>::Cache<fp32> &StateFinite<Scalar>::get_cache();
// template StateFinite<Scalar>::Cache<fp64> &StateFinite<Scalar>::get_cache();
// template StateFinite<Scalar>::Cache<cx32> &StateFinite<Scalar>::get_cache();
// template StateFinite<Scalar>::Cache<cx64> &StateFinite<Scalar>::get_cache();

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
template<typename T>
T StateFinite<Scalar>::get_length() const {
    return safe_cast<T>(mps_sites.size());
}
template double             StateFinite<>::get_length<double>() const;
template size_t             StateFinite<>::get_length<size_t>() const;
template long               StateFinite<>::get_length<long>() const;
template int                StateFinite<>::get_length<int>() const;
template unsigned long long StateFinite<>::get_length<unsigned long long>() const; // hsize_t from hdf5

template<typename Scalar>
template<typename T>
T StateFinite<Scalar>::get_position() const {
    std::optional<T> pos;
    for(const auto &mps : mps_sites)
        if(mps->isCenter()) {
            if(pos) throw except::logic_error("Found multiple centers: first center at {} and another at {}", pos.value(), mps->get_position());
            pos = mps->template get_position<T>();
        }
    // If no center position was found then all sites are "B" sites. In that case, return -1 if T is signed, otherwise throw.
    if(not pos) {
        if constexpr(std::is_signed_v<T>)
            return -1;
        else
            throw except::runtime_error("could not find center position in current state: {}\n"
                                        "hint: Call get_position<long>() to get -1",
                                        get_labels());
    } else
        return pos.value();
}
template size_t             StateFinite<>::get_position<size_t>() const;
template long               StateFinite<>::get_position<long>() const;
template int                StateFinite<>::get_position<int>() const;
template unsigned long long StateFinite<>::get_position<unsigned long long>() const; // hsize_t from hdf5

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
    RealScalar schmidt_min = 1;
    for(const auto &mps : mps_sites) {
        const auto &L = mps->get_L();
        schmidt_min   = std::min(schmidt_min, L.coeff(L.size() - 1).real());
        if(mps->isCenter()) {
            const auto &LC = mps->get_LC();
            schmidt_min    = std::min(schmidt_min, LC.coeff(LC.size() - 1).real());
        }
    }
    return static_cast<double>(schmidt_min);
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
    if (std::is_floating_point_v<Scalar>) return true;
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
            if(not mps->is_real()) throw except::runtime_error("state has imaginary part at mps position {}", mps->get_position());
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
template<typename T>
const MpsSite<Scalar> &StateFinite<Scalar>::get_mps_site(T pos) const {
    if constexpr(std::is_signed_v<T>)
        if(pos < 0) throw except::range_error("get_mps_site(pos): pos out of range: {}", pos);
    if(pos >= get_length<T>()) throw except::range_error("get_mps_site(pos): pos out of range: {}", pos);
    const auto &mps_ptr = *std::next(mps_sites.begin(), safe_cast<long>(pos));
    if(mps_ptr->template get_position<T>() != pos)
        throw except::range_error("get_mps_site(pos): mismatch pos {} != mps pos {}", pos, mps_ptr->template get_position<T>());
    return *mps_ptr;

    //    if(algo == AlgorithmType::fLBIT){
    //        // During fLBIT we can't assume the mps positions are in order since we could have swap operators.
    //        for(const auto & mps_ptr :  mps_sites ){
    //            if(mps_ptr->get_position<T>() == pos) return *mps_ptr;
    //        }
    //        throw except::runtime_error("get_mps_site(pos): pos {} not found", pos);
    //    }else{
    //        // There shouldn't be any swap operator, we can safely assume the mps positions are sorted
    //        const auto &mps_ptr = *std::next(mps_sites.begin(), safe_cast<long>(pos));
    //        if(mps_ptr->get_position<T>() != pos)
    //            throw except::range_error("get_mps_site(pos): mismatch pos {} != mps pos {}", pos, mps_ptr->get_position<T>());
    //        return *mps_ptr;
    //    }
}
template const MpsSite<> &StateFinite<>::get_mps_site(size_t pos) const;
template const MpsSite<> &StateFinite<>::get_mps_site(long pos) const;
template const MpsSite<> &StateFinite<>::get_mps_site(int pos) const;
template const MpsSite<> &StateFinite<>::get_mps_site(unsigned long long pos) const; // hsize_t

template<typename Scalar>
template<typename T>
MpsSite<Scalar> &StateFinite<Scalar>::get_mps_site(T pos) {
    return const_cast<MpsSite<Scalar> &>(std::as_const(*this).template get_mps_site<T>(pos));
}
template MpsSite<> &StateFinite<>::get_mps_site(size_t pos);
template MpsSite<> &StateFinite<>::get_mps_site(long pos);
template MpsSite<> &StateFinite<>::get_mps_site(int pos);
template MpsSite<> &StateFinite<>::get_mps_site(unsigned long long pos); // hsize_t

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
template<typename T>
Eigen::Tensor<T, 3> StateFinite<Scalar>::get_multisite_mps(const std::vector<size_t> &sites, bool use_cache) const {
    if(sites.empty()) throw except::runtime_error("No active sites on which to build a multisite mps tensor");
    auto                t_mps  = tid::tic_scope("gen_mps", tid::level::highest);
    auto                length = get_length<size_t>();
    Eigen::Tensor<T, 3> multisite_mps;
    for(auto &site : sites) {
        const auto mps_key = use_cache ? generate_cache_key(sites, site, "l2r") : "";
        const auto mps_cch = use_cache ? get_cached_mps<T>(mps_key) : std::nullopt;
        if(mps_cch.has_value()) {
            if constexpr(settings::debug_cache) tools::log->trace("multisite_mps: cache_hit: {}", mps_key);
            multisite_mps = mps_cch->get();
        } else {
            const auto &mps       = get_mps_site(site);
            auto        M         = mps.template get_M_as<T>();
            bool        prepend_L = mps.get_label() == "B" and site > 0 and site == sites.front();
            bool        append_L  = mps.get_label() == "A" and site + 1 < length and site == sites.back();
            if(prepend_L) {
                // In this case all sites are "B" and we need to prepend the "L" from the site on the left to make a normalized multisite mps
                if constexpr(settings::debug_state) tools::log->trace("Prepending L to B site {}", site);
                auto        t_prepend = tid::tic_scope("prepend", tid::level::higher);
                const auto &mps_left  = get_mps_site(site - 1); // mps_left is either AC or B
                const auto  L         = mps_left.isCenter() ? mps_left.template get_LC_as<T>() : mps_left.template get_L_as<T>();
                if(L.dimension(0) != M.dimension(1))
                    throw except::logic_error("get_multisite_mps<{}>: mismatching dimensions ({},{}): L (left) {} | M {}", mps_left.get_tag(), mps.get_tag(),
                                              sfinae::type_name<T>(), L.dimensions(), M.dimensions());
                M = tools::common::contraction::contract_bnd_mps(L, M);
            }
            if(append_L) {
                // In this case all sites are "A" and we need to append the "L" from the site on the right to make a normalized multisite mps
                if constexpr(settings::debug_state) tools::log->trace("Appending L to A site {}", site);
                auto        t_append  = tid::tic_scope("append", tid::level::higher);
                const auto &mps_right = get_mps_site(site + 1);
                const auto  L         = mps_right.template get_L_as<T>();
                if(L.dimension(0) != M.dimension(2))
                    throw except::logic_error("get_multisite_mps<{}>: mismatching dimensions: M {} | L (right) {}", sfinae::type_name<T>(), M.dimensions(),
                                              L.dimensions());
                M = tools::common::contraction::contract_mps_bnd(M, L);
            }

            if(&site == &sites.front()) { // First site
                multisite_mps = std::move(M);
            } else { // Next sites
                multisite_mps = tools::common::contraction::contract_mps_mps(multisite_mps, M);
            }

            if(use_cache and not append_L) {
                // If it is the last site, we may have closed off by appending L
                get_cache<T>().mps.emplace_back(std::make_pair(mps_key, multisite_mps)); // We know it is not present already
                shrink_cache();
            }
        }
    }
    if constexpr(settings::debug) {
        // Check the norm of the tensor on debug builds
        auto   t_dbg = tid::tic_scope("debug");
        double norm  = std::abs(tools::common::contraction::contract_mps_norm(multisite_mps));
        if constexpr(settings::debug_state) tools::log->trace("get_multisite_mps<{}>({}): norm ⟨ψ|ψ⟩ = {:.16f}", sfinae::type_name<T>(), sites, norm);
        if(std::abs(norm - 1) > settings::precision::max_norm_error) {
            tools::log->warn("get_multisite_mps<{}>({}): norm error |1-⟨ψ|ψ⟩| = {:.2e} > max_norm_error {:.2e}", sfinae::type_name<T>(), sites,
                             std::abs(norm - 1), settings::precision::max_norm_error);
            //                throw except::runtime_error("get_multisite_mps<fp64>({}): norm error |1-⟨ψ|ψ⟩| = {:.2e} > max_norm_error {:.2e}", sites,
            //                std::abs(norm - 1),
            //                                            settings::precision::max_norm_error);
        }
    }
    return multisite_mps;
    // }
}

template Eigen::Tensor<fp32, 3> StateFinite<>::get_multisite_mps<fp32>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<fp64, 3> StateFinite<>::get_multisite_mps<fp64>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<cx32, 3> StateFinite<>::get_multisite_mps<cx32>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<cx64, 3> StateFinite<>::get_multisite_mps<cx64>(const std::vector<size_t> &sites, bool use_cache) const;

template<typename Scalar>
template<typename T>
const Eigen::Tensor<T, 3> &StateFinite<Scalar>::get_multisite_mps() const {
    if(get_cache<T>().multisite_mps.has_value()) return get_cache<T>().multisite_mps.value();
    get_cache<T>().multisite_mps = get_multisite_mps<T>(active_sites);
    return get_cache<T>().multisite_mps.value();
}
template const Eigen::Tensor<fp32, 3> &StateFinite<>::get_multisite_mps<fp32>() const;
template const Eigen::Tensor<fp64, 3> &StateFinite<>::get_multisite_mps<fp64>() const;
template const Eigen::Tensor<cx32, 3> &StateFinite<>::get_multisite_mps<cx32>() const;
template const Eigen::Tensor<cx64, 3> &StateFinite<>::get_multisite_mps<cx64>() const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 2> StateFinite<Scalar>::get_reduced_density_matrix(const std::vector<size_t> &sites) const {
    auto t_rho        = tid::tic_scope("rho");
    auto cites        = num::range<size_t>(sites.front(), sites.back() + 1); // Contiguous list of all sites E.g. [012|6789] -> [012|345|6789]
    auto costs        = get_reduced_density_matrix_cost<T>(sites);
    auto min_cost_idx = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
    if constexpr(settings::debug_density_matrix)
        tools::log->trace("get_reduced_density_matrix: cost_t2b {} | cost_l2r {} | cost_r2l {}", costs[0], costs[1], costs[2]);
    // if(debug::mem_hwm_in_mb() > 10000) throw except::runtime_error("Exceeded 5G high water mark after min_cost_idx");
    // min_cost_idx = 0; // Disable side to side contractions for a while
    if(min_cost_idx == 0 /* top to bottom */) {
        // We have a contiguous set
        auto mps = get_multisite_mps<T>(sites, true);
        // if(debug::mem_hwm_in_mb() > 10000) throw except::runtime_error("Exceeded 5G high water mark after multisite mps");
        return tools::common::contraction::contract_mps_partial<std::array{1l, 2l}>(mps);
    } else {
        // We probably have a non-contiguous set like [0123]4567[89]
        // Note that for non-contiguous sets, we expect the front and back sites to be at the edges!
        // Therefore, it matters from which side we do the contraction: we want the smaller of the two non-contiguous parts to accumulate
        // the middle part of the system. For instance, on [0123]4567[89] we want to contract from the right, so that [89] contracts 4567 and then [0123].
        // The reason is that
        // * [0123] with 8 free spin indices with dim (2**8)=256 when contracting the middle sites
        // * [89] with 4 free spin indices with dim(2**4)=16
        // So if subsystem A has N more spins than B, then A costs 2**(2*N) times more than B to compute.
        auto &threads   = tenx::threads::get();
        auto  rho_temp  = Eigen::Tensor<T, 4>(); // Will accumulate the sites
        auto  rho_temp2 = Eigen::Tensor<T, 4>();
        auto  M         = Eigen::Tensor<T, 3>();

        // Decide to go from the left or from the right
        // auto site_mean = std::accumulate(sites.begin(), sites.end(), 0.5) / static_cast<double>(sites.size());
        // bool from_left = site_mean >= get_length<double>() / 2.0;
        if(min_cost_idx == 1 /* left to right */) {
            // tools::log->info("from left");
            for(const auto &i : cites) {
                // tools::log->info("contracting site {}", i);
                const auto &mps = get_mps_site(i);
                if(i == sites.front()) {
                    // Could be an A, AC or B. Either way we need the first site to include the left schmidt values
                    // If it is the only site, we also need it to include the right schmidt values.
                    bool use_multisite = mps.get_label() == "B" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    rho_temp.resize(std::array{dim[0], dim[0], dim[2], dim[2]});
                    rho_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({1}, {1})).shuffle(std::array{0, 2, 1, 3});
                } else {
                    // This site could be A, AC or B. Only A lacks schmidt values on the right, so we use multisite when the last site is A.
                    bool use_multisite = i == sites.back() and mps.get_label() == "A";
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto rho_dim       = rho_temp.dimensions();
                    bool do_trace      = std::find(sites.begin(), sites.end(), i) == sites.end();
                    if(do_trace) {
                        auto new_dim = std::array{rho_dim[0], rho_dim[1], mps_dim[2], mps_dim[2]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {1})).contract(M, tenx::idx({2, 3}, {1, 0}));
                    } else {
                        auto new_dim = std::array{rho_dim[0] * mps_dim[0], rho_dim[1] * mps_dim[0], mps_dim[2], mps_dim[2]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {1}))
                                                              .contract(M, tenx::idx({2}, {1}))
                                                              .shuffle(std::array{0, 2, 1, 4, 3, 5})
                                                              .reshape(new_dim);
                    }
                    rho_temp = std::move(rho_temp2);
                    if constexpr(settings::debug_cache) {
                        if(debug::mem_hwm_in_mb() > 10000) {
                            for(const auto &elem : get_cache<T>().mps) tools::log->info("from left: cache memory > 10000 MB: {}", elem.first);
                            // throw except::runtime_error("Exceeded 5G high water mark after rho l2r site {} | sites", i, cites);
                        }
                    }
                }
            }
        } else if(min_cost_idx == 2 /* left to right */) {
            // tools::log->info("from right");
            for(const auto &i : iter::reverse(cites)) {
                const auto &mps = get_mps_site(i);
                // tools::log->info("contracting site {}", i);
                if(i == sites.back()) {
                    // Could be an A, AC or B. Either way we need the last site to include the right schmidt values
                    // If it is the only site, we also need it to include the left schmidt values.
                    bool use_multisite = mps.get_label() == "A" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    rho_temp.resize(std::array{dim[0], dim[0], dim[1], dim[1]});
                    rho_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({2}, {2})).shuffle(std::array{0, 2, 1, 3});
                } else {
                    // This site could be A, AC or B. Only B lacks schmidt values on the left, so we use multisite when the first site is B.
                    bool use_multisite = i == sites.front() and mps.get_label() == "B";
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto rho_dim       = rho_temp.dimensions();
                    bool do_trace      = std::find(sites.begin(), sites.end(), i) == sites.end();
                    if(do_trace) {
                        auto new_dim = std::array{rho_dim[0], rho_dim[1], mps_dim[1], mps_dim[1]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {2})).contract(M, tenx::idx({2, 3}, {2, 0}));
                    } else {
                        auto new_dim = std::array{rho_dim[0] * mps_dim[0], rho_dim[1] * mps_dim[0], mps_dim[1], mps_dim[1]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {2}))
                                                              .contract(M, tenx::idx({2}, {2}))
                                                              .shuffle(std::array{2, 0, 4, 1, 3, 5})
                                                              .reshape(new_dim);
                    }
                    rho_temp = std::move(rho_temp2);
                    if constexpr(settings::debug_cache) {
                        if(debug::mem_hwm_in_mb() > 10000) {
                            for(const auto &elem : get_cache<T>().mps) tools::log->info("from right: cache memory > 10000 MB: {}", elem.first);
                            // throw except::runtime_error("Exceeded 5G high water mark after rho r2l site {} | sites", i, cites);
                        }
                    }
                }
            }
        }
        return rho_temp.trace(std::array{2, 3});
    }
}

template Eigen::Tensor<fp32, 2> StateFinite<>::get_reduced_density_matrix<fp32>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<fp64, 2> StateFinite<>::get_reduced_density_matrix<fp64>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<cx32, 2> StateFinite<>::get_reduced_density_matrix<cx32>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<cx64, 2> StateFinite<>::get_reduced_density_matrix<cx64>(const std::vector<size_t> &sites) const;

template<typename Scalar>
template<typename T>
std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost(const std::vector<size_t> &sites) const {
    auto t_rho         = tid::tic_scope("rho");
    auto cites         = num::range<size_t>(sites.front(), sites.back() + 1); // Contiguous list of all sites E.g. [012|6789] -> [012|345|6789]
    bool is_contiguous = sites == cites;
    // We have a contiguous set
    // Calculate the numerical cost for contracting top to bottom or side to side. Account for both the number of operations and the memory size
    auto ops_t2b = 0.0, mem_t2b = 0.0;
    auto ops_l2r = 0.0, mem_l2r = 0.0;
    auto ops_r2l = 0.0, mem_r2l = 0.0;
    if(is_contiguous) {
        auto bonds   = get_bond_dims(sites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
        auto chiL    = get_mps_site(sites.front()).get_chiL();
        auto chiR    = get_mps_site(sites.back()).get_chiR();
        auto max_chi = std::max(chiR, chiL);
        auto min_chi = std::min(chiR, chiL);
        auto spindim = std::pow(2.0, sites.size());

        mem_t2b = spindim * spindim; // The number of elements in the largest object that will be held in memory
        ops_t2b = spindim * spindim * static_cast<double>(max_chi * min_chi * min_chi + min_chi); // This is the last step, but we will add earlier steps below.
        for(const auto &[i, pos] : iter::enumerate(sites)) {
            auto key = generate_cache_key(sites, pos, "l2r");
            if(has_cached_mps<T>(key)) continue;
            // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
            if(i == 0) { // It will append either chiL or chiR, but we take the worst case scenario here
                if(sites.size() == 1) { ops_t2b += static_cast<double>(2l * chiL * chiR * std::max(chiL, chiR)); }
            } else {
                auto bondR = pos == sites.back() ? chiR : bonds[i];
                ops_t2b += std::pow(2.0, i + 1) * static_cast<double>(chiL * bonds[i - 1] * bondR);
                mem_t2b = std::max(mem_t2b, std::pow(2.0, i + 1) * static_cast<double>(chiL * bondR));
            }
        }
    }

    if(!is_contiguous or static_cast<double>(sizeof(T)) * mem_t2b / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem_t2b = std::numeric_limits<double>::infinity();
        ops_t2b = std::numeric_limits<double>::infinity();
    }

    auto bonds   = get_bond_dims(cites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
    auto chiL    = get_mps_site(cites.front()).get_chiL();
    auto chiR    = get_mps_site(cites.back()).get_chiR();
    auto spindim = 1.0;
    for(const auto &[i, pos] : iter::enumerate(cites)) {
        // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
        auto bondR = pos == sites.back() ? chiR : bonds.at(i);
        if(pos == sites.front()) {
            spindim *= 2;
            ops_l2r += spindim * spindim * static_cast<double>(chiL * bondR * bondR);
            mem_l2r = std::max(mem_l2r, spindim * spindim * static_cast<double>(bondR * bondR));
        } else {
            bool do_trace = std::find(sites.begin(), sites.end(), pos) == sites.end();
            if(do_trace) {
                ops_l2r += spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bonds[i - 1] * bondR); // Upper
                ops_l2r += spindim * spindim * static_cast<double>(4l * bonds[i - 1] * bondR * bondR);        // Lower part 1 of 2
                ops_l2r += spindim * spindim * static_cast<double>(2l * bondR * bondR);                       // Lower part 2 of 2
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bondR);
                auto tmp2 = spindim * spindim * static_cast<double>(bondR * bondR);
                mem_l2r   = std::max(mem_l2r, tmp1 + tmp2);

            } else {
                ops_l2r += spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bonds[i - 1] * bondR); // Upper
                ops_l2r += spindim * spindim * static_cast<double>(4l * bonds[i - 1] * bondR * bondR);        // Lower part
                mem_l2r   = std::max(mem_l2r, spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bondR));
                mem_l2r   = std::max(mem_l2r, spindim * spindim * static_cast<double>(4l * bondR * bondR));
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bondR);
                auto tmp2 = spindim * spindim * static_cast<double>(4l * bondR * bondR);
                mem_l2r   = std::max(mem_l2r, tmp1 + tmp2);
                spindim *= 2;
            }
        }
    }
    ops_l2r += spindim * spindim * static_cast<double>(chiR); // add the last contraction that closes the density matrix
    mem_l2r = std::max(mem_l2r, spindim * spindim);
    if(static_cast<double>(sizeof(T)) * mem_l2r / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem_l2r = std::numeric_limits<double>::infinity();
        ops_l2r = std::numeric_limits<double>::infinity();
    }

    spindim = 1.0;
    for(const auto &[i, pos] : iter::enumerate_reverse(cites)) {
        // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
        auto bondL = pos == sites.front() ? chiL : bonds.at(i - 1);
        if(pos == sites.back()) {
            spindim *= 2;
            ops_r2l += spindim * spindim * static_cast<double>(chiR * bondL * bondL);
            mem_r2l = std::max(mem_r2l, spindim * spindim * static_cast<double>(bondL * bondL));
        } else {
            bool do_trace = std::find(sites.begin(), sites.end(), pos) == sites.end();
            if(do_trace) {
                ops_r2l += spindim * spindim * static_cast<double>(2l * bondL * bonds[i] * bonds[i]); // Upper
                ops_r2l += spindim * spindim * static_cast<double>(4l * bondL * bondL * bonds[i]);    // Lower part 1 of 2
                ops_r2l += spindim * spindim * static_cast<double>(2l * bondL * bondL);               // Lower part 2 of 2
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bondL * bonds[i]);
                auto tmp2 = spindim * spindim * static_cast<double>(bondL * bondL);
                mem_r2l   = std::max(mem_r2l, tmp1 + tmp2);
            } else {
                ops_r2l += spindim * spindim * static_cast<double>(2l * bondL * bonds[i] * bonds[i]); // Upper
                ops_r2l += spindim * spindim * static_cast<double>(4l * bondL * bondL * bonds[i]);    // Lower part 1 of 2
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bondL * bonds[i]);
                auto tmp2 = spindim * spindim * static_cast<double>(4l * bondL * bondL);
                mem_r2l   = std::max(mem_r2l, tmp1 + tmp2);
                spindim *= 2;
            }
        }
    }
    ops_r2l += spindim * spindim * static_cast<double>(chiL); // add the last contraction that closes the density matrix
    mem_r2l = std::max(mem_r2l, spindim * spindim);
    if(static_cast<double>(sizeof(T)) * mem_r2l / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem_r2l = std::numeric_limits<double>::infinity();
        ops_r2l = std::numeric_limits<double>::infinity();
    }
    return std::array{ops_t2b + mem_t2b, ops_l2r + mem_l2r, ops_r2l + mem_r2l};
}
template std::array<double, 3> StateFinite<>::get_reduced_density_matrix_cost<fp32>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<>::get_reduced_density_matrix_cost<fp64>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<>::get_reduced_density_matrix_cost<cx32>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<>::get_reduced_density_matrix_cost<cx64>(const std::vector<size_t> &sites) const;

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
template<typename T>
StateFinite<Scalar>::optional_tensor4ref<T> StateFinite<Scalar>::load_trf_from_cache(const std::string &key) const {
    if(key.empty()) return {};
    auto it = std::find_if(get_cache<T>.trf.begin(), get_cache<T>.trf.end(), [&key](const auto &elem) -> bool { return elem.first == key; });
    if(it != get_cache<T>.trf.end()) {
        if constexpr(settings::debug_cache)
            tools::log->trace("load_trf_from_cache<{}>: cache_hit: {} | {} | {}", sfinae::type_name<T>(), key, it->second.dimensions());
        return std::cref(it->second);
    }
    return std::nullopt;
}

template<typename Scalar>
template<typename T>
StateFinite<Scalar>::optional_tensor4ref<T> StateFinite<Scalar>::load_trf_from_cache(const std::vector<size_t> &sites, const size_t pos,
                                                                                     std::string_view side) const {
    if(sites.empty()) return {};
    assert(pos >= sites.front());
    assert(pos <= sites.back());
    auto key = generate_cache_key(sites, pos, side);
    return load_trf_from_cache<T>(key);
}

template<typename Scalar>
template<typename T>
void StateFinite<Scalar>::save_trf_into_cache(const Eigen::Tensor<T, 4> &trf, const std::string &key) const {
    if(key.empty()) return;
    auto it = std::find_if(get_cache<T>().trf.rbegin(), get_cache<T>().trf.rend(), [&key](const auto &elem) -> bool { return elem.first == key; });
    if constexpr(settings::debug_cache) {
        // if(!cache.trf_real.contains(key)) tools::log->trace("save_trf_into_cache: key: {} | {}", key, trf.dimensions());
        if(it == get_cache<T>().trf.rend()) tools::log->trace("save_trf_into_cache<{}>: key: {} | {}", sfinae::type_name<T>(), key, trf.dimensions());
    }
    // cache.trf_real[key] = trf;
    if(it == get_cache<T>().trf.rend()) get_cache<T>().trf.emplace_back(std::make_pair(key, trf));
    shrink_cache();
}

template<typename Scalar>
template<typename T>
void StateFinite<Scalar>::save_trf_into_cache(const Eigen::Tensor<T, 4> &trf, const std::vector<size_t> &sites, size_t pos, std::string_view side) const {
    if(sites.empty()) return;
    if(side.empty()) return;
    assert(pos >= sites.front());
    assert(pos <= sites.back());
    auto key = generate_cache_key(sites, pos, side);
    if(side.starts_with('l') and key.ends_with("L]")) return;   // It cannot grow l2r any more
    if(side.starts_with('r') and key.starts_with("[L")) return; // It cannot grow r2l any more
    save_trf_into_cache<T>(trf, key);
}

template<typename Scalar>
template<typename T>
std::optional<typename StateFinite<Scalar>::TrfCacheEntry<T>> StateFinite<Scalar>::get_optimal_trf_from_cache(const std::vector<size_t> &sites,
                                                                                                              std::string_view           side) const {
    // We want to find the cheapest cache entry to start from, that has the most sites contracted into it already
    std::optional<TrfCacheEntry<T>> cacheEntry = std::nullopt;
    if(side.starts_with('l')) {
        for(const auto &posR : iter::reverse(sites)) { // posL is fixed, move the posR cursor towards posL
            auto key        = generate_cache_key(sites, posR, "l2r");
            auto nremaining = sites.back() - posR;
            auto ncontained = sites.size() - nremaining;
            // if(auto it = cache.trf_real.find(key); it != cache.trf_real.end()) {
            auto it = std::find_if(get_cache<T>().trf.rbegin(), get_cache<T>().trf.rend(), [&key](const auto &elem) -> bool { return elem.first == key; });
            if(it != get_cache<T>().trf.rend()) {
                if constexpr(settings::debug_cache)
                    tools::log->trace("get_optimal_trf_from_cache<{}>: cache hit: pos {} | {} | sites {}", sfinae::type_name<T>(), posR, side, sites);
                cacheEntry = TrfCacheEntry<T>{.pos        = posR,
                                              .side       = "l2r",
                                              .key        = key,
                                              .ncontained = ncontained,
                                              .nremaining = nremaining,
                                              .cost       = std::numeric_limits<double>::quiet_NaN(),
                                              .trf        = std::cref(it->second)};
                break;
            }
        }
    }
    if(side.starts_with('r')) {
        for(const auto &posL : sites) { // posR is fixed, move the posL cursor towards posR
            auto key        = generate_cache_key(sites, posL, "r2l");
            auto ncontained = sites.back() - posL + 1;
            auto nremaining = sites.size() - ncontained;
            // if(auto it = cache.trf_real.find(key); it != cache.trf_real.end()) {
            auto it = std::find_if(get_cache<T>().trf.rbegin(), get_cache<T>().trf.rend(), [&key](const auto &elem) -> bool { return elem.first == key; });
            if(it != get_cache<T>().trf.rend()) {
                if constexpr(settings::debug_cache)
                    tools::log->trace("get_optimal_trf_from_cache<{}>: cache hit: pos {} | {} | sites {}", sfinae::type_name<T>(), posL, side, sites);
                cacheEntry = TrfCacheEntry<T>{.pos        = posL,
                                              .side       = "r2l",
                                              .key        = key,
                                              .ncontained = ncontained,
                                              .nremaining = nremaining,
                                              .cost       = std::numeric_limits<double>::quiet_NaN(),
                                              .trf        = std::cref(it->second)};
                break;
            }
        }
    }
    cacheEntry->cost = get_transfer_matrix_cost(sites, side, cacheEntry);
    return cacheEntry;
}

template<typename Scalar>
template<typename T>
std::optional<typename StateFinite<Scalar>::TrfCacheEntry<T>> StateFinite<Scalar>::get_optimal_trf_from_cache(const std::vector<size_t> &sites) const {
    // We want to inspect the cache to find out which is the cheapest cache entry to start from.
    // We are looking for the longest cache entry (in number of sites) .
    std::optional<TrfCacheEntry<T>> cacheL = get_optimal_trf_from_cache<T>(sites, "l2r");
    std::optional<TrfCacheEntry<T>> cacheR = get_optimal_trf_from_cache<T>(sites, "r2l");

    // Return the cache entry that would be cheapest to complete
    if(cacheL.has_value() and cacheR.has_value()) {
        if constexpr(settings::debug_cache) tools::log->trace("get_optimal_trf_from_cache: comparing cacheL {} | cacheR {}", cacheL->key, cacheR->key);
        if(cacheL->cost < cacheR->cost) { return cacheL; }
        if(cacheR->cost < cacheL->cost) { return cacheR; }

        // Compare the number of sites remaining
        if(cacheL->nremaining < cacheR->nremaining) { return cacheL; }
        if(cacheR->nremaining < cacheL->nremaining) { return cacheR; }
        // In this case there are an equal number of remaining sites to contract.
        // The cheapest is the one with the smallest static dimension on the respective sides.
        if(cacheL->trf.get().dimension(0) < cacheR->trf.get().dimension(2)) { return cacheL; }
        if(cacheR->trf.get().dimension(2) < cacheL->trf.get().dimension(0)) { return cacheR; }
        // If those are also equal, then take the one with the smallest dynamic dimension.
        // If those are equal too... then it doesn't matter
        if(cacheL->trf.get().dimension(2) <= cacheR->trf.get().dimension(0)) { return cacheL; }
        if(cacheR->trf.get().dimension(0) <= cacheL->trf.get().dimension(2)) { return cacheR; }
    }
    if(cacheL.has_value()) return cacheL;
    if(cacheR.has_value()) return cacheR;
    return std::nullopt;
}

template<typename Scalar>
template<typename T>
StateFinite<Scalar>::optional_tensor3ref<T> StateFinite<Scalar>::get_cached_mps(const std::string &key) const {
    auto &cache = get_cache<T>();
    auto  it    = std::find_if(cache.mps.begin(), cache.mps.end(), [&](const auto &elem) -> bool { return elem.first == key; });
    if(it != cache.mps.end()) return std::cref(it->second);
    return std::nullopt;
}

template<typename Scalar>
template<typename T>
bool StateFinite<Scalar>::has_cached_mps(const std::string &key) const {
    auto &cache = get_cache<T>();
    auto  it    = std::find_if(cache.mps.rbegin(), cache.mps.rend(), [&](const auto &elem) -> bool { return elem.first == key; });
    if constexpr(settings::debug_cache)
        if(it != cache.mps.rend()) tools::log->trace("multisite_mps: cache_hit: {}", key);
    return it != cache.mps.rend();
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 2> StateFinite<Scalar>::get_transfer_matrix(const std::vector<size_t> &sites, std::string_view side) const {
    auto  t_trf        = tid::tic_scope("trf");
    auto  chiL         = get_mps_site(sites.front()).get_chiL();
    auto  chiR         = get_mps_site(sites.back()).get_chiR();
    auto  costs        = get_transfer_matrix_costs<T>(sites, side);
    auto  min_cost_idx = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
    auto &threads      = tenx::threads::get();
    if constexpr(settings::debug_transfer_matrix) tools::log->trace("cost_t2b {} | cost_s2s {} ({})", costs[0], costs[1], side);
    if(min_cost_idx == 0 /* top to bottom */) {
        if constexpr(settings::debug_transfer_matrix) tools::log->trace("from top");
        auto mps                  = get_multisite_mps<T>(sites, true);
        auto dim                  = std::array{mps.dimension(1) * mps.dimension(2), mps.dimension(1) * mps.dimension(2)};
        auto res                  = Eigen::Tensor<T, 2>(dim);
        res.device(*threads->dev) = mps.conjugate().contract(mps, tenx::idx({0}, {0})).reshape(dim);
        return res;
    } else {
        auto trf_temp = Eigen::Tensor<T, 4>(); // Will accumulate the sites
        auto M        = Eigen::Tensor<T, 3>();
        auto trf_tmp4 = Eigen::Tensor<T, 4>(); // Scratch space for contractions
        auto trf_tmp5 = Eigen::Tensor<T, 5>(); // Scratch space for contractions

        /*
         * We accumulate the transfer matrix such that it has the same index ordering going from left or right,
         * so that the caches are compatible without having to transpose
         *  left     right
         * 0-----2  0-----2
         *    |        |
         * 1-----3  1-----3
         *
         */
        auto trf_cache = get_optimal_trf_from_cache<T>(sites, side);
        // auto side      = trf_cache.has_value() ? trf_cache->side : (min_cost_idx == 1 ? "l2r" : "r2l");

        if(side.starts_with('l') /* left to right */) {
            if constexpr(settings::debug_transfer_matrix) tools::log->trace("from left");
            for(const auto &i : sites) {
                if(trf_cache.has_value() and i <= trf_cache->pos) {
                    if(i == trf_cache->pos) trf_temp = trf_cache->trf.get();
                    continue;
                }
                const auto &mps = get_mps_site(i);
                if(i == sites.front()) {
                    // Could be an A, AC or B. Either way we need the first site to include the left schmidt values
                    // If it is the only site, we also need it to include the right schmidt values.
                    bool use_multisite = mps.get_label() == "B" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    trf_temp.resize(std::array{dim[1], dim[1], dim[2], dim[2]});
                    trf_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).shuffle(std::array{0, 2, 1, 3});
                } else {
                    // This site could be A, AC or B. Only A lacks schmidt values on the right, so we use multisite when the last site is A.
                    bool use_multisite = i == sites.back() and mps.get_label() == "A";
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto trf_dim       = trf_temp.dimensions();
                    auto new_dim       = std::array{trf_dim[0], trf_dim[1], mps_dim[2], mps_dim[2]};
                    trf_tmp4.resize(new_dim);
                    trf_tmp4.device(*threads->dev) = trf_temp.contract(M.conjugate(), tenx::idx({2}, {1})).contract(M, tenx::idx({2, 3}, {1, 0}));
                    trf_temp                       = std::move(trf_tmp4);
                }
                save_trf_into_cache<T>(trf_temp, sites, i, "l2r");
            }
        } else if(side.starts_with('r') /* right to left */) {
            if constexpr(settings::debug_transfer_matrix) tools::log->trace("from right");
            for(const auto &i : iter::reverse(sites)) {
                if(trf_cache.has_value() and i >= trf_cache->pos) {
                    if(i == trf_cache->pos) trf_temp = trf_cache->trf.get();
                    continue;
                }
                const auto &mps = get_mps_site(i);
                if(i == sites.back()) {
                    // Could be an A, AC or B. Either way we need the last site to include the right schmidt values
                    // If it is the only site, we also need it to include the left schmidt values.
                    bool use_multisite = mps.get_label() == "A" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    trf_temp.resize(std::array{dim[1], dim[1], dim[2], dim[2]});
                    trf_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).shuffle(std::array{0, 2, 1, 3});

                    // auto dim = M.dimensions();
                    // trf_temp.resize(std::array{dim[2], dim[2], dim[1], dim[1]});
                    // trf_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).shuffle(std::array{1, 3, 0, 2});

                } else {
                    // This site could be A, AC or B. Only B lacks schmidt values on the left, so we use multisite when the first site is B.
                    bool use_multisite = i == sites.front() and mps.get_label() == "B";
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto trf_dim       = trf_temp.dimensions();
                    auto tm5_dim       = std::array{mps_dim[0], mps_dim[1], trf_dim[0], trf_dim[2], trf_dim[3]};
                    auto new_dim       = std::array{mps_dim[1], mps_dim[1], trf_dim[2], trf_dim[3]};
                    trf_tmp5.resize(tm5_dim);
                    trf_tmp5.device(*threads->dev) = M.contract(trf_temp, tenx::idx({2}, {1}));
                    trf_temp.resize(new_dim);
                    trf_temp.device(*threads->dev) = M.conjugate().contract(trf_tmp5, tenx::idx({0, 2}, {0, 2}));
                }
                save_trf_into_cache<T>(trf_temp, sites, i, "r2l");
            }
        }

        auto new_dim = std::array{chiL, chiR, chiL, chiR};
        trf_tmp4.resize(new_dim);
        trf_tmp4.device(*threads->dev) = trf_temp.shuffle(std::array{0, 2, 1, 3});
        return trf_tmp4.reshape(std::array{chiL * chiR, chiL * chiR});
    }
}

template<typename Scalar>
template<typename T>
double StateFinite<Scalar>::get_transfer_matrix_cost(const std::vector<size_t> &sites, std::string_view side,
                                                     const std::optional<TrfCacheEntry<T>> &trf_cache) const {
    auto   bonds = get_bond_dims(sites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
    auto   chiL  = get_mps_site(sites.front()).get_chiL();
    auto   chiR  = get_mps_site(sites.back()).get_chiR();
    double ops   = 0.0;
    double mem   = 0.0;
    if(side.starts_with('l')) {
        for(const auto &[i, pos] : iter::enumerate(sites)) {
            if(trf_cache.has_value() and side == trf_cache->side and pos <= trf_cache->pos) continue;
            // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
            if(pos == sites.front() and i < bonds.size()) {
                ops += static_cast<double>(chiL * chiL * bonds[i] * bonds[i] * 2); // The left-most site
                mem = std::max(mem, static_cast<double>(chiL * chiL * bonds[i] * bonds[i]));
            } else { // Appending sites in the interior
                auto bondR = pos == sites.back() ? chiR : bonds.at(i);
                ops += static_cast<double>(chiL * chiL * bonds[i - 1] * bonds[i - 1] * bondR * 2); // Upper mps
                ops += static_cast<double>(chiL * chiL * bonds[i - 1] * bondR * bondR * 4);        // Lower mps part 1 of 2
                ops += static_cast<double>(chiL * chiL * bondR * bondR * 2);                       // Lower mps part 2 of 2
                auto tmp1 = static_cast<double>(chiL * chiL * bonds[i - 1] * bondR * 2);
                auto tmp2 = std::max(tmp1, static_cast<double>(chiL * chiL * bondR * bondR * 4));
                auto tmp3 = std::max(tmp2, static_cast<double>(chiL * chiL * bondR * bondR));
                mem       = std::max(mem, tmp2 + tmp3);
            }
        }
    }
    if(side.starts_with('r')) {
        for(const auto &[i, pos] : iter::enumerate_reverse(sites)) {
            if(trf_cache.has_value() and side == trf_cache->side and pos >= trf_cache->pos) continue;
            // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
            if(pos == sites.back() and i - 1 < bonds.size()) {
                ops += static_cast<double>(chiR * chiR * bonds[i - 1] * bonds[i - 1] * 2); // The right-most site
                mem = std::max(mem, static_cast<double>(chiR * chiR * bonds[i - 1] * bonds[i - 1]));
            } else { // Appending sites in the interior
                auto bondL = pos == sites.front() ? chiL : bonds.at(i - 1);
                ops += static_cast<double>(chiR * chiR * bonds[i] * bonds[i] * bondL * 2); // Upper mps
                ops += static_cast<double>(chiR * chiR * bonds[i] * bondL * bondL * 4);    // Lower mps part 1 of 2
                ops += static_cast<double>(chiR * chiR * bondL * bondL * 2);               // Lower mps part 2 of 2
                auto tmp1 = static_cast<double>(chiR * chiR * bonds[i] * bondL * 2);
                auto tmp2 = std::max(tmp1, static_cast<double>(chiR * chiR * bondL * bondL * 4));
                auto tmp3 = std::max(tmp2, static_cast<double>(chiR * chiR * bondL * bondL));
                mem       = std::max(mem, tmp2 + tmp3);
            }
        }
    }
    if(static_cast<double>(sizeof(T)) * mem / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem = std::numeric_limits<double>::infinity();
        ops = std::numeric_limits<double>::infinity();
    }
    return ops + mem;
}

template<typename Scalar>
template<typename T>
std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs(const std::vector<size_t> &sites, std::string_view side) const {
    if(sites.empty()) throw except::logic_error("get_transfer_matrix_cost: sites is empty");
    auto cites = num::range<size_t>(sites.front(), sites.back() + 1); // Contiguous list of all sites
    if(sites != cites) throw except::logic_error("get_transfer_matrix_cost: sites is not contiguous: {}", sites);
    // We have a contiguous set
    auto bonds   = get_bond_dims(sites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
    auto chiL    = get_mps_site(sites.front()).get_chiL();
    auto chiR    = get_mps_site(sites.back()).get_chiR();
    auto spindim = std::pow(2.0, sites.size());
    // Calculate the numerical cost for contracting top to bottom or side to side
    auto ops_t2b = spindim * static_cast<double>(chiL * chiR * chiL * chiR); // This is the last step, but we will add earlier steps below.
    auto mem_t2b =
        static_cast<double>(chiL * chiR * chiL * chiR); // This is the number of elements in the last object we will compare with earlier steps below.
    for(size_t i = 0; i < sites.size(); ++i) {
        auto key = generate_cache_key(sites, sites.front() + i, "l2r");
        if(has_cached_mps<T>(key)) continue;
        if(i == 0 and sites.size() == 1) { // It will append either chiL or chiR but we take the worst case scenario here
            ops_t2b += static_cast<double>(2l * chiL * chiR * std::max(chiL, chiR));
            mem_t2b = std::max(mem_t2b, static_cast<double>(2l * chiL * chiR));
        }
        if(i + 1 < bonds.size()) { // TTT...T*T
            ops_t2b += std::pow(2.0, i + 1) * static_cast<double>(chiL * bonds[i] * bonds[i + 1] * 2l);
            mem_t2b = std::max(mem_t2b, std::pow(2.0, i + 1) * static_cast<double>(chiL * bonds[i + 1] * 2l));
        } else if(i + 1 == sites.size() and i == bonds.size()) { // The last site
            ops_t2b += std::pow(2.0, i + 1) * static_cast<double>(2l * chiL * bonds[i - 1] * chiR);
            mem_t2b = std::max(mem_t2b, std::pow(2.0, i + 1) * static_cast<double>(2l * chiL * chiR));
        }
    }
    auto cost_t2b = ops_t2b + mem_t2b;

    if(side.starts_with('l')) {
        auto trf_cacheL = get_optimal_trf_from_cache<T>(sites, "l2r");
        auto cost_l2r   = get_transfer_matrix_cost(sites, "l2r", trf_cacheL);
        if(trf_cacheL.has_value()) trf_cacheL->cost = cost_l2r;
        return std::array{cost_t2b, cost_l2r};
    } else if(side.starts_with('r')) {
        auto trf_cacheR = get_optimal_trf_from_cache<T>(sites, "r2l");
        auto cost_r2l   = get_transfer_matrix_cost(sites, "r2l", trf_cacheR);
        if(trf_cacheR.has_value()) trf_cacheR->cost = cost_r2l;
        return std::array{cost_t2b, cost_r2l};
    } else {
        throw except::logic_error("get_transfer_matrix_costs: invalid side: {}", side);
    }
}

template Eigen::Tensor<fp32, 2> StateFinite<>::get_transfer_matrix<fp32>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<fp64, 2> StateFinite<>::get_transfer_matrix<fp64>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<cx32, 2> StateFinite<>::get_transfer_matrix<cx32>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<cx64, 2> StateFinite<>::get_transfer_matrix<cx64>(const std::vector<size_t> &sites, std::string_view side) const;

template<typename Scalar>
double StateFinite<Scalar>::get_trf_cache_gbts() const {
    double size_fp32 = 0, size_fp64 = 0, size_cx32 = 0, size_cx64 = 0;
    for(const auto &elem : get_cache<fp32>().trf) size_fp32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<fp64>().trf) size_fp64 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx32>().trf) size_cx32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx64>().trf) size_cx64 += static_cast<double>(elem.second.size());

    size_fp32 *= 4.0 / std::pow(1024.0, 3.0);
    size_fp64 *= 8.0 / std::pow(1024.0, 3.0);
    size_cx32 *= 8.0 / std::pow(1024.0, 3.0);
    size_cx64 *= 16.0 / std::pow(1024.0, 3.0);
    return size_fp32 + size_fp64 + size_cx32 + size_cx64;
}

template<typename Scalar>
double StateFinite<Scalar>::get_mps_cache_gbts() const {
    double size_fp32 = 0, size_fp64 = 0, size_cx32 = 0, size_cx64 = 0;
    for(const auto &elem : get_cache<fp32>().mps) size_fp32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<fp64>().mps) size_fp64 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx32>().mps) size_cx32 += static_cast<double>(elem.second.size());
    for(const auto &elem : get_cache<cx64>().mps) size_cx64 += static_cast<double>(elem.second.size());

    size_fp32 *= 4.0 / std::pow(1024.0, 3.0);
    size_fp64 *= 8.0 / std::pow(1024.0, 3.0);
    size_cx32 *= 8.0 / std::pow(1024.0, 3.0);
    size_cx64 *= 16.0 / std::pow(1024.0, 3.0);
    return size_fp32 + size_fp64 + size_cx32 + size_cx64;
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
std::vector<double> StateFinite<Scalar>::get_truncation_errors() const {
    return tools::finite::measure::truncation_errors(*this);
}
template<typename Scalar>
std::vector<double> StateFinite<Scalar>::get_truncation_errors_active() const {
    return tools::finite::measure::truncation_errors_active(*this);
}
template<typename Scalar>
double StateFinite<Scalar>::get_truncation_error_active_max() const {
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
    measurements = MeasurementsStateFinite();
}

template<typename Scalar>
void StateFinite<Scalar>::clear_cache(LogPolicy logPolicy) const {
    if(logPolicy == LogPolicy::VERBOSE or (settings::debug and logPolicy == LogPolicy::DEBUG)) { tools::log->trace("Clearing state cache"); }
    cache_fp32 = Cache<fp32>();
    cache_fp64 = Cache<fp64>();
    cache_cx32 = Cache<cx32>();
    cache_cx64 = Cache<cx64>();
}

template<typename Scalar>
void StateFinite<Scalar>::shrink_cache() const {
    while(cache_fp32.mps.size() > max_mps_cache_size) cache_fp32.mps.pop_front();
    while(cache_fp64.mps.size() > max_mps_cache_size) cache_fp64.mps.pop_front();
    while(cache_cx32.mps.size() > max_mps_cache_size) cache_cx32.mps.pop_front();
    while(cache_cx64.mps.size() > max_mps_cache_size) cache_cx64.mps.pop_front();
    while(cache_fp32.trf.size() > max_trf_cache_size) cache_fp32.trf.pop_front();
    while(cache_fp64.trf.size() > max_trf_cache_size) cache_fp64.trf.pop_front();
    while(cache_cx32.trf.size() > max_trf_cache_size) cache_cx32.trf.pop_front();
    while(cache_cx64.trf.size() > max_trf_cache_size) cache_cx64.trf.pop_front();

    while(get_mps_cache_gbts() + get_trf_cache_gbts() > std::max(0.0, settings::precision::max_cache_gbts)) {
        /* clang-format off */
        using namespace settings;
        using namespace settings::precision;
        if(!cache_fp32.mps.empty()) {cache_fp32.mps.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del mps: {}", max_cache_gbts, cache_fp32.mps.front().first);}
        if(!cache_fp64.mps.empty()) {cache_fp64.mps.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del mps: {}", max_cache_gbts, cache_fp64.mps.front().first);}
        if(!cache_cx32.mps.empty()) {cache_cx32.mps.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del mps: {}", max_cache_gbts, cache_cx32.mps.front().first);}
        if(!cache_cx64.mps.empty()) {cache_cx64.mps.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del mps: {}", max_cache_gbts, cache_cx64.mps.front().first);}
        if(!cache_fp32.trf.empty()) {cache_fp32.trf.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del trf: {}", max_cache_gbts, cache_fp32.trf.front().first);}
        if(!cache_fp64.trf.empty()) {cache_fp64.trf.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del trf: {}", max_cache_gbts, cache_fp64.trf.front().first);}
        if(!cache_cx32.trf.empty()) {cache_cx32.trf.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del trf: {}", max_cache_gbts, cache_cx32.trf.front().first);}
        if(!cache_cx64.trf.empty()) {cache_cx64.trf.pop_front(); if constexpr(debug_cache) tools::log->trace("shrink_cache (max {}): del trf: {}", max_cache_gbts, cache_cx64.trf.front().first);}
        /* clang-format on */
    }
}

template<typename Scalar>
void StateFinite<Scalar>::tag_active_sites_normalized(bool tag) const {
    if(tag_normalized_sites.size() != get_length()) throw except::runtime_error("Cannot tag active sites, size mismatch in site list");
    for(auto &site : active_sites) tag_normalized_sites[site] = tag;
}

template<typename Scalar>
void StateFinite<Scalar>::tag_all_sites_normalized(bool tag) const {
    if(tag_normalized_sites.size() != get_length()) throw except::runtime_error("Cannot untag all sites, size mismatch in site list");
    tag_normalized_sites = std::vector<bool>(get_length(), tag);
}

template<typename Scalar>
void StateFinite<Scalar>::tag_site_normalized(size_t pos, bool tag) const {
    if(tag_normalized_sites.size() != get_length()) throw except::runtime_error("Cannot untag all sites, size mismatch in site list");
    tag_normalized_sites[pos] = tag;
}

template<typename Scalar>
bool StateFinite<Scalar>::is_normalized_on_all_sites() const {
    if(tag_normalized_sites.size() != get_length()) throw except::runtime_error("Cannot check normalization status on all sites, size mismatch in site list");
    // If all tags are false then we should definitely normalize:
    auto normalized_none = std::none_of(tag_normalized_sites.begin(), tag_normalized_sites.end(), [](bool v) { return v; });
    if(normalized_none) {
        tools::log->debug("{} normalized: false (none)", get_name());
        return false;
    }

    if constexpr(settings::debug) {
        auto normalized_some = std::any_of(tag_normalized_sites.begin(), tag_normalized_sites.end(), [](bool v) { return v; });
        if(normalized_some) {
            // In debug mode we check if the tags are truthful
            for(const auto &mps : mps_sites) {
                auto pos = mps->template get_position<size_t>();
                if(not tag_normalized_sites[pos]) {
                    if(mps->is_normalized(settings::precision::max_norm_error)) tag_normalized_sites[pos] = true;
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
        // We don't need this check if the tags already told us the state isn't normalized
        auto norm       = tools::finite::measure::norm(*this, false);
        normalized_fast = std::abs(norm - 1.0) <= settings::precision::max_norm_error;
        msg += fmt::format(" | fast {} {:.3e}", normalized_fast, norm);
    }
    if constexpr(settings::debug) {
        if(normalized_tags and normalized_fast) {
            auto norm       = tools::finite::measure::norm(*this, true);
            normalized_full = std::abs(norm - 1.0) <= settings::precision::max_norm_error;
            msg += fmt::format(" | full {} {:.3e}", normalized_full, norm);
        }
        if(normalized_tags and normalized_fast and normalized_full) {
            std::vector<long> site_list;
            for(const auto &mps : mps_sites) {
                if(not mps->is_normalized(settings::precision::max_norm_error)) { site_list.emplace_back(mps->template get_position<long>()); }
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
