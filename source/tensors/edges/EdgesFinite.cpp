#include "EdgesFinite.h"
#include "EdgesFinite.impl.h"
//
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/cast.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvPair.h"
#include "tensors/site/env/EnvVar.h"
#include "tools/common/log.h"

template class EdgesFinite<fp32>;
template class EdgesFinite<fp64>;
template class EdgesFinite<fp128>;
template class EdgesFinite<cx32>;
template class EdgesFinite<cx64>;
template class EdgesFinite<cx128>;

template<typename Scalar>
EdgesFinite<Scalar>::EdgesFinite() = default; // Can't initialize lists since we don't know the model size yet

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
EdgesFinite<Scalar>::~EdgesFinite() = default; // default dtor
template<typename Scalar>
EdgesFinite<Scalar>::EdgesFinite(EdgesFinite &&other) noexcept = default; // default move ctor
template<typename Scalar>
EdgesFinite<Scalar> &EdgesFinite<Scalar>::operator=(EdgesFinite &&other) noexcept = default; // default move assign
template<typename Scalar>
EdgesFinite<Scalar>::EdgesFinite(const EdgesFinite &other) : active_sites(other.active_sites) {
    eneL.clear();
    eneR.clear();
    varL.clear();
    varR.clear();
    eneL.reserve(other.eneL.size());
    eneR.reserve(other.eneR.size());
    varL.reserve(other.varL.size());
    varR.reserve(other.varR.size());
    for(const auto &other_eneL : other.eneL) eneL.emplace_back(std::make_unique<EnvEne<Scalar>>(*other_eneL));
    for(const auto &other_eneR : other.eneR) eneR.emplace_back(std::make_unique<EnvEne<Scalar>>(*other_eneR));
    for(const auto &other_varL : other.varL) varL.emplace_back(std::make_unique<EnvVar<Scalar>>(*other_varL));
    for(const auto &other_varR : other.varR) varR.emplace_back(std::make_unique<EnvVar<Scalar>>(*other_varR));
}

template<typename Scalar>
EdgesFinite<Scalar> &EdgesFinite<Scalar>::operator=(const EdgesFinite &other) {
    // check for self-assignment
    if(this != &other) {
        active_sites = other.active_sites;
        eneL.clear();
        eneR.clear();
        varL.clear();
        varR.clear();
        eneL.reserve(other.eneL.size());
        eneR.reserve(other.eneR.size());
        varL.reserve(other.varL.size());
        varR.reserve(other.varR.size());
        for(const auto &other_eneL : other.eneL) eneL.emplace_back(std::make_unique<EnvEne<Scalar>>(*other_eneL));
        for(const auto &other_eneR : other.eneR) eneR.emplace_back(std::make_unique<EnvEne<Scalar>>(*other_eneR));
        for(const auto &other_varL : other.varL) varL.emplace_back(std::make_unique<EnvVar<Scalar>>(*other_varL));
        for(const auto &other_varR : other.varR) varR.emplace_back(std::make_unique<EnvVar<Scalar>>(*other_varR));
    }
    return *this;
}

template<typename Scalar>
void EdgesFinite<Scalar>::initialize(size_t model_size) {
    for(size_t pos = 0; pos < model_size; pos++) {
        eneL.emplace_back(std::make_unique<EnvEne<Scalar>>(pos, "L", "ene"));
        varL.emplace_back(std::make_unique<EnvVar<Scalar>>(pos, "L", "var"));
        eneR.emplace_back(std::make_unique<EnvEne<Scalar>>(pos, "R", "ene"));
        varR.emplace_back(std::make_unique<EnvVar<Scalar>>(pos, "R", "var"));
    }
}

template<typename Scalar>
size_t EdgesFinite<Scalar>::get_length() const {
    if(not num::all_equal(eneL.size(), eneR.size(), varL.size(), varR.size()))
        throw std::runtime_error(
            fmt::format("Size mismatch in environments: eneL {} | eneR {} | varL {} | varR {}", eneL.size(), eneR.size(), varL.size(), varR.size()));
    return eneL.size();
}

/* clang-format off */
template<typename Scalar>bool EdgesFinite<Scalar>::is_real() const {
    for(const auto &env : eneL) if(not env->is_real()) return false;
    for(const auto &env : eneR) if(not env->is_real()) return false;
    for(const auto &env : varL) if(not env->is_real()) return false;
    for(const auto &env : varR) if(not env->is_real()) return false;
    return true;
}

template<typename Scalar>bool EdgesFinite<Scalar>::has_nan() const {
    for(const auto &env : eneL) if(env->has_nan()) return true;
    for(const auto &env : eneR) if(env->has_nan()) return true;
    for(const auto &env : varL) if(env->has_nan()) return true;
    for(const auto &env : varR) if(env->has_nan()) return true;
    return false;
}

template<typename Scalar>void EdgesFinite<Scalar>::assert_validity() const {
    for(const auto &env : eneL) env->assert_validity();
    for(const auto &env : eneR) env->assert_validity();
    for(const auto &env : varL) env->assert_validity();
    for(const auto &env : varR) env->assert_validity();

    if(settings::model::model_type == ModelType::ising_sdual) {
        for(const auto &env : eneL) if(not env->is_real()) throw except::runtime_error("eneL has imaginary part at position {}", env->get_position());
        for(const auto &env : eneR) if(not env->is_real()) throw except::runtime_error("eneR has imaginary part at position {}", env->get_position());
        for(const auto &env : varL) if(not env->is_real()) throw except::runtime_error("varL has imaginary part at position {}", env->get_position());
        for(const auto &env : varR) if(not env->is_real()) throw except::runtime_error("varR has imaginary part at position {}", env->get_position());
    }
}
/* clang-format on */

template<typename Scalar>
void EdgesFinite<Scalar>::eject_edges_inactive_ene(std::optional<std::vector<size_t>> sites) {
    if(not sites) sites = active_sites;
    if(not sites or sites->empty()) {
        throw std::runtime_error("Could not eject inactive ene edges: There are no active sites");
        // If there are no active sites we may just as well
        // eject everything and let the next rebuild take
        // care of it.
        eject_edges_all();
        return;
    }
    for(auto &env : eneL)
        if(env->get_position() > sites->front()) env->clear();
    for(auto &env : eneR)
        if(env->get_position() < sites->back()) env->clear();
}

template<typename Scalar>
void EdgesFinite<Scalar>::eject_edges_inactive_var(std::optional<std::vector<size_t>> sites) {
    if(not sites) sites = active_sites;
    if(not sites or sites->empty()) {
        throw std::runtime_error("Could not eject inactive var edges: There are no active sites");
        // If there are no active sites we may just as well
        // eject everything and let the next rebuild take
        // care of it.
        eject_edges_all();
        return;
    }
    for(auto &env : varL)
        if(env->get_position() > sites->front()) env->clear();
    for(auto &env : varR)
        if(env->get_position() < sites->back()) env->clear();
}

template<typename Scalar>
void EdgesFinite<Scalar>::eject_edges_inactive(std::optional<std::vector<size_t>> sites) {
    eject_edges_inactive_ene(sites);
    eject_edges_inactive_var(sites);
}

template<typename Scalar>
void EdgesFinite<Scalar>::eject_edges_all_ene() {
    for(auto &env : eneL) env->clear();
    for(auto &env : eneR) env->clear();
}
template<typename Scalar>
void EdgesFinite<Scalar>::eject_edges_all_var() {
    for(auto &env : varL) env->clear();
    for(auto &env : varR) env->clear();
}
template<typename Scalar>
void EdgesFinite<Scalar>::eject_edges_all() {
    eject_edges_all_ene();
    eject_edges_all_var();
}

template<typename Scalar>
const EnvEne<Scalar> &EdgesFinite<Scalar>::get_env_eneL(size_t pos) const {
    if(pos >= get_length()) throw except::range_error("get_env_eneL(pos:{}): pos is out of range | system size {}", pos, get_length());
    if(pos >= eneL.size()) throw except::range_error("get_env_eneL(pos:{}): pos is out of range | eneL.size() == {}", pos, eneL.size());
    const auto &env = **std::next(eneL.begin(), safe_cast<long>(pos));
    if(env.get_position() != pos) throw except::logic_error("get_env_eneL(pos:{}): position mismatch {}", pos, env.get_position());
    return env;
}
template<typename Scalar>
const EnvEne<Scalar> &EdgesFinite<Scalar>::get_env_eneR(size_t pos) const {
    if(pos >= get_length()) throw except::range_error("get_env_eneR(pos:{}): pos is out of range | system size {}", pos, get_length());
    if(pos >= eneR.size()) throw except::range_error("get_env_eneR(pos:{}): pos is out of range | eneR.size() == {}", pos, eneR.size());
    const auto &env = **std::next(eneR.begin(), safe_cast<long>(pos));
    if(env.get_position() != pos) throw except::logic_error("get_env_eneR(pos:{}): position mismatch {}", pos, env.get_position());
    return env;
}
template<typename Scalar>
const EnvVar<Scalar> &EdgesFinite<Scalar>::get_env_varL(size_t pos) const {
    if(pos >= get_length()) throw except::range_error("get_env_varL(pos:{}): pos is out of range | system size {}", pos, get_length());
    if(pos >= varL.size()) throw except::range_error("get_env_varL(pos:{}): pos is out of range | varL.size() == {}", pos, varL.size());
    const auto &env = **std::next(varL.begin(), safe_cast<long>(pos));
    if(env.get_position() != pos) throw except::logic_error("get_env_varL(pos:{}): position mismatch {}", pos, env.get_position());
    return env;
}
template<typename Scalar>
const EnvVar<Scalar> &EdgesFinite<Scalar>::get_env_varR(size_t pos) const {
    if(pos >= get_length()) throw except::range_error("get_env_varR(pos:{}): pos is out of range | system size {}", pos, get_length());
    if(pos >= varR.size()) throw except::range_error("get_env_varR(pos:{}): pos is out of range | varR.size() == {}", pos, varR.size());
    const auto &env = **std::next(varR.begin(), safe_cast<long>(pos));
    if(env.get_position() != pos) throw except::logic_error("get_env_varR(pos:{}): position mismatch {}", pos, env.get_position());
    return env;
}

template<typename Scalar>
EnvEne<Scalar> &EdgesFinite<Scalar>::get_env_eneL(size_t pos) {
    return const_cast<EnvEne<Scalar> &>(std::as_const(*this).get_env_eneL(pos));
}
template<typename Scalar>
EnvEne<Scalar> &EdgesFinite<Scalar>::get_env_eneR(size_t pos) {
    return const_cast<EnvEne<Scalar> &>(std::as_const(*this).get_env_eneR(pos));
}
template<typename Scalar>
EnvVar<Scalar> &EdgesFinite<Scalar>::get_env_varL(size_t pos) {
    return const_cast<EnvVar<Scalar> &>(std::as_const(*this).get_env_varL(pos));
}
template<typename Scalar>
EnvVar<Scalar> &EdgesFinite<Scalar>::get_env_varR(size_t pos) {
    return const_cast<EnvVar<Scalar> &>(std::as_const(*this).get_env_varR(pos));
}

template<typename Scalar>
env_pair<const EnvEne<Scalar> &> EdgesFinite<Scalar>::get_ene_active() const {
    if(active_sites.empty()) throw std::logic_error("get_ene_active: no active sites");
    auto posL = active_sites.front();
    auto posR = active_sites.back();
    return {get_env_eneL(posL), get_env_eneR(posR)};
}
template<typename Scalar>
env_pair<const EnvVar<Scalar> &> EdgesFinite<Scalar>::get_var_active() const {
    if(active_sites.empty()) throw std::logic_error("get_var_active: no active sites");
    auto posL = active_sites.front();
    auto posR = active_sites.back();
    return {get_env_varL(posL), get_env_varR(posR)};
}
template<typename Scalar>
env_pair<EnvEne<Scalar> &> EdgesFinite<Scalar>::get_ene_active() {
    if(active_sites.empty()) throw std::logic_error("get_ene_active: no active sites");
    auto posL = active_sites.front();
    auto posR = active_sites.back();
    return {get_env_eneL(posL), get_env_eneR(posR)};
}
template<typename Scalar>
env_pair<EnvVar<Scalar> &> EdgesFinite<Scalar>::get_var_active() {
    if(active_sites.empty()) throw std::logic_error("get_var_active: no active sites");
    auto posL = active_sites.front();
    auto posR = active_sites.back();
    return {get_env_varL(posL), get_env_varR(posR)};
}

template<typename Scalar>
env_pair<const EnvEne<Scalar> &> EdgesFinite<Scalar>::get_env_ene(size_t posL, size_t posR) const {
    if(posL > posR) throw except::range_error("get_env_ene(posL,posR): posL is out of range posL {} > posR {}", posL, posR);
    return {get_env_eneL(posL), get_env_eneR(posR)};
}
template<typename Scalar>
env_pair<const EnvVar<Scalar> &> EdgesFinite<Scalar>::get_env_var(size_t posL, size_t posR) const {
    if(posL > posR) throw except::range_error("get_env_var(posL,posR): posL is out of range posL {} > posR {}", posL, posR);
    return {get_env_varL(posL), get_env_varR(posR)};
}

template<typename Scalar>
env_pair<EnvEne<Scalar> &> EdgesFinite<Scalar>::get_env_ene(size_t posL, size_t posR) {
    if(posL > posR) throw except::range_error("get_env_ene(posL,posR): posL is out of range posL {} > posR {}", posL, posR);
    return {get_env_eneL(posL), get_env_eneR(posR)};
}
template<typename Scalar>
env_pair<EnvVar<Scalar> &> EdgesFinite<Scalar>::get_env_var(size_t posL, size_t posR) {
    if(posL > posR) throw except::range_error("get_env_var(posL,posR): posL is out of range posL {} > posR {}", posL, posR);
    return {get_env_varL(posL), get_env_varR(posR)};
}

template<typename Scalar>
env_pair<const EnvEne<Scalar> &> EdgesFinite<Scalar>::get_env_ene(size_t pos) const {
    return {get_env_eneL(pos), get_env_eneR(pos)};
}
template<typename Scalar>
env_pair<const EnvVar<Scalar> &> EdgesFinite<Scalar>::get_env_var(size_t pos) const {
    return {get_env_varL(pos), get_env_varR(pos)};
}
template<typename Scalar>
env_pair<EnvEne<Scalar> &> EdgesFinite<Scalar>::get_env_ene(size_t pos) {
    return {get_env_eneL(pos), get_env_eneR(pos)};
}
template<typename Scalar>
env_pair<EnvVar<Scalar> &> EdgesFinite<Scalar>::get_env_var(size_t pos) {
    return {get_env_varL(pos), get_env_varR(pos)};
}

template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_env_ene_blk(size_t posL, size_t posR) const {
    return {get_env_ene(posL).L.get_block(), get_env_ene(posR).R.get_block()};
}
template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_env_var_blk(size_t posL, size_t posR) const {
    return {get_env_var(posL).L.get_block(), get_env_var(posR).R.get_block()};
}

template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_env_ene_blk(size_t posL, size_t posR) {
    return {get_env_ene(posL).L.get_block(), get_env_ene(posR).R.get_block()};
}

template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_env_var_blk(size_t posL, size_t posR) {
    return {get_env_var(posL).L.get_block(), get_env_var(posR).R.get_block()};
}

template<typename Scalar>
env_pair<const EnvEne<Scalar> &> EdgesFinite<Scalar>::get_multisite_env_ene(std::optional<std::vector<size_t>> sites) const {
    if(not sites) sites = active_sites;
    if(sites.value().empty()) throw std::runtime_error("Could not get edges: active site list is empty");
    return get_env_ene(sites.value().front(), sites.value().back());
}

template<typename Scalar>
env_pair<const EnvVar<Scalar> &> EdgesFinite<Scalar>::get_multisite_env_var(std::optional<std::vector<size_t>> sites) const {
    if(not sites) sites = active_sites;
    if(sites.value().empty()) throw std::runtime_error("Could not get edges: active site list is empty");
    return get_env_var(sites.value().front(), sites.value().back());
}

template<typename Scalar>
env_pair<EnvEne<Scalar> &> EdgesFinite<Scalar>::get_multisite_env_ene(std::optional<std::vector<size_t>> sites) {
    if(not sites) sites = active_sites;
    if(sites.value().empty()) throw std::runtime_error("Could not get edges: active site list is empty");
    return get_env_ene(sites.value().front(), sites.value().back());
}

template<typename Scalar>
env_pair<EnvVar<Scalar> &> EdgesFinite<Scalar>::get_multisite_env_var(std::optional<std::vector<size_t>> sites) {
    if(not sites) sites = active_sites;
    if(sites.value().empty()) throw std::runtime_error("Could not get edges: active site list is empty");
    return get_env_var(sites.value().front(), sites.value().back());
}

template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_multisite_env_ene_blk(std::optional<std::vector<size_t>> sites) const {
    auto envs = get_multisite_env_ene(std::move(sites));
    return {envs.L.get_block(), envs.R.get_block()};
}

template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_multisite_env_var_blk(std::optional<std::vector<size_t>> sites) const {
    auto envs = get_multisite_env_var(std::move(sites));
    return {envs.L.get_block(), envs.R.get_block()};
}

template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_multisite_env_ene_blk(std::optional<std::vector<size_t>> sites) {
    auto envs = get_multisite_env_ene(std::move(sites));
    return {envs.L.get_block(), envs.R.get_block()};
}

template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3> &> EdgesFinite<Scalar>::get_multisite_env_var_blk(std::optional<std::vector<size_t>> sites) {
    auto envs = get_multisite_env_var(std::move(sites));
    return {envs.L.get_block(), envs.R.get_block()};
}

template<typename Scalar>
std::pair<std::vector<size_t>, std::vector<size_t>> EdgesFinite<Scalar>::get_active_ids() const {
    std::pair<std::vector<size_t>, std::vector<size_t>> ids;
    auto                                               &ene_ids = ids.first;
    auto                                               &var_ids = ids.second;
    ene_ids = {get_env_eneL(active_sites.front()).get_unique_id(), get_env_eneR(active_sites.back()).get_unique_id()};
    var_ids = {get_env_varL(active_sites.front()).get_unique_id(), get_env_varR(active_sites.back()).get_unique_id()};
    return ids;
}

template<typename Scalar>
std::array<long, 3> EdgesFinite<Scalar>::get_dims_eneL(size_t pos) const {
    return get_env_eneL(pos).dimensions();
}
template<typename Scalar>
std::array<long, 3> EdgesFinite<Scalar>::get_dims_eneR(size_t pos) const {
    return get_env_eneR(pos).dimensions();
}
template<typename Scalar>
std::array<long, 3> EdgesFinite<Scalar>::get_dims_varL(size_t pos) const {
    return get_env_varL(pos).dimensions();
}
template<typename Scalar>
std::array<long, 3> EdgesFinite<Scalar>::get_dims_varR(size_t pos) const {
    return get_env_varR(pos).dimensions();
}