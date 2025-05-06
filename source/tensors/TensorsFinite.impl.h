#pragma once
#include "TensorsFinite.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/sfinae.h"
#include "math/cast.h"
#include "math/num.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/env.h"
#include "tools/finite/env/BondExpansionResult.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/measure/spin.h"
#include "tools/finite/mpo.h"
#include "tools/finite/mps.h"
#include "tools/finite/multisite.h"
#include "tools/finite/ops.h"

//
#include "math/linalg.h"
#include <tools/finite/opt_meta.h>



template<typename T>
Eigen::Tensor<T, 2> contract_mpo_env(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 3> &envL, const Eigen::Tensor<T, 3> &envR) {
    /*!
     *      |-------- 1               2--------|
     *      |                 0                |
     *      |                 |                |
     *   [ envL ]---- j  j--[mpo]--k  k ----[ envR ]
     *      |                 |                |
     *      |                 0                |
     *      |-------- 1               2 -------|
     */

    long                dim0_up = mpo.dimension(2);
    long                dim1_up = envL.dimension(0);
    long                dim2_up = envR.dimension(0);
    long                dim0_dn = mpo.dimension(3);
    long                dim1_dn = envL.dimension(1);
    long                dim2_dn = envR.dimension(1);
    std::array<long, 2> dims    = {dim0_up * dim1_up * dim2_up, dim0_dn * dim1_dn * dim2_dn};
    auto                t_con   = tid::tic_token("contract");
    Eigen::Tensor<T, 2> ham(dims);
    auto               &threads = tenx::threads::get();
    ham.device(*threads->dev) =
        envL.contract(mpo, tenx::idx({2}, {0})).contract(envR, tenx::idx({2}, {2})).shuffle(tenx::array6{3, 1, 5, 2, 0, 4}).reshape(dims);
    return ham;
}


template<typename Scalar>
TensorsFinite<Scalar>::TensorsFinite()
    : state(std::make_unique<StateFinite<Scalar>>()), model(std::make_unique<ModelFinite<Scalar>>()), edges(std::make_unique<EdgesFinite<Scalar>>()) {
    tools::log->trace("Constructing TensorsFinite");
}

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
TensorsFinite<Scalar>::~TensorsFinite() = default; // default dtor
template<typename Scalar>
TensorsFinite<Scalar>::TensorsFinite(TensorsFinite &&other) = default; // default move ctor
template<typename Scalar>
TensorsFinite<Scalar> &TensorsFinite<Scalar>::operator=(TensorsFinite &&other) = default; // default move assign

template<typename Scalar>
TensorsFinite<Scalar>::TensorsFinite(const TensorsFinite &other)
    : cache_fp32(other.cache_fp32),   //
      cache_fp64(other.cache_fp64),   //
      cache_fp128(other.cache_fp128), //
      cache_cx32(other.cache_cx32),   //
      cache_cx64(other.cache_cx64),   //
      cache_cx128(other.cache_cx128), //
      state(std::make_unique<StateFinite<Scalar>>(*other.state)), model(std::make_unique<ModelFinite<Scalar>>(*other.model)),
      edges(std::make_unique<EdgesFinite<Scalar>>(*other.edges)), active_sites(other.active_sites), measurements(other.measurements) {}
template<typename Scalar>
TensorsFinite<Scalar> &TensorsFinite<Scalar>::operator=(const TensorsFinite &other) {
    // check for self-assignment
    if(this != &other) {
        cache_fp32   = other.cache_fp32;
        cache_fp64   = other.cache_fp64;
        cache_cx32   = other.cache_cx32;
        cache_cx64   = other.cache_cx64;
        cache_cx128  = other.cache_cx128;
        state        = std::make_unique<StateFinite<Scalar>>(*other.state);
        model        = std::make_unique<ModelFinite<Scalar>>(*other.model);
        edges        = std::make_unique<EdgesFinite<Scalar>>(*other.edges);
        active_sites = other.active_sites;
        measurements = other.measurements;
        sync_active_sites();
    }
    return *this;
}

template<typename Scalar>
TensorsFinite<Scalar>::TensorsFinite(AlgorithmType algo_type, ModelType model_type, size_t model_size, long position) : TensorsFinite() {
    initialize(algo_type, model_type, model_size, position);
}

/* clang-format off */
template<typename Scalar> StateFinite<Scalar>       &TensorsFinite<Scalar>::get_state() { return *state; }
template<typename Scalar> ModelFinite<Scalar>       &TensorsFinite<Scalar>::get_model() { return *model; }
template<typename Scalar> EdgesFinite<Scalar>       &TensorsFinite<Scalar>::get_edges() { return *edges; }
template<typename Scalar> const StateFinite<Scalar> &TensorsFinite<Scalar>::get_state() const { return *state; }
template<typename Scalar> const ModelFinite<Scalar> &TensorsFinite<Scalar>::get_model() const { return *model; }
template<typename Scalar> const EdgesFinite<Scalar> &TensorsFinite<Scalar>::get_edges() const { return *edges; }
/* clang-format on */

template<typename Scalar>
void TensorsFinite<Scalar>::initialize(AlgorithmType algo_type, ModelType model_type, size_t model_size, long position) {
    tools::log->debug("Initializing tensors: algorithm [{}] | model [{}] | sites [{}] | position [{}]", enum2sv(algo_type), enum2sv(model_type), model_size,
                      position);
    state->initialize(algo_type, model_size, position);
    model->initialize(model_type, model_size);
    edges->initialize(model_size);
}

template<typename Scalar>
void TensorsFinite<Scalar>::initialize_model() {
    auto tic = tid::tic_scope("init_model");
    model->randomize();
    rebuild_mpo();
    rebuild_mpo_squared();
    compress_mpo_squared();
}

template<typename Scalar>
void TensorsFinite<Scalar>::initialize_state(ResetReason reason, StateInit state_init, StateInitType state_type, std::string_view axis, bool use_eigenspinors,
                                             long bond_lim, std::string &pattern) {
    state->clear_measurements();
    tools::log->debug("Initializing state [{}] to [{}] | Reason [{}] | Type [{}] | Sector [{}] | bond_lim {} | eigspinors {} | pattern {}", state->get_name(),
                      enum2sv(state_init), enum2sv(reason), enum2sv(state_type), axis, bond_lim, use_eigenspinors, pattern);

    tools::log->debug("Initializing state - Before: norm {:.16f} | spin components {::+.16f}", fp(tools::finite::measure::norm(get_state())),
                      fv(tools::finite::measure::spin_components(get_state())));

    tools::finite::mps::initialize_state(get_state(), state_init, state_type, axis, use_eigenspinors, bond_lim, pattern);
    state->assert_validity();
    tools::log->debug("Initializing state - After : norm {:.16f} | spin components {::+.16f}", fp(tools::finite::measure::norm(get_state())),
                      fv(tools::finite::measure::spin_components(get_state())));
}

template<typename Scalar>
void TensorsFinite<Scalar>::normalize_state(std::optional<svd::config> svd_cfg, NormPolicy norm_policy) {
    // Normalize if unity was lost for some reason (numerical error buildup)
    bool modified = tools::finite::mps::normalize_state(get_state(), svd_cfg, norm_policy);
    if(modified) {
        state->clear_cache();
        clear_measurements();
    }
}

template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> TensorsFinite<Scalar>::get_multisite_env_ene_blk() const {
    return get_edges().get_multisite_env_ene_blk();
}
template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> TensorsFinite<Scalar>::get_multisite_env_var_blk() const {
    return get_edges().get_multisite_env_var_blk();
}

template<typename Scalar>
void TensorsFinite<Scalar>::project_to_nearest_axis(std::string_view axis, std::optional<svd::config> svd_cfg) {
    auto sign = tools::finite::ops::project_to_nearest_axis(get_state(), axis, svd_cfg);
    if(sign != 0) clear_cache();
}

template<typename Scalar>
struct DebugStatus {
    using RealScalar                                        = decltype(std::real(std::declval<Scalar>()));
    RealScalar                                          ene = std::numeric_limits<RealScalar>::quiet_NaN();
    RealScalar                                          red = std::numeric_limits<RealScalar>::quiet_NaN();
    RealScalar                                          var = std::numeric_limits<RealScalar>::quiet_NaN();
    std::string                                         tag;
    std::pair<std::vector<size_t>, std::vector<size_t>> env_ids;
    std::vector<size_t>                                 mpo_ids;

    [[nodiscard]] std::string msg() const {
        std::string msg;
        msg.append(fmt::format("Energy {:<20} = {:>20.16f}\n", tag, fp(ene)));
        msg.append(fmt::format("Shift  {:<20} = {:>20.16f}\n", tag, fp(red)));
        msg.append(fmt::format("σ²H    {:<20} = {:>20.16f}\n", tag, fp(var)));
        return msg;
    }
    void print() const {
        tools::log->debug("Energy   [{:<20}] = {:>20.16f}", tag, fp(ene));
        tools::log->debug("Shift    [{:<20}] = {:>20.16f}", tag, fp(red));
        tools::log->debug("σ²H      [{:<20}] = {:>20.16f}", tag, fp(var));
    }
};

template<typename Scalar>
void TensorsFinite<Scalar>::set_parity_shift_mpo(OptRitz ritz, std::string_view axis) {
    if(ritz == OptRitz::NONE or axis.empty()) {
        model->set_parity_shift_mpo(ritz, 0, ""); // reset the parity shift
        return;
    }
    if(not qm::spin::half::is_valid_axis(axis)) return;
    auto sign = qm::spin::half::get_sign(axis);
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(sign == 0) {
        tools::log->info("set_parity_shift_mpo: no sector sign given for the target axis: taking the closest sector");
        sign = tools::finite::measure::spin_sign(get_state(), axis);
    }
    auto parity_shift_new = std::make_tuple(ritz, sign, axus);
    auto parity_shift_old = model->get_parity_shift_mpo();
    if(parity_shift_new == parity_shift_old) {
        tools::log->debug("set_parity_shift_mpo: not needed -- parity shift is already [{} {} {}]", enum2sv(ritz), sign, axus);
        return;
    }
    model->set_parity_shift_mpo(ritz, sign, axus); // will ignore the sign on the axis string if present
}

template<typename Scalar>
void TensorsFinite<Scalar>::set_parity_shift_mpo_squared(std::string_view axis) {
    if(not qm::spin::half::is_valid_axis(axis)) return;
    auto sign = qm::spin::half::get_sign(axis);
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(sign == 0) {
        tools::log->info("set_parity_shift_mpo_squared: no sector sign given for the target axis: taking the closest sector");
        sign = tools::finite::measure::spin_sign(get_state(), axis);
    }
    auto parity_shift_new = std::make_pair(sign, axus);
    auto parity_shift_old = model->get_parity_shift_mpo_squared();
    if(parity_shift_new == parity_shift_old) {
        tools::log->debug("set_parity_shift_mpo_squared: not needed -- parity shift is already [{} {}]", sign, axus);
        return;
    }

    model->set_parity_shift_mpo_squared(sign, axus); // will ignore the sign on the axis string if present
}

template<typename Scalar>
std::optional<DebugStatus<Scalar>> get_status(TensorsFinite<Scalar> &tensors, std::string_view tag) {
    if constexpr(not settings::debug) return std::nullopt;
    tensors.model->clear_cache();
    tensors.measurements = MeasurementsTensorsFinite<Scalar>();
    tensors.rebuild_mpo();
    tensors.rebuild_mpo_squared();
    tensors.rebuild_edges();
    DebugStatus<Scalar> deb;
    deb.ene     = tools::finite::measure::energy(tensors);
    deb.red     = tools::finite::measure::energy_shift(tensors);
    deb.var     = tools::finite::measure::energy_variance(tensors);
    deb.tag     = tag;
    deb.mpo_ids = tensors.model->get_active_ids();
    deb.env_ids = tensors.edges->get_active_ids();
    return deb;
}

template<typename Scalar>
void TensorsFinite<Scalar>::set_energy_shift_mpo(Scalar energy_shift) {
    if(std::isnan(std::real(energy_shift))) throw std::logic_error("TensorsFinite<Scalar>::set_energy_shift_mpo: got std::real(energy_shift) == NAN");
    if(std::isnan(std::imag(energy_shift))) throw std::logic_error("TensorsFinite<Scalar>::set_energy_shift_mpo: got std::imag(energy_shift) == NAN");
    if(std::abs(model->get_energy_shift_mpo() - energy_shift) <= std::numeric_limits<RealScalar>::epsilon()) {
        tools::log->debug("MPO energy is already shifted by {:.16f} : No shift needed.", fp(energy_shift));
        return;
    }
    std::vector<std::optional<DebugStatus<Scalar>>> debs;
    debs.emplace_back(get_status(*this, "Before shift"));

    measurements = MeasurementsTensorsFinite<Scalar>(); // Resets model-related measurements but not state measurements, which can remain
    model->clear_cache();

    tools::log->info("Shifting MPO energy: {:.16f} {:+.16f}i", fp(std::real(energy_shift)), fp(std::imag(energy_shift)));
    model->set_energy_shift_mpo(energy_shift);
    model->assert_validity();

    debs.emplace_back(get_status(*this, "After shift"));

    if(energy_shift != Scalar(0.0)) {
        auto &bef = debs.front();
        auto &aft = debs.back();
        if(bef and aft) {
            if(std::abs(bef->red - aft->red) > std::numeric_limits<RealScalar>::epsilon()) {
                if(bef->mpo_ids == aft->mpo_ids) throw except::runtime_error("MPO id's are unchanged after energy shift\n{}\n{}", bef->msg(), aft->msg());
                // It only makes sense to check the interior envs, since the edges are constant.
                // When the number of active sites is L, then only the edge envs are affected, but these are constant anyway!
                if(bef->env_ids == aft->env_ids and edges->active_sites.size() != get_length<size_t>())
                    throw except::runtime_error("ENV id's are unchanged after energy shift\n{}\n{}", bef->msg(), aft->msg());
            }
        }
    }

    for(const auto &deb : debs)
        if(deb) deb->print();

    if constexpr(settings::debug) {
        auto &bef = debs.front();
        auto &aft = debs.back();
        if(bef and aft) {
            RealScalar ratio_var     = std::abs(aft->var / bef->var);
            RealScalar delta_ene     = std::abs(bef->ene - aft->ene);
            RealScalar delta_var     = std::abs(bef->var - aft->var);
            RealScalar delta_ene_rel = delta_ene / std::abs(aft->ene) * 100;
            RealScalar delta_var_rel = delta_var / std::abs(aft->var) * 100;
            RealScalar critical_cancellation_max_decimals =
                std::numeric_limits<RealScalar>::digits10 -
                std::max<RealScalar>(static_cast<RealScalar>(0.0), static_cast<RealScalar>(std::log10(std::pow(std::abs(energy_shift), 2))));
            RealScalar critical_cancellation_error = std::pow<RealScalar>(static_cast<RealScalar>(10), -critical_cancellation_max_decimals);
            tools::log->debug("Variance change              {:>20.16f}", fp(delta_var));
            tools::log->debug("Variance change percent      {:>20.16f}", fp(delta_var_rel));
            tools::log->debug("Critical cancellation decs   {:>20.16f}", fp(critical_cancellation_max_decimals));
            tools::log->debug("Critical cancellation error  {:>20.16f}", fp(critical_cancellation_error));
            if(delta_var > critical_cancellation_error * static_cast<RealScalar>(1e3)) {
                tools::log->warn("Variance changed significantly after energy shift+compression");
                if(delta_var > static_cast<RealScalar>(1e6) * critical_cancellation_error)
                    throw except::runtime_error("Energy reduction destroyed variance precision");
            }
            if(delta_ene_rel > static_cast<RealScalar>(1e-8)) {
                tools::log->warn("Energy changed significantly after energy shift+compression");
                if(delta_ene_rel > static_cast<RealScalar>(1e-6))
                    throw except::runtime_error("Energy shift changed energy level {:.16f} -> {:.16f} (Δ/E = {:.3f} %)", fp(bef->ene), fp(aft->ene),
                                                fp(delta_ene_rel));
            }
            if(ratio_var < static_cast<RealScalar>(0.8) or ratio_var > static_cast<RealScalar>(1.2))
                tools::log->warn("Variance changed significantly after energy shift: old {:.6e} | new {:.6e}", fp(bef->var), fp(aft->var));
            else if(ratio_var < static_cast<RealScalar>(0.01) or ratio_var > static_cast<RealScalar>(100))
                throw except::runtime_error("Variance changed too much after energy shift: old {:.6e} | new {:.6e}", fp(bef->var), fp(aft->var));
        }
    }
}

template<typename Scalar>
std::tuple<OptRitz, int, std::string_view> TensorsFinite<Scalar>::get_parity_shift_mpo() {
    return model->get_parity_shift_mpo();
}
template<typename Scalar>
std::pair<int, std::string_view> TensorsFinite<Scalar>::get_parity_shift_mpo_squared() {
    return model->get_parity_shift_mpo_squared();
}
template<typename Scalar>
Scalar TensorsFinite<Scalar>::get_energy_shift_mpo() {
    return model->get_energy_shift_mpo();
}

template<typename Scalar>
void TensorsFinite<Scalar>::rebuild_mpo() {
    if(model->has_mpo()) {
        tools::log->trace("rebuild_mpo: the model already has mpos.");
        return; // They should have been cleared before rebulding them
    }
    tools::log->trace("Rebuilding MPO");
    measurements = MeasurementsTensorsFinite<Scalar>(); // Resets model-related measurements but not state measurements, which can remain
    model->clear_cache();
    model->build_mpo();
    if constexpr(settings::debug) model->assert_validity();
}

template<typename Scalar>
void TensorsFinite<Scalar>::rebuild_mpo_squared() {
    if(state->get_algorithm() == AlgorithmType::fLBIT) return;
    if(model->has_mpo_squared()) {
        tools::log->trace("rebuild_mpo_squared: the model already has mpos squared.");
        return; // They should have been cleared before rebulding them
    }
    tools::log->trace("Rebuilding MPO²");
    measurements = MeasurementsTensorsFinite<Scalar>(); // Resets model-related measurements but not state measurements, which can remain
    model->clear_cache();
    model->build_mpo_squared(); // Will build with energy and parity shifts if they are set, but it will not compress.
    if constexpr(settings::debug) model->assert_validity();
}

template<typename Scalar>
void TensorsFinite<Scalar>::compress_mpo_squared() {
    if(settings::precision::use_compressed_mpo_squared == MpoCompress::NONE) return;

    if(model->has_compressed_mpo_squared()) {
        // throw std::runtime_error("compress_mpo_squared: the model already has compressed mpo squared.");
        tools::log->trace("compress_mpo_squared: the model already has compressed mpo squared.");
        return; // They don't need to be compressed again. Rebuild before compressing them
    }
    tools::log->info("Compressing MPO²");
    measurements = MeasurementsTensorsFinite<Scalar>(); // Resets model-related measurements but not state measurements, which can remain
    model->compress_mpo_squared();

    if constexpr(settings::debug) model->assert_validity();
}

// Active sites
template<typename Scalar>
void TensorsFinite<Scalar>::sync_active_sites() {
    if(num::all_equal(active_sites, state->active_sites, model->active_sites, edges->active_sites)) return;
    if(not active_sites.empty())
        activate_sites(active_sites);
    else if(not state->active_sites.empty())
        activate_sites(state->active_sites);
    else if(not model->active_sites.empty())
        activate_sites(model->active_sites);
    else if(not edges->active_sites.empty())
        activate_sites(edges->active_sites);
    else
        clear_active_sites();
}

template<typename Scalar>
void TensorsFinite<Scalar>::clear_active_sites() {
    if constexpr(settings::debug) tools::log->trace("Clearing active sites {}", active_sites);
    active_sites.clear();
    state->active_sites.clear();
    model->active_sites.clear();
    edges->active_sites.clear();
}

template<typename Scalar>
void TensorsFinite<Scalar>::activate_sites(const std::vector<size_t> &sites) {
    tools::log->trace("Activating sites: {}", sites);
    if(num::all_equal(sites, active_sites, state->active_sites, model->active_sites, edges->active_sites)) return;
    active_sites        = sites;
    state->active_sites = active_sites;
    model->active_sites = active_sites;
    edges->active_sites = active_sites;
    clear_cache();
    clear_measurements();
}

template<typename Scalar>
void TensorsFinite<Scalar>::activate_sites() {
    sync_active_sites();
    if(active_sites.empty()) {
        if(position_is_at(-1)) throw except::logic_error("activate_sites: cannot activate a default site when pos == -1");
        activate_sites({get_position<size_t>()});
    }
}

template<typename Scalar>
void TensorsFinite<Scalar>::activate_sites(long threshold, size_t max_sites, size_t min_sites) {
    activate_sites(tools::finite::multisite::generate_site_list(get_state(), threshold, max_sites, min_sites));
}

template<typename Scalar>
std::array<long, 3> TensorsFinite<Scalar>::active_problem_dims() const {
    return tools::finite::multisite::get_dimensions(get_state(), active_sites);
}
template<typename Scalar>
long TensorsFinite<Scalar>::active_problem_size() const {
    return tools::finite::multisite::get_problem_size(get_state(), active_sites);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::is_real() const {
    if(settings::model::model_type == ModelType::ising_sdual) {
        if(not state->is_real()) tools::log->critical("state has imaginary part");
        if(not model->is_real()) tools::log->critical("model has imaginary part");
        if(not edges->is_real()) tools::log->critical("edges has imaginary part");
    }
    return state->is_real() and model->is_real() and edges->is_real();
}
template<typename Scalar>
bool TensorsFinite<Scalar>::has_nan() const {
    if(state->has_nan()) tools::log->critical("state has nan");
    if(model->has_nan()) tools::log->critical("model has nan");
    if(edges->has_nan()) tools::log->critical("edges has nan");
    return state->has_nan() or model->has_nan() or edges->has_nan();
}
template<typename Scalar>
void TensorsFinite<Scalar>::assert_validity() const {
    if(settings::model::model_type == ModelType::ising_sdual) {
        if(not state->is_real()) throw except::runtime_error("state has imaginary part");
        if(not model->is_real()) throw except::runtime_error("model has imaginary part");
        if(not edges->is_real()) throw except::runtime_error("edges has imaginary part");
    }

    state->assert_validity();
    model->assert_validity();
    edges->assert_validity();
    assert_edges();
}

template<typename Scalar>
bool TensorsFinite<Scalar>::has_center_point() const {
    return state->has_center_point();
}

template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_the_middle() const {
    return state->position_is_the_middle();
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_the_middle_any_direction() const {
    return state->position_is_the_middle_any_direction();
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_outward_edge_left(size_t nsite) const {
    return state->position_is_outward_edge_left(nsite);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_outward_edge_right(size_t nsite) const {
    return state->position_is_outward_edge_right(nsite);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_outward_edge(size_t nsite) const {
    return state->position_is_outward_edge(nsite);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_inward_edge_left(size_t nsite) const {
    return state->position_is_inward_edge_left(nsite);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_inward_edge_right(size_t nsite) const {
    return state->position_is_inward_edge_right(nsite);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_inward_edge(size_t nsite) const {
    return state->position_is_inward_edge(nsite);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_at(long pos) const {
    return state->position_is_at(pos);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_at(long pos, int dir) const {
    return state->position_is_at(pos, dir);
}
template<typename Scalar>
bool TensorsFinite<Scalar>::position_is_at(long pos, int dir, bool isCenter) const {
    return state->position_is_at(pos, dir, isCenter);
}
template<typename Scalar>
size_t TensorsFinite<Scalar>::move_center_point(std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_single_site(get_state(), svd_cfg);
    if(moves != 0) clear_active_sites();
    return moves;
}
template<typename Scalar>
size_t TensorsFinite<Scalar>::move_center_point_to_pos(long pos, std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_to_pos(get_state(), pos, svd_cfg);
    if(moves != 0) clear_active_sites();
    return moves;
}
template<typename Scalar>
size_t TensorsFinite<Scalar>::move_center_point_to_inward_edge(std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_to_inward_edge(get_state(), svd_cfg);
    if(moves != 0) clear_active_sites();
    return moves;
}
template<typename Scalar>
size_t TensorsFinite<Scalar>::move_center_point_to_middle(std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_to_middle(get_state(), svd_cfg);
    if(moves != 0) clear_active_sites();
    return moves;
}

template<typename Scalar>
void TensorsFinite<Scalar>::merge_multisite_mps(const Eigen::Tensor<Scalar, 3> &multisite_tensor, MergeEvent mevent, std::optional<svd::config> svd_cfg,
                                                LogPolicy log_policy) {
    // Make sure the active sites are the same everywhere
    auto t_merge = tid::tic_scope("merge");
    if(not num::all_equal(active_sites, state->active_sites, model->active_sites, edges->active_sites))
        throw except::runtime_error("All active sites are not equal: tensors {} | state {} | model {} | edges {}", active_sites, state->active_sites,
                                    model->active_sites, edges->active_sites);
    clear_measurements(log_policy);
    tools::finite::mps::merge_multisite_mps(get_state(), multisite_tensor, active_sites, get_position<long>(), mevent, svd_cfg, log_policy);
    normalize_state(svd_cfg, NormPolicy::IFNEEDED);
}

template<typename Scalar>
BondExpansionResult<Scalar> TensorsFinite<Scalar>::expand_bonds(const OptMeta &opt_meta) {
    // if(active_sites.empty()) throw except::runtime_error("No active sites for subspace expansion");
    auto t_exp     = tid::tic_scope("bondexp");
    auto expresult = BondExpansionResult<Scalar>();
    if(get_position<long>() < 0) {
        expresult.msg = "Negative position";
        return expresult;
    }
    if(!opt_meta.svd_cfg.has_value()) {
        expresult.msg = "svd not configured";
        return expresult;
    }
    if(active_sites.empty()) activate_sites({get_position<size_t>()});
    rebuild_edges(); // Use fresh edges
    if constexpr(settings::debug) assert_validity();
    if(has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::POSTOPT_1SITE) and opt_meta.optExit != OptExit::NONE) {
        expresult = tools::finite::env::expand_bond_postopt_1site(get_state(), get_model(), get_edges(), opt_meta);
    } else if(has_any_flags(opt_meta.bondexp_policy, BondExpansionPolicy::PREOPT_NSITE_REAR, BondExpansionPolicy::PREOPT_NSITE_FORE)) {
        expresult = tools::finite::env::expand_bond_preopt_nsite(get_state(), get_model(), get_edges(), opt_meta);
    } else {
        expresult.msg =
            fmt::format("The policy does not allow bond expansion: {} | OptExit: {}", flag2str(opt_meta.bondexp_policy), flag2str(opt_meta.optExit));
        return expresult;
    }
    if(expresult.ok) clear_measurements();
    if constexpr(settings::debug) assert_validity();
    return expresult;
}

template<typename Scalar>
void TensorsFinite<Scalar>::move_site_mps(const size_t site, const long steps, std::vector<size_t> &sites_mps, std::optional<long> new_pos) {
    if(sites_mps.size() != get_length<size_t>()) {
        sites_mps.clear();
        for(const auto &mps : state->mps_sites) sites_mps.emplace_back(mps->template get_position<size_t>());
    }
    long dir = steps < 0l ? -1l : 1l;

    for(long step = 0; std::abs(step) < std::abs(steps); step += dir) {
        long posL = std::min(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        long posR = std::max(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        if(posL == posR) break;
        if(posL < 0 or posL >= get_length<long>()) break;
        if(posR < 0 or posR >= get_length<long>()) break;
        tools::log->debug("swapping mps sites {} <--> {}", posL, posR);
        // Move the MPS site
        tools::finite::mps::swap_sites(get_state(), safe_cast<size_t>(posL), safe_cast<size_t>(posR), sites_mps, GateMove::OFF);
    }
    if(new_pos) {
        if(new_pos.value() != std::clamp(new_pos.value(), 0l, get_length<long>()))
            throw except::runtime_error("move_site: expected new_pos in range [0,{}]. Got {}", get_length<long>(), new_pos.value());
        move_center_point_to_pos(new_pos.value());
        activate_sites(std::vector<size_t>{safe_cast<size_t>(new_pos.value())});
    }

    tools::log->debug("Sites mps: {}", sites_mps);
    clear_cache();
    clear_measurements();
}

template<typename Scalar>
void TensorsFinite<Scalar>::move_site_mpo(const size_t site, const long steps, std::vector<size_t> &sites_mpo) {
    if(sites_mpo.size() != get_length<size_t>()) {
        sites_mpo.clear();
        for(const auto &mpo : model->MPO) sites_mpo.emplace_back(mpo->get_position());
    }
    long dir = steps < 0l ? -1l : 1l;

    for(long step = 0; std::abs(step) < std::abs(steps); step += dir) {
        long posL = std::min(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        long posR = std::max(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        if(posL == posR) break;
        if(posL < 0 or posL >= get_length<long>()) break;
        if(posR < 0 or posR >= get_length<long>()) break;
        tools::log->debug("swapping mpo sites {} <--> {}", posL, posR);
        // Move the MPO site
        tools::finite::mpo::swap_sites(get_model(), safe_cast<size_t>(posL), safe_cast<size_t>(posR), sites_mpo);
    }
    tools::log->debug("Sites mpo: {}", sites_mpo);
    clear_cache();
    clear_measurements();
}

template<typename Scalar>
void TensorsFinite<Scalar>::move_site_mps_to_pos(const size_t site, const long tgt_pos, std::vector<size_t> &sites_mps, std::optional<long> new_pos) {
    if(sites_mps.size() != get_length<size_t>()) {
        sites_mps.clear();
        for(const auto &mps : state->mps_sites) sites_mps.emplace_back(mps->template get_position<size_t>());
    }
    while(true) {
        // We must first find the src_pos of the given site in the list.
        // Example:
        //      site = 0
        //      tgt_pos = 0
        //      sites = {1,2,3,4,0,5,6,7}
        //  We can then calculate:
        //      src_pos = find(sites.begin(), sites.end(), site) = 4
        auto src_itr = std::find(sites_mps.begin(), sites_mps.end(), site);
        if(src_itr == sites_mps.end()) throw except::logic_error("site {} was not found in sites_mps: {}", site, sites_mps);
        auto src_pos = std::distance(sites_mps.begin(), src_itr);
        if(src_pos == tgt_pos) break;
        long step = tgt_pos < src_pos ? -1l : 1l;
        long posL = src_pos + (step < 0 ? -1l : 0);
        long posR = src_pos + (step < 0 ? 0 : 1l);
        if(posL == posR) break;
        if(posL < 0 or posL >= get_length<long>()) break;
        if(posR < 0 or posR >= get_length<long>()) break;
        tools::log->debug("swapping mps sites {} <--> {}", posL, posR);
        // Move the MPS site
        tools::finite::mps::swap_sites(get_state(), safe_cast<size_t>(posL), static_cast<size_t>(posR), sites_mps, GateMove::OFF);
    }
    if(new_pos) {
        if(new_pos.value() != std::clamp(new_pos.value(), 0l, get_length<long>()))
            throw except::runtime_error("move_site: expected new_pos in range [0,{}]. Got {}", get_length<long>(), new_pos.value());
        move_center_point_to_pos(new_pos.value());
        activate_sites(std::vector<size_t>{static_cast<size_t>(new_pos.value())});
    }

    tools::log->debug("Sites mps: {}", sites_mps);
    clear_cache();
    clear_measurements();
}

template<typename Scalar>
void TensorsFinite<Scalar>::move_site_mpo_to_pos(const size_t site, const long tgt_pos, std::vector<size_t> &sites_mpo) {
    if(sites_mpo.size() != get_length<size_t>()) {
        sites_mpo.clear();
        for(const auto &mpo : model->MPO) sites_mpo.emplace_back(mpo->get_position());
    }

    while(true) {
        // We must first find the src_pos of the given site in the list.
        // Example:
        //      site = 0
        //      tgt_pos = 0
        //      sites = {1,2,3,4,0,5,6,7}
        //  We can then calculate:
        //      src_pos = find(sites.begin(), sites.end(), site) = 4
        auto src_itr = std::find(sites_mpo.begin(), sites_mpo.end(), site);
        if(src_itr == sites_mpo.end()) throw except::logic_error("site {} was not found in sites_mpo: {}", site, sites_mpo);
        auto src_pos = std::distance(sites_mpo.begin(), src_itr);
        if(src_pos == tgt_pos) break;
        long step = tgt_pos < src_pos ? -1l : 1l;
        long posL = src_pos + (step < 0 ? -1l : 0);
        long posR = src_pos + (step < 0 ? 0 : 1l);
        if(posL == posR) break;
        if(posL < 0 or posL >= get_length<long>()) break;
        if(posR < 0 or posR >= get_length<long>()) break;
        tools::log->debug("swapping mpo sites {} <--> {}", posL, posR);
        // Move the MPO site
        tools::finite::mpo::swap_sites(get_model(), static_cast<size_t>(posL), static_cast<size_t>(posR), sites_mpo);
    }
    tools::log->debug("Sites mpo: {}", sites_mpo);
    clear_cache();
    clear_measurements();
}

template<typename Scalar>
void TensorsFinite<Scalar>::move_site(const size_t site, const long steps, std::vector<size_t> &sites_mps, std::vector<size_t> &sites_mpo,
                                      std::optional<long> new_pos) {
    move_site_mps(site, steps, sites_mps, new_pos);
    move_site_mpo(site, steps, sites_mpo);
}

template<typename Scalar>
void TensorsFinite<Scalar>::move_site_to_pos(const size_t site, const long tgt_pos, std::optional<std::vector<size_t>> &sites_mps,
                                             std::optional<std::vector<size_t>> &sites_mpo, std::optional<long> new_pos) {
    if(not sites_mps) sites_mps = std::vector<size_t>{};
    if(not sites_mpo) sites_mpo = std::vector<size_t>{};
    move_site_mps_to_pos(site, tgt_pos, sites_mps.value(), new_pos);
    move_site_mpo_to_pos(site, tgt_pos, sites_mpo.value());
    if(sites_mps != sites_mpo) throw except::logic_error("sites mismatch \n sites_mps {}\n sites_mpo {}", sites_mps.value(), sites_mpo.value());
}

template<typename Scalar>
void TensorsFinite<Scalar>::assert_edges() const {
    tools::finite::env::assert_edges(get_state(), get_model(), get_edges());
}
template<typename Scalar>
void TensorsFinite<Scalar>::assert_edges_ene() const {
    tools::finite::env::assert_edges_ene(get_state(), get_model(), get_edges());
}
template<typename Scalar>
void TensorsFinite<Scalar>::assert_edges_var() const {
    tools::finite::env::assert_edges_var(get_state(), get_model(), get_edges());
}
template<typename Scalar>
void TensorsFinite<Scalar>::rebuild_edges() {
    activate_sites();
    tools::finite::env::rebuild_edges(get_state(), get_model(), get_edges());
}
template<typename Scalar>
void TensorsFinite<Scalar>::rebuild_edges_ene() {
    activate_sites();
    tools::finite::env::rebuild_edges_ene(get_state(), get_model(), get_edges());
}
template<typename Scalar>
void TensorsFinite<Scalar>::rebuild_edges_var() {
    activate_sites();
    tools::finite::env::rebuild_edges_var(get_state(), get_model(), get_edges());
}

template<typename Scalar>
void TensorsFinite<Scalar>::clear_measurements(LogPolicy logPolicy) const {
    if(logPolicy == LogPolicy::VERBOSE or (settings::debug and logPolicy == LogPolicy::DEBUG)) tools::log->trace("Clearing tensors measurements");
    measurements = MeasurementsTensorsFinite<Scalar>();
    state->clear_measurements(logPolicy);
}

template<typename Scalar>
void TensorsFinite<Scalar>::clear_cache(LogPolicy logPolicy) const {
    if(logPolicy == LogPolicy::VERBOSE or (settings::debug and logPolicy == LogPolicy::DEBUG)) tools::log->trace("Clearing tensors cache");
    cache_fp32 = Cache<fp32>();
    cache_fp64 = Cache<fp64>();
    cache_fp128 = Cache<fp128>();
    cache_cx32 = Cache<cx32>();
    cache_cx64 = Cache<cx64>();
    cache_cx128 = Cache<cx128>();
    model->clear_cache(logPolicy);
    state->clear_cache(logPolicy);
}