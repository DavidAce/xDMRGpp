#include "TensorsInfinite.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateInfinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/infinite/env.h"
#include "tools/infinite/measure.h"
#include "tools/infinite/mps.h"

template class TensorsInfinite<cx64>;
template class TensorsInfinite<cx128>;

template<typename Scalar>
TensorsInfinite<Scalar>::TensorsInfinite()
    : state(std::make_unique<StateInfinite<Scalar>>()), model(std::make_unique<ModelInfinite<Scalar>>()), edges(std::make_unique<EdgesInfinite<Scalar>>()) {
    tools::log->trace("Constructing TensorsInfinite");
}

// We need to make a destructor manually because we enclose
// the classes with unique_ptr. Otherwise unique_ptr will
// forcibly inline its own default deleter.
// This is a classic pimpl idiom.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
template<typename Scalar>
TensorsInfinite<Scalar>::~TensorsInfinite() = default; // default dtor
template<typename Scalar>
TensorsInfinite<Scalar>::TensorsInfinite(TensorsInfinite &&other) = default; // default move ctor
template<typename Scalar>
TensorsInfinite<Scalar> &TensorsInfinite<Scalar>::operator=(TensorsInfinite &&other) = default; // default move assign

template<typename Scalar>
TensorsInfinite<Scalar>::TensorsInfinite(const TensorsInfinite &other)
    : state(std::make_unique<StateInfinite<Scalar>>(*other.state)), model(std::make_unique<ModelInfinite<Scalar>>(*other.model)),
      edges(std::make_unique<EdgesInfinite<Scalar>>(*other.edges)), measurements(other.measurements) {}

template<typename Scalar>
TensorsInfinite<Scalar> &TensorsInfinite<Scalar>::operator=(const TensorsInfinite &other) {
    // check for self-assignment
    if(this != &other) {
        state        = std::make_unique<StateInfinite<Scalar>>(*other.state);
        model        = std::make_unique<ModelInfinite<Scalar>>(*other.model);
        edges        = std::make_unique<EdgesInfinite<Scalar>>(*other.edges);
        measurements = other.measurements;
    }
    return *this;
}

template<typename Scalar>
void TensorsInfinite<Scalar>::initialize(ModelType model_type_) {
    state->initialize(model_type_);
    model->initialize(model_type_);
    edges->initialize();
}

template<typename Scalar>
void TensorsInfinite<Scalar>::initialize_model() {
    auto t_rnd = tid::tic_scope("rnd_model", tid::level::higher);
    model->randomize();
    model->rebuild_mpo_squared();
    reset_edges();
}

template<typename Scalar>
void TensorsInfinite<Scalar>::reset_to_random_product_state(std::string_view sector, bool use_eigenspinors, std::string &pattern) {
    eject_edges();
    state->clear_cache(); // Other caches can remain intact
    tools::infinite::mps::random_product_state(*state, sector, use_eigenspinors, pattern);
    reset_edges();
}

template<typename Scalar>
bool TensorsInfinite<Scalar>::is_real() const {
    return state->is_real() and model->is_real() and edges->is_real();
}
template<typename Scalar>
bool TensorsInfinite<Scalar>::has_nan() const {
    return state->has_nan() or model->has_nan() or edges->has_nan();
}
template<typename Scalar>
void TensorsInfinite<Scalar>::assert_validity() const {
    state->assert_validity();
    model->assert_validity();
    edges->assert_validity();
}

template<typename Scalar>
size_t TensorsInfinite<Scalar>::get_length() const {
    return edges->get_length();
}

template<typename Scalar>
void TensorsInfinite<Scalar>::reset_edges() {
    tools::infinite::env::reset_edges(*state, *model, *edges);
    clear_measurements();
}

template<typename Scalar>
void TensorsInfinite<Scalar>::eject_edges() {
    edges->eject_edges();
    clear_measurements();
}

template<typename Scalar>
void TensorsInfinite<Scalar>::merge_twosite_tensor(const Eigen::Tensor<Scalar, 3> &twosite_tensor, MergeEvent mevent, std::optional<svd::config> svd_cfg) {
    state->clear_cache();
    clear_measurements();
    tools::infinite::mps::merge_twosite_tensor(*state, twosite_tensor, mevent, svd_cfg);
    //    normalize_state(bond_lim, svd_threshold, NormPolicy::IFNEEDED);
}

template<typename Scalar>
void TensorsInfinite<Scalar>::enlarge() {
    tools::infinite::env::enlarge_edges(*state, *model, *edges);
    state->swap_AB();
    clear_cache();
    clear_measurements();
}

template<typename Scalar>
void TensorsInfinite<Scalar>::clear_measurements() const {
    state->clear_measurements();
    measurements = MeasurementsTensorsInfinite<Scalar>();
}

template<typename Scalar>
void TensorsInfinite<Scalar>::clear_cache() const {
    state->clear_cache();
    model->clear_cache();
}