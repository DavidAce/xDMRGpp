#include "../env.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/state/StateInfinite.h"

template<typename Scalar>
void tools::infinite::env::reset_edges(const StateInfinite<Scalar> &state, const ModelInfinite<Scalar> &model, EdgesInfinite<Scalar> &edges) {
    edges.get_ene().L.set_edge_dims(state.get_mps_siteA(), model.get_mpo_siteA());
    edges.get_ene().R.set_edge_dims(state.get_mps_siteB(), model.get_mpo_siteB());
    edges.get_var().L.set_edge_dims(state.get_mps_siteA(), model.get_mpo_siteA());
    edges.get_var().R.set_edge_dims(state.get_mps_siteB(), model.get_mpo_siteB());
}
template void tools::infinite::env::reset_edges(const StateInfinite<fp32> &state, const ModelInfinite<fp32> &model, EdgesInfinite<fp32> &edges);
template void tools::infinite::env::reset_edges(const StateInfinite<fp64> &state, const ModelInfinite<fp64> &model, EdgesInfinite<fp64> &edges);
template void tools::infinite::env::reset_edges(const StateInfinite<fp128> &state, const ModelInfinite<fp128> &model, EdgesInfinite<fp128> &edges);
template void tools::infinite::env::reset_edges(const StateInfinite<cx32> &state, const ModelInfinite<cx32> &model, EdgesInfinite<cx32> &edges);
template void tools::infinite::env::reset_edges(const StateInfinite<cx64> &state, const ModelInfinite<cx64> &model, EdgesInfinite<cx64> &edges);
template void tools::infinite::env::reset_edges(const StateInfinite<cx128> &state, const ModelInfinite<cx128> &model, EdgesInfinite<cx128> &edges);

template<typename Scalar>
void tools::infinite::env::enlarge_edges(const StateInfinite<Scalar> &state, const ModelInfinite<Scalar> &model, EdgesInfinite<Scalar> &edges) {
    auto ene = edges.get_ene();
    ene.L    = ene.L.enlarge(state.get_mps_siteA(), model.get_mpo_siteA());
    ene.R    = ene.R.enlarge(state.get_mps_siteB(), model.get_mpo_siteB());

    auto var = edges.get_var();
    var.L    = var.L.enlarge(state.get_mps_siteA(), model.get_mpo_siteA());
    var.R    = var.R.enlarge(state.get_mps_siteB(), model.get_mpo_siteB());
}

template void tools::infinite::env::enlarge_edges(const StateInfinite<fp32> &state, const ModelInfinite<fp32> &model, EdgesInfinite<fp32> &edges);
template void tools::infinite::env::enlarge_edges(const StateInfinite<fp64> &state, const ModelInfinite<fp64> &model, EdgesInfinite<fp64> &edges);
template void tools::infinite::env::enlarge_edges(const StateInfinite<fp128> &state, const ModelInfinite<fp128> &model, EdgesInfinite<fp128> &edges);
template void tools::infinite::env::enlarge_edges(const StateInfinite<cx32> &state, const ModelInfinite<cx32> &model, EdgesInfinite<cx32> &edges);
template void tools::infinite::env::enlarge_edges(const StateInfinite<cx64> &state, const ModelInfinite<cx64> &model, EdgesInfinite<cx64> &edges);
template void tools::infinite::env::enlarge_edges(const StateInfinite<cx128> &state, const ModelInfinite<cx128> &model, EdgesInfinite<cx128> &edges);