#pragma once

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;

namespace tools::finite::env {

    /* clang-format off */
    template<typename Scalar> void                           assert_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           assert_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           assert_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);

    template<typename Scalar> void                           rebuild_edges_x2(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges_var_x2(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);
    template<typename Scalar> void                           rebuild_edges_ene_x2(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges);

    /* clang-format on */

}
