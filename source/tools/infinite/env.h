#pragma once
template<typename Scalar>
class StateInfinite;
template<typename Scalar>
class ModelInfinite;
template<typename Scalar>
class EdgesInfinite;

namespace tools::infinite::env {
    template<typename Scalar>
    void reset_edges(const StateInfinite<Scalar> &state, const ModelInfinite<Scalar> &model, EdgesInfinite<Scalar> &edges);
    template<typename Scalar>
    void enlarge_edges(const StateInfinite<Scalar> &state, const ModelInfinite<Scalar> &model, EdgesInfinite<Scalar> &edges);
}
