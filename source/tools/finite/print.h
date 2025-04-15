#pragma once

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
class TensorsFinite;
namespace tools::finite::print {
    template<typename Scalar>
    extern void dimensions(const StateFinite<Scalar> &state);
    template<typename Scalar>
    extern void dimensions(const EdgesFinite<Scalar> &edges);
    template<typename Scalar>
    extern void dimensions(const TensorsFinite<Scalar> &tensors);
    template<typename Scalar>
    extern void model(const ModelFinite<Scalar> &model);

}