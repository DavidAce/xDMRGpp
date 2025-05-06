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
    void dimensions(const TensorsFinite<Scalar> &tensors);
    template<typename Scalar>
    void model(const ModelFinite<Scalar> &model);

}