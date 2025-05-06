#pragma once
template<typename Scalar>
class StateInfinite;
template<typename Scalar>
class ModelInfinite;
template<typename Scalar>
class TensorsInfinite;
namespace tools::infinite::print {
    template<typename Scalar>
    void dimensions(const TensorsInfinite<Scalar> &tensors); /*!< Print the tensor dimensions for all \f$\Gamma\f$-tensors. */
    template<typename Scalar>
    void print_state_compact(const StateInfinite<Scalar> &state); /*!< Print the tensor dimensions for all \f$\Gamma\f$-tensors. */
    template<typename Scalar>
    void print_hamiltonians(const ModelInfinite<Scalar> &model);
}