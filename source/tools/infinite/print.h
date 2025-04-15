#pragma once
template <typename Scalar> class StateInfinite;
template <typename Scalar> class ModelInfinite;
template <typename Scalar> class TensorsInfinite;
namespace tools::infinite::print {
    extern void dimensions(const TensorsInfinite &tensors);      /*!< Print the tensor dimensions for all \f$\Gamma\f$-tensors. */
    extern void print_state_compact(const StateInfinite &state); /*!< Print the tensor dimensions for all \f$\Gamma\f$-tensors. */
    extern void print_hamiltonians(const ModelInfinite &model);
}