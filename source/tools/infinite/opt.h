#pragma once
#include "config/enums.h"
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

template <typename Scalar> class TensorsInfinite;
template <typename Scalar> class StateInfinite;
namespace tools::infinite::opt {
    template<typename Scalar> extern Eigen::Tensor<Scalar, 3> find_ground_state(const TensorsInfinite<Scalar> &tensors, std::string_view ritz);
    template<typename Scalar> extern Eigen::Tensor<Scalar, 3> find_ground_state(const TensorsInfinite<Scalar> &tensors, OptRitz ritz);
    template<typename Scalar> extern Eigen::Tensor<Scalar, 3> time_evolve_state(const StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 2> &U);
}
