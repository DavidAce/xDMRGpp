#pragma once
#include "debug/exceptions.h"
#include "math/svd/config.h"
#include <optional>
#include <string_view>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename Scalar>
class StateInfinite;
enum class MergeEvent;
namespace tools::infinite::mps {
    template<typename Scalar>
    void merge_twosite_tensor(StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 3> &twosite_tensor, MergeEvent mevent,
                              std::optional<svd::config> svd_cfg = std::nullopt);

    template<typename Scalar>
    void random_product_state([[maybe_unused]] const StateInfinite<Scalar> &state, [[maybe_unused]] std::string_view sector,
                              [[maybe_unused]] bool use_eigenspinors, [[maybe_unused]] std::string &pattern) {
        throw except::runtime_error("random product state for infinite state not implemented yet");
    }
}
