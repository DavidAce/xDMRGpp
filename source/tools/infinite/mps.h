#pragma once
#include "config/enums.h"
#include "math/svd/config.h"
#include "math/tenx/fwd_decl.h"
#include <complex>
#include <optional>
#include <set>
#include <string>

template<typename Scalar>
class StateInfinite;
namespace tools::infinite::mps {
    template<typename Scalar>
    extern void merge_twosite_tensor(StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 3> &twosite_tensor, MergeEvent mevent,
                                     std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>
    extern void random_product_state(const StateInfinite<Scalar> &state, [[maybe_unused]] std::string_view sector, bool use_eigenspinors,
                                     [[maybe_unused]] std::string &pattern);
}