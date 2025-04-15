#pragma once
#include "gate.h"
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "qm.h"
#include <complex>
#include <vector>

namespace qm::time {
    /* clang-format off */
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar, 2>> Suzuki_Trotter_1st_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar, 2>> Suzuki_Trotter_2nd_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar, 2>> Suzuki_Trotter_4th_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar, 2>> get_twosite_time_evolution_operators(cx128 delta_t, size_t susuki_trotter_order, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar, 2>> compute_G(cx128 a, size_t susuki_trotter_order, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
    extern std::pair<std::vector<qm::Gate>, std::vector<qm::Gate>> get_time_evolution_gates(cx128 delta_t, const std::vector<qm::Gate> &hams_nsite);
    /* clang-format on */

}