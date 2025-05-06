#pragma once
#include "math/float.h"
#include "math/svd/config.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
template<typename Scalar>
class StateFinite;
template<typename Scalar>
class MpoSite;

namespace tools::finite::ops {
    template<typename Scalar> using RealScalar = decltype(std::real(std::declval<Scalar>()));
    /* clang-format off */
    template<typename Scalar>               void                  apply_mpo                        (StateFinite<Scalar>& state, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,3> &Ledge, const Eigen::Tensor<Scalar,3> & Redge);
    template<typename Scalar>               void                  apply_mpos                       (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge, bool adjoint = false);
    template<typename Scalar>               void                  apply_mpos                       (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,3> & Ledge, const Eigen::Tensor<Scalar,3> & Redge, bool adjoint = false);
    template<typename Scalar>               void                  apply_mpos_general               (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge, const svd::config & svd_cfg);
    template<typename Scalar>               void                  apply_mpos_general               (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const svd::config & svd_cfg);
    template<typename Scalar> [[nodiscard]] std::optional<RealScalar<Scalar>>
                                                                  get_spin_component_along_axis    (StateFinite<Scalar>& state, std::string_view axis);
    template<typename Scalar>               void                  project_to_axis                  (StateFinite<Scalar>& state, const Eigen::Matrix2cd & paulimatrix, int sign, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> [[nodiscard]] int                   project_to_nearest_axis          (StateFinite<Scalar>& state, std::string_view axis, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> [[nodiscard]] StateFinite<Scalar>   get_projection_to_axis           (const StateFinite<Scalar>& state, const Eigen::Matrix2cd & paulimatrix, int sign, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> [[nodiscard]] StateFinite<Scalar>   get_projection_to_nearest_axis   (const StateFinite<Scalar>& state, std::string_view  axis, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename T, typename Scalar> [[nodiscard]] Scalar    overlap                          (const StateFinite<Scalar>& state1, const StateFinite<Scalar>& state2);
    /* clang-format on */

    template<typename Scalar>
    void apply_mpo(StateFinite<Scalar> &state, const Eigen::Tensor<Scalar, 4> &mpo, const Eigen::Tensor<Scalar, 3> &Ledge,
                   const Eigen::Tensor<Scalar, 3> &Redge) {
        std::vector<Eigen::Tensor<Scalar, 4>> mpos(state.get_length(), mpo);
        apply_mpos(state, mpos, Ledge, Redge);
    }
    template<typename Scalar>
    StateFinite<Scalar> get_projection_to_axis(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &paulimatrix, int sign,
                                               std::optional<svd::config> svd_cfg) {
        auto state_projected = state;
        project_to_axis(state_projected, paulimatrix, sign, svd_cfg);
        return state_projected;
    }
    template<typename Scalar>
    StateFinite<Scalar> get_projection_to_nearest_axis(const StateFinite<Scalar> &state, std::string_view axis, std::optional<svd::config> svd_cfg) {
        auto                  state_projected = state;
        [[maybe_unused]] auto sign            = project_to_nearest_axis(state_projected, axis, svd_cfg);
        return state_projected;
    }
}