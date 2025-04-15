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

    /* clang-format off */
    template<typename Scalar>               extern void                  apply_mpo                        (StateFinite<Scalar>& state, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,3> &Ledge, const Eigen::Tensor<Scalar,3> & Redge);
    template<typename Scalar>               extern void                  apply_mpos                       (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge, bool adjoint = false);
    template<typename Scalar>               extern void                  apply_mpos                       (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,3> & Ledge, const Eigen::Tensor<Scalar,3> & Redge, bool adjoint = false);
    template<typename Scalar>               extern void                  apply_mpos_general               (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge, const svd::config & svd_cfg);
    template<typename Scalar>               extern void                  apply_mpos_general               (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const svd::config & svd_cfg);
    template<typename Scalar> [[nodiscard]] extern std::optional<double> get_spin_component_along_axis    (StateFinite<Scalar>& state, std::string_view axis);
    template<typename Scalar>               extern void                  project_to_axis                  (StateFinite<Scalar>& state, const Eigen::MatrixXcd & paulimatrix, int sign, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> [[nodiscard]] extern int                   project_to_nearest_axis          (StateFinite<Scalar>& state, std::string_view axis, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> [[nodiscard]] extern StateFinite<Scalar>   get_projection_to_axis           (const StateFinite<Scalar>& state, const Eigen::MatrixXcd & paulimatrix, int sign, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> [[nodiscard]] extern StateFinite<Scalar>   get_projection_to_nearest_axis   (const StateFinite<Scalar>& state, std::string_view  axis, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> [[nodiscard]] extern Scalar                overlap                          (const StateFinite<Scalar>& state1, const StateFinite<Scalar>& state2);
    /* clang-format on */
}