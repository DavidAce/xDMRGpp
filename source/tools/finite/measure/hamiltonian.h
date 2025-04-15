#pragma once
#include "math/float.h"
#include "math/svd/config.h"
#include "math/tenx/fwd_decl.h"
#include <optional>

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class TensorsFinite;
template<typename Scalar>
class MpoSite;
template<typename Scalar>
class MpsSite;
template<typename Scalar>
struct MeasurementsTensorsFinite;
template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;
template<typename T>
struct env_pair;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian                        (const Eigen::Tensor<Scalar, 3> &mps, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian                        (const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs, const env_pair<const EnvEne<Scalar> &> &envs);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian_squared                (const Eigen::Tensor<Scalar, 3> &mps, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian_squared                (const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs, const env_pair<const EnvVar<Scalar> &> &envs);


    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian                       (const TensorsFinite<Scalar> & tensors);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian                       (const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian                       (const std::vector<size_t> & sites, const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian_squared               (const TensorsFinite<Scalar> & tensors);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian_squared               (const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges);
    template<typename Scalar> [[nodiscard]] Scalar expval_hamiltonian_squared               (const std::vector<size_t> & sites, const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges);


    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_minus_energy_shift               (const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy                                  (const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_variance                         (const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_normalized                       (const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, RealScalar<Scalar> energy_min, RealScalar<Scalar> energy_max, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);

    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_shift                    (const TensorsFinite<Scalar> & tensors);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_minus_energy_shift       (const TensorsFinite<Scalar> & tensors);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy                          (const TensorsFinite<Scalar> & tensors);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_variance                 (const TensorsFinite<Scalar> & tensors);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_normalized               (const TensorsFinite<Scalar> & tensors, RealScalar<Scalar> energy_minimum, RealScalar<Scalar> energy_maximum);

    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_minus_energy_shift       (const StateFinite<Scalar> & state, const TensorsFinite<Scalar> & tensors, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy                          (const StateFinite<Scalar> & state, const TensorsFinite<Scalar> & tensors, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_variance                 (const StateFinite<Scalar> & state, const TensorsFinite<Scalar> & tensors, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_normalized               (const StateFinite<Scalar> & state, const TensorsFinite<Scalar> & tensors, RealScalar<Scalar> energy_minimum, RealScalar<Scalar> energy_maximum, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);


    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_minus_energy_shift      (const Eigen::Tensor<Scalar,3> & multisite_mps, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy                         (const Eigen::Tensor<Scalar,3> & multisite_mps, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_variance                (const Eigen::Tensor<Scalar,3> & multisite_mps, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_normalized              (const Eigen::Tensor<Scalar,3> & multisite_mps, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, RealScalar<Scalar> energy_min, RealScalar<Scalar> energy_max, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);


    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_minus_energy_shift   (const Eigen::Tensor<Scalar,3> &mps, const TensorsFinite<Scalar> & tensors, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy                      (const Eigen::Tensor<Scalar,3> &mps, const TensorsFinite<Scalar> & tensors, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_variance             (const Eigen::Tensor<Scalar,3> &mps, const TensorsFinite<Scalar> & tensors, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> energy_normalized           (const Eigen::Tensor<Scalar,3> &mps, const TensorsFinite<Scalar> & tensors, RealScalar<Scalar> energy_minimum, RealScalar<Scalar> energy_maximum, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> * measurements = nullptr);

    /* clang-format on */
}
