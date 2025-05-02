#pragma once
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

template<typename Scalar>
class TensorsInfinite;
template<typename Scalar>
class StateInfinite;
template<typename Scalar>
class ModelInfinite;
template<typename Scalar>
class EdgesInfinite;
namespace tools::infinite::measure {
    /* clang-format off */
    template<typename Scalar>
    using RealScalar = decltype(std::real(std::declval<Scalar>()));

    template<typename Scalar> extern size_t length                           (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern size_t length                           (const EdgesInfinite<Scalar> & edges);
    template<typename Scalar> extern long   bond_dimension                   (const StateInfinite<Scalar> & state);
    template<typename Scalar> extern double truncation_error                 (const StateInfinite<Scalar> & state);
    template<typename Scalar> extern RealScalar<Scalar> norm                 (const StateInfinite<Scalar> & state);
    template<typename Scalar> extern RealScalar<Scalar> entanglement_entropy (const StateInfinite<Scalar> & state);

    template<typename state_or_mps_type, typename Scalar>
    RealScalar<Scalar> energy_minus_energy_shift     (const state_or_mps_type & state, const ModelInfinite<Scalar> & model, const EdgesInfinite<Scalar> & edges);
    template<typename state_or_mps_type, typename Scalar>
    RealScalar<Scalar> energy_mpo                      (const state_or_mps_type & state, const ModelInfinite<Scalar> & model, const EdgesInfinite<Scalar> & edges);
    template<typename state_or_mps_type, typename Scalar>
    RealScalar<Scalar> energy_per_site_mpo             (const state_or_mps_type & state, const ModelInfinite<Scalar> & model, const EdgesInfinite<Scalar> & edges);
    template<typename state_or_mps_type, typename Scalar>
    RealScalar<Scalar> energy_variance_mpo             (const state_or_mps_type & state, const ModelInfinite<Scalar> & model, const EdgesInfinite<Scalar> & edges);
    template<typename state_or_mps_type, typename Scalar>
    RealScalar<Scalar> energy_variance_per_site_mpo    (const state_or_mps_type & state, const ModelInfinite<Scalar> & model, const EdgesInfinite<Scalar> & edges);


    template<typename Scalar> extern RealScalar<Scalar> energy_mpo                      (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_per_site_mpo             (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_variance_mpo             (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_variance_per_site_mpo    (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_mpo                      (const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_per_site_mpo             (const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_variance_mpo             (const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_variance_per_site_mpo    (const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_per_site_ham             (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_per_site_mom             (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_variance_per_site_ham    (const TensorsInfinite<Scalar> & tensors);
    template<typename Scalar> extern RealScalar<Scalar> energy_variance_per_site_mom    (const TensorsInfinite<Scalar> & tensors);

    /* clang-format on */

}