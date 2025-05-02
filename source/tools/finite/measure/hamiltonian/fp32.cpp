#include "../hamiltonian.impl.h"

using Scalar = fp32;
using Real   = fp32;

/* clang-format off */
template Scalar tools::finite::measure::expval_hamiltonian<Scalar>(const Eigen::Tensor<Scalar, 3> &mps , const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template Scalar tools::finite::measure::expval_hamiltonian<Scalar>(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template Scalar tools::finite::measure::expval_hamiltonian<Scalar>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template Scalar  tools::finite::measure::expval_hamiltonian(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs, const env_pair<const EnvEne<Scalar> &> &envs);

template Scalar  tools::finite::measure::expval_hamiltonian(const TensorsFinite<Scalar> &tensors);

template Scalar  tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<Scalar, 3> &mps, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template Scalar  tools::finite::measure::expval_hamiltonian_squared<Scalar>(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template Scalar tools::finite::measure::expval_hamiltonian_squared<Scalar>(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);

template Scalar  tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs, const env_pair<const EnvVar<Scalar> &> &envs);

template Scalar  tools::finite::measure::expval_hamiltonian_squared(const TensorsFinite<Scalar> &tensors);

template Real tools::finite::measure::energy_minus_energy_shift(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy (const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, MeasurementsTensorsFinite<Scalar> *measurements);

template Real  tools::finite::measure::energy(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_variance(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_variance(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_shift(const TensorsFinite<Scalar> &tensors);

template Real tools::finite::measure::energy_minus_energy_shift(const TensorsFinite<Scalar> &tensors);

template Real tools::finite::measure::energy(const TensorsFinite<Scalar> &tensors);

template Real tools::finite::measure::energy_variance(const TensorsFinite<Scalar> &tensors);

template Real tools::finite::measure::energy_variance(const Eigen::Tensor<Scalar, 3> &multisite_mps,  const TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_normalized(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, Real energy_min, Real energy_max, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_normalized(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, Real energy_min, Real energy_max, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_normalized(const TensorsFinite<Scalar> &tensors, Real emin, Real emax);

template Real tools::finite::measure::energy_normalized(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors, Real emin, Real emax, MeasurementsTensorsFinite<Scalar> *measurements);

template Real tools::finite::measure::energy_normalized(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors, Real emin, Real emax, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements);

/* clang-format on */
