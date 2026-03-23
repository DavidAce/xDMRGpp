#pragma once
#include <complex>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
class TensorsFinite;
template<typename Scalar>
class MpoSite;
template<typename Scalar>
class MpsSite;
template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;
template<typename T>
struct env_pair;

struct MeasurementFiniteIds {
    std::vector<size_t>                sites{};
    std::optional<std::vector<size_t>> mps_ids     = std::nullopt;
    std::optional<std::vector<size_t>> mpo_ids     = std::nullopt;
    std::optional<std::vector<size_t>> mpo_sq_ids  = std::nullopt;
    std::optional<std::vector<size_t>> env_ene_ids = std::nullopt;
    std::optional<std::vector<size_t>> env_var_ids = std::nullopt;
    std::optional<long long>           maxiter     = std::nullopt;
    std::optional<double>              reltol      = std::nullopt;

    [[nodiscard]] auto operator==(const MeasurementFiniteIds &) const -> bool = default;
};

template<typename Value>
struct MeasurementFiniteCached {
    Value                data{};
    MeasurementFiniteIds ids{};

    [[nodiscard]] const Value &value() const { return data; }
    [[nodiscard]] Value       &value() { return data; }
};

template<typename Scalar>
struct MeasurementsTensorsFinite {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MpsRefs    = std::vector<std::reference_wrapper<const MpsSite<Scalar>>>;
    using MpoRefs    = std::vector<std::reference_wrapper<const MpoSite<Scalar>>>;
    using EneEnvs    = env_pair<const EnvEne<Scalar> &>;
    using VarEnvs    = env_pair<const EnvVar<Scalar> &>;

    std::optional<size_t>                              length                     = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> energy                     = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> energy_variance            = std::nullopt;
    std::optional<RealScalar>                          energy_shift               = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> energy_minus_energy_shift  = std::nullopt;
    std::optional<MeasurementFiniteCached<Scalar>>     expval_hamiltonian         = std::nullopt;
    std::optional<MeasurementFiniteCached<Scalar>>     expval_hamiltonian_squared = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> local_hamiltonian_norm             = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> local_hamiltonian_squared_norm     = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> global_hamiltonian_trace           = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> global_hamiltonian_squared_trace   = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> residual_norm_h1                   = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> residual_norm_h2                   = std::nullopt;
    std::optional<MeasurementFiniteCached<RealScalar>> residual_norm_full                 = std::nullopt;

    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy_minus_energy_shift(const MpsRefs &mps, const MpoRefs &mpo,
                                                                                                  const EneEnvs &envs) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy_variance(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs_ene,
                                                                                        const VarEnvs &envs_var) const;
    [[nodiscard]] const MeasurementFiniteCached<Scalar>     *get_cached_expval_hamiltonian(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs) const;
    [[nodiscard]] const MeasurementFiniteCached<Scalar>     *get_cached_expval_hamiltonian_squared(const MpsRefs &mps, const MpoRefs &mpo,
                                                                                                   const VarEnvs &envs) const;

    void set_cached_energy(RealScalar value, const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs);
    void set_cached_energy_minus_energy_shift(RealScalar value, const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs);
    void set_cached_energy_variance(RealScalar value, const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs_ene, const VarEnvs &envs_var);
    void set_cached_expval_hamiltonian(Scalar value, const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs);
    void set_cached_expval_hamiltonian_squared(Scalar value, const MpsRefs &mps, const MpoRefs &mpo, const VarEnvs &envs);

    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                               const EdgesFinite<Scalar> &edges) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *
        get_cached_energy_minus_energy_shift(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy_variance(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                                        const EdgesFinite<Scalar> &edges) const;
    [[nodiscard]] const MeasurementFiniteCached<Scalar>     *get_cached_expval_hamiltonian(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                                           const EdgesFinite<Scalar> &edges) const;
    [[nodiscard]] const MeasurementFiniteCached<Scalar> *
        get_cached_expval_hamiltonian_squared(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) const;

    void set_cached_energy(RealScalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    void set_cached_energy_minus_energy_shift(RealScalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                              const EdgesFinite<Scalar> &edges);
    void set_cached_energy_variance(RealScalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    void set_cached_expval_hamiltonian(Scalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges);
    void set_cached_expval_hamiltonian_squared(Scalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                               const EdgesFinite<Scalar> &edges);

    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy_minus_energy_shift(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_energy_variance(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<Scalar>     *get_cached_expval_hamiltonian(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<Scalar>     *get_cached_expval_hamiltonian_squared(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_local_hamiltonian_norm(const TensorsFinite<Scalar> &tensors, long long maxiter,
                                                                                                double reltol) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_local_hamiltonian_squared_norm(const TensorsFinite<Scalar> &tensors,
                                                                                                        long long maxiter, double reltol) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_global_hamiltonian_trace(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_global_hamiltonian_squared_trace(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_residual_norm_h1(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_residual_norm_h2(const TensorsFinite<Scalar> &tensors) const;
    [[nodiscard]] const MeasurementFiniteCached<RealScalar> *get_cached_residual_norm_full(const TensorsFinite<Scalar> &tensors) const;

    void set_cached_energy(RealScalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_energy_minus_energy_shift(RealScalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_energy_variance(RealScalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_expval_hamiltonian(Scalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_expval_hamiltonian_squared(Scalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_local_hamiltonian_norm(RealScalar value, const TensorsFinite<Scalar> &tensors, long long maxiter, double reltol);
    void set_cached_local_hamiltonian_squared_norm(RealScalar value, const TensorsFinite<Scalar> &tensors, long long maxiter, double reltol);
    void set_cached_global_hamiltonian_trace(RealScalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_global_hamiltonian_squared_trace(RealScalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_residual_norm_h1(RealScalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_residual_norm_h2(RealScalar value, const TensorsFinite<Scalar> &tensors);
    void set_cached_residual_norm_full(RealScalar value, const TensorsFinite<Scalar> &tensors);

    private:
    template<typename Value>
    [[nodiscard]] static const MeasurementFiniteCached<Value> *get_cached_impl(const std::optional<MeasurementFiniteCached<Value>> &cache,
                                                                               const MeasurementFiniteIds                          &ids);
    template<typename Value>
    static void set_cached_impl(std::optional<MeasurementFiniteCached<Value>> &cache, Value value, MeasurementFiniteIds ids);

    template<typename Site>
    [[nodiscard]] static std::vector<size_t> get_sites(const std::vector<std::reference_wrapper<const Site>> &refs);
    template<typename EnvRef>
    [[nodiscard]] static std::vector<size_t> get_env_ids(const env_pair<EnvRef> &envs);
    [[nodiscard]] static std::vector<size_t> get_mps_ids(const MpsRefs &mps);
    [[nodiscard]] static std::vector<size_t> get_mpo_ids(const MpoRefs &mpo);
    [[nodiscard]] static std::vector<size_t> get_mpo_sq_ids(const MpoRefs &mpo);
    [[nodiscard]] static MpsRefs              get_all_mps_refs(const StateFinite<Scalar> &state);
    [[nodiscard]] static MpoRefs              get_all_mpo_refs(const ModelFinite<Scalar> &model);
    static void                              assert_matching_sites(const std::vector<size_t> &sites, const std::vector<size_t> &other_sites);
    template<typename EnvRef>
    static void                               assert_matching_env_sites(const std::vector<size_t> &sites, const env_pair<EnvRef> &envs);
    [[nodiscard]] static MeasurementFiniteIds get_ids_energy_like(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs);
    [[nodiscard]] static MeasurementFiniteIds get_ids_h2_like(const MpsRefs &mps, const MpoRefs &mpo, const VarEnvs &envs);
    [[nodiscard]] static MeasurementFiniteIds get_ids_variance_like(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs_ene, const VarEnvs &envs_var);
    [[nodiscard]] static MeasurementFiniteIds get_ids_global_h1_like(const MpoRefs &mpo, const EneEnvs &envs);
    [[nodiscard]] static MeasurementFiniteIds get_ids_global_h2_like(const MpoRefs &mpo, const VarEnvs &envs);
    [[nodiscard]] static MeasurementFiniteIds get_ids_state_model_like(const MpsRefs &mps, const MpoRefs &mpo);
    [[nodiscard]] static MeasurementFiniteIds get_ids_local_norm_h1_like(const MpoRefs &mpo, const EneEnvs &envs, long long maxiter, double reltol);
    [[nodiscard]] static MeasurementFiniteIds get_ids_local_norm_h2_like(const MpoRefs &mpo, const VarEnvs &envs, long long maxiter, double reltol);
};

#include "MeasurementsTensorsFinite.tmpl.h"
