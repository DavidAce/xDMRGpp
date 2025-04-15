#pragma once

#include "config/enums.h"
#include "math/float.h"
#include "math/svd/config.h"
#include "measure/MeasurementsTensorsFinite.h"
#include "tensors/site/env/EnvPair.h"
#include <array>
#include <complex>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

class TensorsLocal;
template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
struct BondExpansionResult;
namespace tools::finite::opt {
    struct OptMeta;
}
template<typename Scalar = cx64>
class TensorsFinite {
    using OptMeta    = tools::finite::opt::OptMeta;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    private:
    template<typename T>
    struct Cache {
        std::optional<std::vector<size_t>> cached_sites_hamiltonian         = std::nullopt;
        std::optional<std::vector<size_t>> cached_sites_hamiltonian_squared = std::nullopt;
        std::optional<Eigen::Tensor<T, 2>> effective_hamiltonian            = std::nullopt;
        std::optional<Eigen::Tensor<T, 2>> effective_hamiltonian_squared    = std::nullopt;
    };

    mutable Cache<fp32> cache_fp32;
    mutable Cache<fp64> cache_fp64;
    mutable Cache<cx32> cache_cx32;
    mutable Cache<cx64> cache_cx64;
    template<typename T>
    Cache<T> &get_cache();
    template<typename T>
    Cache<T> &get_cache() const;

    public:
    std::unique_ptr<StateFinite<Scalar>> state;
    std::unique_ptr<ModelFinite<Scalar>> model;
    std::unique_ptr<EdgesFinite<Scalar>> edges;

    std::vector<size_t>                       active_sites;
    mutable MeasurementsTensorsFinite<Scalar> measurements;

    // This class should have these responsibilities:
    //  - Initialize/randomize the tensors
    //  - Move/manage center position
    //  - Rebuild edges
    //  - Activate sites
    //  - Manage caches

    TensorsFinite();
    ~TensorsFinite();                                     // Read comment on implementation
    TensorsFinite(TensorsFinite &&other);                 // default move ctor
    TensorsFinite &operator=(TensorsFinite &&other);      // default move assign
    TensorsFinite(const TensorsFinite &other);            // copy ctor
    TensorsFinite &operator=(const TensorsFinite &other); // copy assign
    TensorsFinite(AlgorithmType algo_type, ModelType model_type, size_t model_size, long position);

    StateFinite<Scalar>       &get_state();
    ModelFinite<Scalar>       &get_model();
    EdgesFinite<Scalar>       &get_edges();
    const StateFinite<Scalar> &get_state() const;
    const ModelFinite<Scalar> &get_model() const;
    const EdgesFinite<Scalar> &get_edges() const;

    void initialize(AlgorithmType algo_type, ModelType model_type, size_t model_size, long position);
    void initialize_model();
    void initialize_state(ResetReason reason, StateInit state_init, StateInitType state_type, std::string_view axis, bool use_eigenspinors, long bond_lim,
                          std::string &pattern);
    void normalize_state(std::optional<svd::config> svd_cfg = std::nullopt, NormPolicy policy = NormPolicy::IFNEEDED);
    /* clang-format off */
    template<typename T> [[nodiscard]] const Eigen::Tensor<T, 3>            &get_multisite_mps() const;
    template<typename T> [[nodiscard]] const Eigen::Tensor<T, 4>            &get_multisite_mpo() const;
    template<typename T> [[nodiscard]] const Eigen::Tensor<T, 4>            &get_multisite_mpo_squared() const;
    [[nodiscard]]                           env_pair<const Eigen::Tensor<Scalar, 3> &>   get_multisite_env_ene_blk() const;
    [[nodiscard]]                           env_pair<const Eigen::Tensor<Scalar, 3> &>   get_multisite_env_var_blk() const;
    template<typename T> [[nodiscard]] env_pair<Eigen::Tensor<T, 3>> get_multisite_env_ene_blk_as() const;
    template<typename T> [[nodiscard]] env_pair<Eigen::Tensor<T, 3>> get_multisite_env_var_blk_as() const;
    template<typename T> [[nodiscard]] const Eigen::Tensor<T, 2> &get_effective_hamiltonian() const;
    template<typename T> [[nodiscard]] const Eigen::Tensor<T, 2> &get_effective_hamiltonian_squared() const;
    /* clang-format on */

    void                                                     project_to_nearest_axis(std::string_view axis, std::optional<svd::config> svd_cfg = std::nullopt);
    void                                                     set_parity_shift_mpo(OptRitz ritz, std::string_view axis);
    void                                                     set_parity_shift_mpo_squared(std::string_view axis);
    void                                                     set_energy_shift_mpo(Scalar energy_shift);
    [[nodiscard]] std::tuple<OptRitz, int, std::string_view> get_parity_shift_mpo();
    [[nodiscard]] std::pair<int, std::string_view>           get_parity_shift_mpo_squared();
    [[nodiscard]] Scalar                                     get_energy_shift_mpo();

    void rebuild_mpo();
    void rebuild_mpo_squared();
    void compress_mpo_squared();

    void assert_validity() const;

    template<typename T = size_t>
    [[nodiscard]] T get_position() const;
    template<typename T = size_t>
    [[nodiscard]] T get_length() const;

    [[nodiscard]] bool is_real() const;
    [[nodiscard]] bool has_nan() const;
    [[nodiscard]] bool has_center_point() const;
    [[nodiscard]] bool position_is_the_middle() const;
    [[nodiscard]] bool position_is_the_middle_any_direction() const;
    [[nodiscard]] bool position_is_outward_edge_left(size_t nsite = 1) const;
    [[nodiscard]] bool position_is_outward_edge_right(size_t nsite = 1) const;
    [[nodiscard]] bool position_is_outward_edge(size_t nsite = 1) const;
    [[nodiscard]] bool position_is_inward_edge_left(size_t nsite = 1) const;
    [[nodiscard]] bool position_is_inward_edge_right(size_t nsite = 1) const;
    [[nodiscard]] bool position_is_inward_edge(size_t nsite = 1) const;
    [[nodiscard]] bool position_is_at(long pos) const;
    [[nodiscard]] bool position_is_at(long pos, int dir) const;
    [[nodiscard]] bool position_is_at(long pos, int dir, bool isCenter) const;

    void                sync_active_sites();
    void                clear_active_sites();
    void                activate_sites(const std::vector<size_t> &sites);
    void                activate_sites();
    void                activate_sites(long threshold, size_t max_sites, size_t min_sites = 1);
    std::array<long, 3> active_problem_dims() const;
    long                active_problem_size() const;
    size_t              move_center_point(std::optional<svd::config> svd_cfg = std::nullopt);
    size_t              move_center_point_to_pos(long pos, std::optional<svd::config> svd_cfg = std::nullopt);
    size_t              move_center_point_to_inward_edge(std::optional<svd::config> svd_cfg = std::nullopt);
    size_t              move_center_point_to_middle(std::optional<svd::config> svd_cfg = std::nullopt);
    void merge_multisite_mps(const Eigen::Tensor<cx64, 3> &multisite_tensor, MergeEvent mevent, std::optional<svd::config> svd_cfg = std::nullopt,
                             LogPolicy log_policy = LogPolicy::SILENT);

    BondExpansionResult<Scalar> expand_bonds(const OptMeta &optMeta);

    void move_site_mps(const size_t site, const long steps, std::vector<size_t> &sites_mps, std::optional<long> new_pos = std::nullopt);
    void move_site_mpo(const size_t site, const long steps, std::vector<size_t> &sites_mpo);
    void move_site_mps_to_pos(const size_t site, const long tgt_pos, std::vector<size_t> &sites_mps, std::optional<long> new_pos = std::nullopt);
    void move_site_mpo_to_pos(const size_t site, const long tgt_pos, std::vector<size_t> &sites_mpo);
    void move_site(const size_t site, const long steps, std::vector<size_t> &sites_mps, std::vector<size_t> &sites_mpo,
                   std::optional<long> new_pos = std::nullopt);
    void move_site_to_pos(const size_t site, const long tgt_pos, std::optional<std::vector<size_t>> &sites_mps, std::optional<std::vector<size_t>> &sites_mpo,
                          std::optional<long> new_pos = std::nullopt);

    void assert_edges() const;
    void assert_edges_ene() const;
    void assert_edges_var() const;
    void rebuild_edges();
    void rebuild_edges_ene();
    void rebuild_edges_var();
    void clear_measurements(LogPolicy logPolicy = LogPolicy::SILENT) const;
    void clear_cache(LogPolicy logPolicy = LogPolicy::SILENT) const;
};
