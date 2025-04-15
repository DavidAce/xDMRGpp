#pragma once

#include "config/enums.h"
#include "math/float.h"
#include <complex>
#include <measure/MeasurementsStateFinite.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename Scalar>
class MpsSite;
template<typename Scalar> class TensorsFinite;
/**
 * \class StateFinite
 *
 * \brief Stores the finite mps state and components to operate on it, such as environments and model mpo's.
 *
 * The finite state mps is always partitioned into two lists, corresponding to the two sides left and right of the current position.
 * Each site contains a "Vidal" site, i.e., a bond matrix \f$ \Lambda \f$ and an mps tensor \f$\Gamma \f$ in Vidal's notation.
 * In the left side, the sites are left normalized, with the bond matrix on the left \f$ \Lambda \Gamma \ = A\f$.
 * In the right side, the sites are right normalized, with the bond matrix on the right \f$ \Gamma \Lambda = B\f$.
 * At the center is the center bond matrix MPS_C, a.k.a. \f$ \Lambda^C \f$. For a state with 10 sites the layout is seen below
 *
 * \code
 * |Finite state> =  ---MPS_L(0)--- ... MPS_L(4)---MPS_L(5)---MPS_C---MPS_R(6)---MPS_R(7)---...---MPS_R(9)
 * \endcode
 *
 *  The numbers in parentheses denote the position in the chain, note that this isn't the same as the position in the corresponding containers.
 */

template<typename Scalar = cx64>
class StateFinite {
    private:
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    template<typename T>
    struct Cache {
        std::optional<Eigen::Tensor<T, 3>>                      multisite_mps = std::nullopt;
        std::deque<std::pair<std::string, Eigen::Tensor<T, 3>>> mps           = {};
        std::deque<std::pair<std::string, Eigen::Tensor<T, 4>>> trf           = {};
        std::unordered_map<std::string, Eigen::Tensor<T, 4>>    temporary_rho = {};
    };
    enum class FindStaleKeys { OFF, ON };

    template<typename T>
    struct TrfCacheEntry {
        using trfref    = std::reference_wrapper<const Eigen::Tensor<T, 4>>;
        size_t      pos = -1ul;
        std::string side;
        std::string key;
        size_t      ncontained = -1ul;
        size_t      nremaining = -1ul;
        double      cost       = std::numeric_limits<double>::quiet_NaN();
        trfref      trf;
        // std::vector<std::string> stale_keys; // Can be used to erase older cache entries
    };
    static constexpr size_t max_mps_cache_size = 20; // Max mps cache size in units of elements
    static constexpr size_t max_trf_cache_size = 20; // Max transfer matrix cache size in units of elements
    int                     direction          = 1;
    mutable Cache<fp32>     cache_fp32;
    mutable Cache<fp64>     cache_fp64;
    mutable Cache<cx32>     cache_cx32;
    mutable Cache<cx64>     cache_cx64;
    template<typename T>
    Cache<T> &get_cache();
    template<typename T>
    Cache<T>                 &get_cache() const;
    mutable std::vector<bool> tag_normalized_sites;
    std::string               name;
    AlgorithmType             algo = AlgorithmType::ANY;
    std::vector<double>       convrates;

    template<typename T>
    using optional_tensor4ref = std::optional<std::reference_wrapper<const Eigen::Tensor<T, 4>>>;
    template<typename T>
    using optional_tensor3ref = std::optional<std::reference_wrapper<const Eigen::Tensor<T, 3>>>;
    template<typename T>
    optional_tensor4ref<T> load_trf_from_cache(const std::string &key) const;
    template<typename T>
    optional_tensor4ref<T> load_trf_from_cache(const std::vector<size_t> &sites, size_t pos, std::string_view side) const;
    template<typename T>
    void save_trf_into_cache(const Eigen::Tensor<T, 4> &trf, const std::string &key) const;
    template<typename T>
    void save_trf_into_cache(const Eigen::Tensor<T, 4> &trf, const std::vector<size_t> &sites, size_t pos, std::string_view side) const;
    template<typename T>
    std::optional<TrfCacheEntry<T>> get_optimal_trf_from_cache(const std::vector<size_t> &sites, std::string_view side) const;
    template<typename T>
    std::optional<TrfCacheEntry<T>> get_optimal_trf_from_cache(const std::vector<size_t> &sites) const;
    template<typename T>
    optional_tensor3ref<T> get_cached_mps(const std::string &key) const;
    template<typename T>
    bool        has_cached_mps(const std::string &key) const;
    std::string generate_cache_key(const std::vector<size_t> &sites, const size_t pos, std::string_view side) const;
    template<typename T>
    double get_transfer_matrix_cost(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<T>> &trf_cache) const;

    public:
    std::vector<std::unique_ptr<MpsSite<Scalar>>> mps_sites;
    std::vector<size_t>                           active_sites;
    mutable MeasurementsStateFinite<Scalar>       measurements;
    size_t popcount = -1ul; /*!< Number of 1's or particles in the product state pattern. Used in the fLBIT algorithm, which conserves the particle number. */

    public:
    StateFinite();
    ~StateFinite() noexcept;                              // Read comment on implementation
    StateFinite(StateFinite &&other) noexcept;            // default move ctor
    StateFinite &operator=(StateFinite &&other) noexcept; // default move assign
    StateFinite(const StateFinite &other);                // copy ctor
    StateFinite &operator=(const StateFinite &other);     // copy assign
    StateFinite(AlgorithmType algo_type, size_t model_size, long position, long spin_dim = 2);
    void initialize(AlgorithmType algo_type, size_t model_size, long position, long spin_dim = 2);

    void                           set_name(std::string_view statename);
    [[nodiscard]] std::string_view get_name() const;

    void                        set_algorithm(const AlgorithmType &algo_type);
    [[nodiscard]] AlgorithmType get_algorithm() const;

    const Eigen::Tensor<Scalar, 1> &get_bond(long posL, long posR) const;
    const Eigen::Tensor<Scalar, 1> &get_midchain_bond() const;
    const Eigen::Tensor<Scalar, 1> &current_bond() const;

    template<typename T = size_t>
    [[nodiscard]] T get_length() const;
    template<typename T = size_t>
    [[nodiscard]] T get_position() const;

    void                                        set_positions();
    void                                        flip_direction();
    [[nodiscard]] int                           get_direction() const;
    [[nodiscard]] std::vector<std::string_view> get_labels() const;
    [[nodiscard]] std::array<long, 3>           dimensions_1site() const;
    [[nodiscard]] std::array<long, 3>           dimensions_2site() const;
    [[nodiscard]] std::array<long, 3>           dimensions_nsite() const;
    [[nodiscard]] long                          size_1site() const;
    [[nodiscard]] long                          size_2site() const;
    [[nodiscard]] long                          size_nsite() const;
    [[nodiscard]] long                          get_largest_bond() const;
    [[nodiscard]] long                          get_largest_bond(const std::vector<size_t> &sites) const;
    [[nodiscard]] double                        get_smallest_schmidt_value() const;

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
    [[nodiscard]] bool has_center_point() const;
    [[nodiscard]] bool is_real() const;
    [[nodiscard]] bool has_nan() const;

    void assert_validity() const;
    // For individual sites
    template<typename T = size_t>
    const MpsSite<Scalar> &get_mps_site(T pos) const;
    template<typename T = size_t>
    MpsSite<Scalar>       &get_mps_site(T pos);
    const MpsSite<Scalar> &get_mps_site() const;
    MpsSite<Scalar>       &get_mps_site();
    // For multiple sites
    void                                                       set_mps(const std::vector<MpsSite<Scalar>> &mps_list);
    std::vector<std::reference_wrapper<const MpsSite<Scalar>>> get_mps(const std::vector<size_t> &sites) const;
    std::vector<std::reference_wrapper<MpsSite<Scalar>>>       get_mps(const std::vector<size_t> &sites);
    std::vector<MpsSite<Scalar>>                               get_mps_copy(const std::vector<size_t> &sites);
    std::vector<std::reference_wrapper<const MpsSite<Scalar>>> get_mps_active() const;
    std::vector<std::reference_wrapper<MpsSite<Scalar>>>       get_mps_active();

    // For multisite
    std::array<long, 3>              active_dimensions() const;
    long                             active_problem_size() const;
    std::vector<long>                get_bond_dims(const std::vector<size_t> &sites) const;
    std::vector<long>                get_bond_dims_active() const;
    std::vector<long>                get_spin_dims(const std::vector<size_t> &sites) const;
    std::vector<long>                get_spin_dims() const;
    long                             get_spin_dim() const;
    std::vector<std::array<long, 3>> get_mps_dims(const std::vector<size_t> &sites) const;
    std::vector<std::array<long, 3>> get_mps_dims_active() const;
    template<typename T>
    Eigen::Tensor<T, 3> get_multisite_mps(const std::vector<size_t> &sites, bool use_cache = false) const;
    template<typename T>
    const Eigen::Tensor<T, 3> &get_multisite_mps() const;
    template<typename T>
    Eigen::Tensor<T, 2> get_reduced_density_matrix(const std::vector<size_t> &sites) const;
    template<typename T>
    std::array<double, 3> get_reduced_density_matrix_cost(const std::vector<size_t> &sites) const;
    template<typename T>
    Eigen::Tensor<T, 2> get_transfer_matrix(const std::vector<size_t> &sites, std::string_view side) const;
    template<typename T>
    std::array<double, 2> get_transfer_matrix_costs(const std::vector<size_t> &sites, std::string_view side) const;
    double                get_trf_cache_gbts() const;
    double                get_mps_cache_gbts() const;
    std::array<double, 2> get_cache_sizes() const;

    public:
    void                set_truncation_error(size_t pos, double error);
    void                set_truncation_error(double error);
    void                set_truncation_error_LC(double error);
    void                keep_max_truncation_errors(std::vector<double> &other_error);
    double              get_truncation_error(size_t pos) const;
    double              get_truncation_error() const;
    double              get_truncation_error_LC() const;
    double              get_truncation_error_midchain() const;
    std::vector<double> get_truncation_errors() const;
    std::vector<double> get_truncation_errors_active() const;
    double              get_truncation_error_active_max() const;

    size_t num_sites_truncated(double truncation_threshold) const;
    size_t num_bonds_at_limit(long bond_lim) const;
    bool   is_at_bond_limit(long bond_lim) const;
    size_t num_bonds_at_maximum(const std::vector<size_t> &sites) const;
    bool   is_truncated(double truncation_error_limit) const;
    void   clear_measurements(LogPolicy logPolicy = LogPolicy::SILENT) const;
    void   clear_cache(LogPolicy logPolicy = LogPolicy::SILENT) const;
    void   shrink_cache() const;

    void                     tag_active_sites_normalized(bool tag) const;
    void                     tag_all_sites_normalized(bool tag) const;
    void                     tag_site_normalized(size_t pos, bool tag) const;
    const std::vector<bool> &get_normalization_tags() const;
    bool                     is_normalized_on_all_sites() const;
    bool                     is_normalized_on_any_sites() const;
    bool                     is_normalized_on_active_sites() const;
    bool                     is_normalized_on_non_active_sites() const;
    std::vector<size_t>      get_active_ids() const;
};
