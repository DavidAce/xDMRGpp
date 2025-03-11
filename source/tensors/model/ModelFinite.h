#pragma once
#include "config/enums.h"
#include "math/float.h"
#include "math/svd/config.h"
#include <any>
#include <array>
#include <complex>
#include <memory>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>

class MpoSite;
class TensorsFinite;
class ModelLocal;

class ModelFinite {
    private:
    friend TensorsFinite;
    template<typename T>
    struct Cache {
        std::optional<std::vector<size_t>>                   cached_sites          = std::nullopt;
        std::optional<Eigen::Tensor<T, 4>>                   multisite_mpo         = std::nullopt;
        std::optional<Eigen::Tensor<T, 2>>                   multisite_ham         = std::nullopt;
        std::optional<Eigen::Tensor<T, 4>>                   multisite_mpo_squared = std::nullopt;
        std::optional<Eigen::Tensor<T, 2>>                   multisite_ham_squared = std::nullopt;
        std::unordered_map<std::string, Eigen::Tensor<T, 4>> multisite_mpo_temps; // Keeps previous results for reuse
        std::optional<std::vector<size_t>>                   multisite_mpo_ids         = std::nullopt;
        std::optional<std::vector<size_t>>                   multisite_mpo_squared_ids = std::nullopt;
        // std::unordered_map<std::string, Eigen::Tensor<cx128, 4>> multisite_mpo_t_temps; // Keeps previous results for reuse
        // std::optional<Eigen::Tensor<cx128, 4>>                   multisite_mpo_t = std::nullopt;
        // std::optional<Eigen::Tensor<cx128, 2>>                   multisite_ham_t = std::nullopt;
        // std::unordered_map<std::string, Eigen::Tensor<cx64, 4>>   multisite_mpo_squared_temps;   // Keeps previous results for reuse
        // std::unordered_map<std::string, Eigen::Tensor<cx128, 4>> multisite_mpo_squared_t_temps; // Keeps previous results for reuse
    };

    mutable Cache<fp32>  cache_fp32;
    mutable Cache<fp64>  cache_fp64;
    mutable Cache<fp128> cache_fp128;
    mutable Cache<cx32>  cache_cx32;
    mutable Cache<cx64>  cache_cx64;
    mutable Cache<cx128> cache_cx128;
    template<typename T>
    Cache<T> &get_cache();
    template<typename T>
    Cache<T> &get_cache() const;

    //    std::vector<Eigen::Tensor<cx64, 4>> get_compressed_mpos(std::vector<Eigen::Tensor<cx64, 4>> mpos);
    void                                                     randomize();
    [[nodiscard]] bool                                       has_mpo() const;
    [[nodiscard]] bool                                       has_mpo_squared() const;
    void                                                     clear_mpo_squared();
    void                                                     set_energy_shift_mpo(double energy_shift);
    void                                                     set_parity_shift_mpo(OptRitz, int sign, std::string_view axis);
    void                                                     set_parity_shift_mpo_squared(int sign, std::string_view axis);
    [[nodiscard]] std::tuple<OptRitz, int, std::string_view> get_parity_shift_mpo() const;
    [[nodiscard]] std::pair<int, std::string_view>           get_parity_shift_mpo_squared() const;
    [[nodiscard]] bool                                       has_parity_shifted_mpo() const;
    [[nodiscard]] bool                                       has_parity_shifted_mpo_squared() const;

    public:
    std::vector<std::unique_ptr<MpoSite>> MPO; /*!< A list of stored Hamiltonian MPO tensors,indexed by chain position. */
    std::vector<size_t>                   active_sites;
    ModelType                             model_type = ModelType::ising_tf_rf;

    public:
    ModelFinite();
    ModelFinite(ModelType model_type_, size_t model_size);
    ~ModelFinite();                                   // Read comment on implementation
    ModelFinite(ModelFinite &&other);                 // default move ctor
    ModelFinite &operator=(ModelFinite &&other);      // default move assign
    ModelFinite(const ModelFinite &other);            // copy ctor
    ModelFinite &operator=(const ModelFinite &other); // copy assign

    void                                                             initialize(ModelType model_type_, size_t model_size);
    void                                                             assert_validity() const;
    [[nodiscard]] size_t                                             get_length() const;
    [[nodiscard]] bool                                               is_real() const;
    [[nodiscard]] bool                                               has_nan() const;
    [[nodiscard]] const MpoSite                                     &get_mpo(size_t pos) const;
    [[nodiscard]] MpoSite                                           &get_mpo(size_t pos);
    void                                                             build_mpo();
    void                                                             build_mpo_squared();
    void                                                             compress_mpo();
    void                                                             compress_mpo_squared();
    [[nodiscard]] std::vector<std::reference_wrapper<const MpoSite>> get_mpo(const std::vector<size_t> &sites) const;
    [[nodiscard]] std::vector<std::reference_wrapper<MpoSite>>       get_mpo(const std::vector<size_t> &sites);
    [[nodiscard]] std::vector<std::reference_wrapper<const MpoSite>> get_mpo_active() const;
    [[nodiscard]] std::vector<std::reference_wrapper<MpoSite>>       get_mpo_active();
    [[nodiscard]] std::vector<Eigen::Tensor<cx64, 4>>                get_all_mpo_tensors(MposWithEdges withEdges = MposWithEdges::OFF);
    [[nodiscard]] std::vector<Eigen::Tensor<cx128, 4>>               get_all_mpo_tensors_t(MposWithEdges withEdges = MposWithEdges::OFF);
    [[nodiscard]] std::vector<Eigen::Tensor<cx64, 4>>                get_compressed_mpos(MposWithEdges withEdges = MposWithEdges::OFF);
    [[nodiscard]] std::vector<Eigen::Tensor<cx64, 4>>                get_compressed_mpos_squared(MposWithEdges withEdges = MposWithEdges::OFF);
    [[nodiscard]] std::vector<Eigen::Tensor<cx64, 4>>                get_mpos_energy_shifted_view(double energy_per_site) const;
    // [[nodiscard]] std::vector<Eigen::Tensor<cx64, 4>>                get_mpos_squared_shifted_view(double energy_per_site, MposWithEdges withEdges =
    // MposWithEdges::OFF) const;

    [[nodiscard]] bool                  has_energy_shifted_mpo() const; // For shifted energy MPO's
    [[nodiscard]] double                get_energy_shift_mpo() const;
    [[nodiscard]] double                get_energy_shift_mpo_per_site() const;
    [[nodiscard]] bool                  has_compressed_mpo_squared() const;
    [[nodiscard]] std::vector<std::any> get_parameter(std::string_view fieldname);
    [[nodiscard]] double                get_energy_upper_bound() const;

    // For local operations
    ModelLocal get_local(const std::vector<size_t> &sites) const;
    ModelLocal get_local() const;

    // For multisite
    [[nodiscard]] std::array<long, 4> active_dimensions() const;
    [[nodiscard]] std::array<long, 4> active_dimensions_squared() const;

    /* clang-format off */
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 4>   get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody = std::nullopt, bool with_edgeL = false, bool with_edgeR = false) const;
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 2>   get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody = std::nullopt) const;
    template<typename Scalar> [[nodiscard]] const Eigen::Tensor<Scalar, 4>   &get_multisite_mpo() const;
    template<typename Scalar> [[nodiscard]] const Eigen::Tensor<Scalar, 2>   &get_multisite_ham() const;

    [[nodiscard]] Eigen::Tensor<cx128, 4>  get_multisite_mpo_t(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody = std::nullopt, bool with_edgeL = false, bool with_edgeR = false) const;
    [[nodiscard]] Eigen::Tensor<cx128, 2>  get_multisite_ham_t(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody = std::nullopt) const;
    [[nodiscard]] const Eigen::Tensor<cx128, 4> &get_multisite_mpo_t() const;
    [[nodiscard]] const Eigen::Tensor<cx128, 2> &get_multisite_ham_t() const;

    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 4>        get_multisite_mpo_shifted_view(double energy_per_site) const;
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 4>        get_multisite_mpo_squared_shifted_view(double energy_per_site) const;
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 4>        get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody = std::nullopt) const;
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 2>        get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody = std::nullopt) const;
    template<typename Scalar> [[nodiscard]] const Eigen::Tensor<Scalar, 4> &get_multisite_mpo_squared() const;
    template<typename Scalar> [[nodiscard]] const Eigen::Tensor<Scalar, 2> &get_multisite_ham_squared() const;
    /* clang-format on */

    void clear_cache(LogPolicy logPolicy = LogPolicy::SILENT) const;
    void clear_cache_squared(LogPolicy logPolicy = LogPolicy::SILENT) const;

    [[nodiscard]] std::vector<size_t> get_active_ids() const;
    [[nodiscard]] std::vector<size_t> get_active_ids_sq() const;
};
