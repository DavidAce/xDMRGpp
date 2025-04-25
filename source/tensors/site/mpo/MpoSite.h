#pragma once
#include "config/enums.h"
#include "general/sfinae.h"
#include "h5tb/h5tb.h"
#include "math/float.h"
#include <any>
#include <map>
#include <math/tenx.h>
#include <memory>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>

namespace h5pp {
    class File;
}
template<typename Scalar>
class MpoSite {
    public:
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using QuadScalar = std::conditional_t<sfinae::is_std_complex_v<Scalar>, cx128, fp128>;
    using TableMap   = std::map<std::string, std::any>;
    const ModelType model_type;
    bool            all_mpo_parameters_have_been_set = false;

    protected:
    mutable std::optional<std::size_t> unique_id;
    mutable std::optional<std::size_t> unique_id_sq;

    private:
    mutable std::optional<bool> is_real_cached = std::nullopt;
    mutable std::optional<bool> has_nan_cached = std::nullopt;

    protected:
    // Common parameters
    std::optional<size_t>                   position;                               /*!< Position on a finite chain */
    Scalar                                  energy_shift_mpo       = Scalar{};      /*!< Energy shift for this mpo (to make energy-shifted MPO views) */
    OptRitz                                 parity_shift_ritz_mpo  = OptRitz::NONE; /*!< Shift direction depending on ritz (SR:-1, LR:+1) */
    int                                     parity_shift_sign_mpo  = 0;             /*!< Sign of parity sector to shift for the MPO*/
    std::string                             parity_shift_axus_mpo  = {};            /*!< Unsigned axis {x,y,z} of spin parity sector to shift for the MPO */
    int                                     parity_shift_sign_mpo2 = 0;             /*!< Sign of parity sector to shift for the MPO²*/
    std::string                             parity_shift_axus_mpo2 = {};            /*!< Unsigned axis {x,y,z} of spin parity sector to shift for the MPO² */
    std::array<long, 4>                     extent4{};                              /*!< Extent of pauli matrices in a rank-4 tensor */
    std::array<long, 2>                     extent2{};                              /*!< Extent of pauli matrices in a rank-2 tensor */
    Eigen::Tensor<Scalar, 4>                mpo_internal;
    Eigen::Tensor<QuadScalar, 4>            mpo_internal_q;
    std::optional<Eigen::Tensor<Scalar, 4>> mpo_squared               = std::nullopt;
    double                                  global_energy_upper_bound = 100.0;
    double                                  local_energy_upper_bound  = 0.0;

    virtual Eigen::Tensor<Scalar, 4>     get_mpo(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody = std::nullopt,
                                                 std::optional<std::vector<size_t>> skip = std::nullopt) const = 0;
    virtual Eigen::Tensor<QuadScalar, 4> get_mpo_q(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody = std::nullopt,
                                                   std::optional<std::vector<size_t>> skip = std::nullopt) const;

    template<typename T>
    [[nodiscard]] Eigen::Tensor<T, 1> get_MPO_edge_left(const Eigen::Tensor<T, 4> &mpo) const;
    template<typename T>
    [[nodiscard]] Eigen::Tensor<T, 1> get_MPO_edge_right(const Eigen::Tensor<T, 4> &mpo) const;
    template<typename T>
    [[nodiscard]] Eigen::Tensor<T, 4> apply_edge_left(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeL) const;
    template<typename T>
    [[nodiscard]] Eigen::Tensor<T, 4> apply_edge_right(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeR) const;

    public:
    explicit MpoSite(ModelType model_type_, size_t position_);
    virtual ~MpoSite() = default;

    void build_mpo();
    void build_mpo_q();
    void set_position(size_t new_pos);
    void assert_validity() const;
    void set_energy_shift_mpo(Scalar site_energy);
    void set_energy_shift_mpo2(Scalar site_energy);
    template<typename T>
    Eigen::Tensor<T, 4>                               get_parity_shifted_mpo(const Eigen::Tensor<T, 4> &mpo_build) const;
    Eigen::Tensor<Scalar, 4>                          get_parity_shifted_mpo_squared(const Eigen::Tensor<Scalar, 4> &mpo_build) const;
    void                                              set_parity_shift_mpo(OptRitz ritz, int sign, std::string_view axis);
    std::tuple<OptRitz, int, std::string_view>        get_parity_shift_mpo() const;
    void                                              set_parity_shift_mpo_squared(int sign, std::string_view axis);
    std::pair<int, std::string_view>                  get_parity_shift_mpo_squared() const;
    void                                              build_mpo_squared();
    void                                              set_mpo(const Eigen::Tensor<Scalar, 4> &mpo_sq);
    void                                              set_mpo_squared(const Eigen::Tensor<Scalar, 4> &mpo_sq);
    void                                              clear_mpo();
    void                                              clear_mpo_squared();
    [[nodiscard]] bool                                has_mpo() const;
    [[nodiscard]] bool                                has_mpo_squared() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 4>            get_non_compressed_mpo_squared() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 4>     &MPO() const;
    [[nodiscard]] const Eigen::Tensor<QuadScalar, 4> &MPO_q() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 4>            MPO_energy_shifted_view(Scalar energy_shift_per_site) const;
    [[nodiscard]] Eigen::Tensor<QuadScalar, 4>        MPO_energy_shifted_view_q(Scalar energy_shift_per_site) const;
    [[nodiscard]] Eigen::Tensor<Scalar, 4>            MPO_nbody_view(std::optional<std::vector<size_t>> nbody,
                                                                     std::optional<std::vector<size_t>> skip = std::nullopt) const;
    [[nodiscard]] Eigen::Tensor<QuadScalar, 4>        MPO_nbody_view_q(std::optional<std::vector<size_t>> nbody,
                                                                       std::optional<std::vector<size_t>> skip = std::nullopt) const;

    [[nodiscard]] const Eigen::Tensor<Scalar, 4> &MPO2() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 4>        MPO2_nbody_view(std::optional<std::vector<size_t>> nbody,
                                                                  std::optional<std::vector<size_t>> skip = std::nullopt) const;

    [[nodiscard]] size_t                           get_position() const;
    [[nodiscard]] std::vector<std::string>         get_parameter_names() const;
    [[nodiscard]] std::vector<std::any>            get_parameter_values() const;
    [[nodiscard]] bool                             is_real() const;
    [[nodiscard]] bool                             has_nan() const;
    [[nodiscard]] bool                             has_energy_shifted_mpo() const;
    [[nodiscard]] bool                             has_parity_shifted_mpo() const;
    [[nodiscard]] bool                             has_parity_shifted_mpo2() const;
    [[nodiscard]] bool                             has_compressed_mpo_squared() const;
    [[nodiscard]] Scalar                           get_energy_shift_mpo() const;
    [[nodiscard]] Scalar                           get_energy_shift_mpo2() const;
    [[nodiscard]] double                           get_global_energy_upper_bound() const;
    [[nodiscard]] double                           get_local_energy_upper_bound() const;
    [[nodiscard]] long                             size() const;
    [[nodiscard]] std::array<long, 4>              dimensions() const;
    [[nodiscard]] virtual std::unique_ptr<MpoSite> clone() const                                                             = 0;
    [[nodiscard]] virtual long                     get_spin_dimension() const                                                = 0;
    [[nodiscard]] virtual TableMap                 get_parameters() const                                                    = 0;
    [[nodiscard]] virtual std::any                 get_parameter(std::string_view name) const                                = 0;
    virtual void                                   set_parameter(std::string_view name, std::any value)                      = 0;
    virtual void                                   print_parameter_names() const                                             = 0;
    virtual void                                   print_parameter_values() const                                            = 0;
    virtual void                                   set_parameters(TableMap &parameters)                                      = 0;
    virtual void                                   randomize_hamiltonian()                                                   = 0;
    virtual void                                   set_averages(std::vector<TableMap> all_parameters, bool infinite = false) = 0;
    virtual void                                   save_hamiltonian(h5pp::File &file, std::string_view model_prefix) const   = 0;
    virtual void                                   load_hamiltonian(const h5pp::File &file, std::string_view model_prefix)   = 0;
    void                                           save_mpo(h5pp::File &file, std::string_view model_prefix) const;
    void                                           load_mpo(const h5pp::File &file, std::string_view model_prefix);
    std::size_t                                    get_unique_id() const;
    std::size_t                                    get_unique_id_sq() const;

    template<typename T>
    [[nodiscard]] decltype(auto) MPO_as() const {
        if constexpr(tenx::sfinae::is_quadruple_prec_v<T>) {
            return tenx::asScalarType<T>(MPO_q());
        } else {
            return tenx::asScalarType<T>(MPO());
        }
    }

    template<typename T>
    [[nodiscard]] decltype(auto) MPO_energy_shifted_view_as(Scalar energy_shift_per_site) const {
        if constexpr(tenx::sfinae::is_quadruple_prec_v<T>) {
            return tenx::asScalarType<T>(MPO_energy_shifted_view_q(energy_shift_per_site));
        } else {
            return tenx::asScalarType<T>(MPO_energy_shifted_view(energy_shift_per_site));
        }
    }

    template<typename T>
    [[nodiscard]] decltype(auto) MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip = std::nullopt) const {
        if constexpr(tenx::sfinae::is_quadruple_prec_v<T>) {
            return tenx::asScalarType<T>(MPO_nbody_view_q(nbody, skip));
        } else {
            return tenx::asScalarType<T>(MPO_nbody_view(nbody, skip));
        }
    }

    template<typename T>
    [[nodiscard]] decltype(auto) MPO2_as() const {
        return tenx::asScalarType<T>(MPO2());
    }

    template<typename T>
    [[nodiscard]] decltype(auto) MPO2_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip = std::nullopt) const {
        return tenx::asScalarType<T>(MPO2_nbody_view(nbody, skip));
    }

    template<typename T>
    [[nodiscard]] decltype(auto) get_MPO_edge_left() const {
        if constexpr(tenx::sfinae::is_quadruple_prec_v<T>) {
            return tenx::asScalarType<T>(get_MPO_edge_left(mpo_internal_q));
        } else {
            return tenx::asScalarType<T>(get_MPO_edge_left(mpo_internal));
        }
    }

    template<typename T>
    [[nodiscard]] decltype(auto) get_MPO_edge_right() const {
        if constexpr(tenx::sfinae::is_quadruple_prec_v<T>) {
            return tenx::asScalarType<T>(get_MPO_edge_right(mpo_internal_q));
        } else {
            return tenx::asScalarType<T>(get_MPO_edge_right(mpo_internal));
        }
    }

    template<typename T>
    [[nodiscard]] Eigen::Tensor<T, 1> get_MPO2_edge_left() const;
    template<typename T>
    [[nodiscard]] Eigen::Tensor<T, 1> get_MPO2_edge_right() const;
};
