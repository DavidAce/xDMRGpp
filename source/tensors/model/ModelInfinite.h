#pragma once
#include "config/enums.h"
#include "math/float.h"
#include <complex>
#include <memory>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename Scalar>
class MpoSite;

template<typename Scalar>
class ModelInfinite {
    private:
    struct Cache {
        std::optional<Eigen::Tensor<Scalar, 4>> twosite_mpo_AB = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 4>> twosite_mpo_BA = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> twosite_ham_AB = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> twosite_ham_BA = std::nullopt;
    };
    mutable Cache                    cache;
    std::unique_ptr<MpoSite<Scalar>> HA; /*!< Left hamiltonian MPO */
    std::unique_ptr<MpoSite<Scalar>> HB; /*!< Right hamiltonian MPO */
    public:
    ModelType model_type = ModelType::ising_tf_rf;

    ModelInfinite();
    ~ModelInfinite();                                     // Read comment on implementation
    ModelInfinite(ModelInfinite &&other);                 // default move ctor
    ModelInfinite &operator=(ModelInfinite &&other);      // default move assign
    ModelInfinite(const ModelInfinite &other);            // copy ctor
    ModelInfinite &operator=(const ModelInfinite &other); // copy assign

    void                                  initialize(ModelType model_type_);
    void                                  randomize();
    void                                  reset_mpo_squared();
    void                                  rebuild_mpo_squared();
    std::vector<Eigen::Tensor<Scalar, 4>> get_compressed_mpo_squared();

    bool is_real() const;
    bool has_nan() const;
    void assert_validity() const;

    [[nodiscard]] const MpoSite<Scalar> &get_mpo_siteA() const;
    [[nodiscard]] const MpoSite<Scalar> &get_mpo_siteB() const;
    [[nodiscard]] MpoSite<Scalar>       &get_mpo_siteA();
    [[nodiscard]] MpoSite<Scalar>       &get_mpo_siteB();
    const Eigen::Tensor<Scalar, 4>      &get_2site_mpo_AB() const;
    const Eigen::Tensor<Scalar, 4>      &get_2site_mpo_BA() const;
    const Eigen::Tensor<Scalar, 2>      &get_2site_ham_AB() const;
    const Eigen::Tensor<Scalar, 2>      &get_2site_ham_BA() const;

    Eigen::DSizes<long, 4> dimensions() const;

    [[nodiscard]] bool   is_shifted() const;
    [[nodiscard]] Scalar get_energy_shift_per_site() const;
    void                 set_energy_shift_per_site(Scalar energy_shift_per_site);
    void                 clear_cache();
};
