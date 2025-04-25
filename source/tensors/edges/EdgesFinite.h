#pragma once
#include "config/enums.h"
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "tensors/site/env/EnvPair.h"
#include <complex>
#include <memory>
#include <optional>
#include <vector>

template<typename Scalar> class EnvEne;
template<typename Scalar> class EnvVar;

template<typename Scalar>
class EdgesFinite {
    public:
    std::vector<size_t> active_sites;

    private:
    std::vector<std::unique_ptr<EnvEne<Scalar>>> eneL;
    std::vector<std::unique_ptr<EnvEne<Scalar>>> eneR;
    std::vector<std::unique_ptr<EnvVar<Scalar>>> varL;
    std::vector<std::unique_ptr<EnvVar<Scalar>>> varR;

    public:
    EdgesFinite();
    ~EdgesFinite();                                       // Read comment on implementation
    EdgesFinite(EdgesFinite &&other) noexcept;            // default move ctor
    EdgesFinite &operator=(EdgesFinite &&other) noexcept; // default move assign
    EdgesFinite(const EdgesFinite &other);                // copy ctor
    EdgesFinite &operator=(const EdgesFinite &other);     // copy assign

    void initialize(size_t model_size);

    [[nodiscard]] const EnvEne<Scalar> &get_env_eneL(size_t pos) const;
    [[nodiscard]] const EnvEne<Scalar> &get_env_eneR(size_t pos) const;
    [[nodiscard]] const EnvVar<Scalar> &get_env_varL(size_t pos) const;
    [[nodiscard]] const EnvVar<Scalar> &get_env_varR(size_t pos) const;
    [[nodiscard]] EnvEne<Scalar>       &get_env_eneL(size_t pos);
    [[nodiscard]] EnvVar<Scalar>       &get_env_varR(size_t pos);
    [[nodiscard]] EnvVar<Scalar>       &get_env_varL(size_t pos);
    [[nodiscard]] EnvEne<Scalar>       &get_env_eneR(size_t pos);

    [[nodiscard]] size_t get_length() const;
    [[nodiscard]] bool   is_real() const;
    [[nodiscard]] bool   has_nan() const;
    void                 assert_validity() const;

    void eject_edges_inactive_ene(std::optional<std::vector<size_t>> sites = std::nullopt);
    void eject_edges_inactive_var(std::optional<std::vector<size_t>> sites = std::nullopt);
    void eject_edges_inactive(std::optional<std::vector<size_t>> sites = std::nullopt);

    void eject_edges_all_ene();
    void eject_edges_all_var();
    void eject_edges_all();

    [[nodiscard]] env_pair<const EnvEne<Scalar> &> get_env_ene(size_t posL, size_t posR) const;
    [[nodiscard]] env_pair<const EnvVar<Scalar> &> get_env_var(size_t posL, size_t posR) const;
    [[nodiscard]] env_pair<EnvEne<Scalar> &>       get_env_ene(size_t posL, size_t posR);
    [[nodiscard]] env_pair<EnvVar<Scalar> &>       get_env_var(size_t posL, size_t posR);
    [[nodiscard]] env_pair<const EnvEne<Scalar> &> get_ene_active() const;
    [[nodiscard]] env_pair<const EnvVar<Scalar> &> get_var_active() const;
    [[nodiscard]] env_pair<EnvEne<Scalar> &>       get_ene_active();
    [[nodiscard]] env_pair<EnvVar<Scalar> &>       get_var_active();

    [[nodiscard]] env_pair<const EnvEne<Scalar> &> get_env_ene(size_t pos) const;
    [[nodiscard]] env_pair<const EnvVar<Scalar> &> get_env_var(size_t pos) const;
    [[nodiscard]] env_pair<EnvEne<Scalar> &>       get_env_ene(size_t pos);
    [[nodiscard]] env_pair<EnvVar<Scalar> &>       get_env_var(size_t pos);

    [[nodiscard]] env_pair<const EnvEne<Scalar> &> get_multisite_env_ene(std::optional<std::vector<size_t>> sites = std::nullopt) const;
    [[nodiscard]] env_pair<const EnvVar<Scalar> &> get_multisite_env_var(std::optional<std::vector<size_t>> sites = std::nullopt) const;
    [[nodiscard]] env_pair<EnvEne<Scalar> &>       get_multisite_env_ene(std::optional<std::vector<size_t>> sites = std::nullopt);
    [[nodiscard]] env_pair<EnvVar<Scalar> &>       get_multisite_env_var(std::optional<std::vector<size_t>> sites = std::nullopt);

    /* clang-format off */
    [[nodiscard]] env_pair<const Eigen::Tensor<Scalar, 3> &> get_env_ene_blk(size_t posL, size_t posR) const;
    [[nodiscard]] env_pair<const Eigen::Tensor<Scalar, 3> &> get_env_var_blk(size_t posL, size_t posR) const;
    [[nodiscard]] env_pair<Eigen::Tensor<Scalar, 3> &>       get_env_ene_blk(size_t posL, size_t posR);
    [[nodiscard]] env_pair<Eigen::Tensor<Scalar, 3> &>       get_env_var_blk(size_t posL, size_t posR);
    template<typename T> [[nodiscard]] env_pair<Eigen::Tensor<T, 3>> get_env_ene_blk_as(size_t posL, size_t posR) const;
    template<typename T> [[nodiscard]] env_pair<Eigen::Tensor<T, 3>> get_env_var_blk_as(size_t posL, size_t posR) const;

    [[nodiscard]] env_pair<const Eigen::Tensor<Scalar, 3> &> get_multisite_env_ene_blk(std::optional<std::vector<size_t>> sites = std::nullopt) const;
    [[nodiscard]] env_pair<const Eigen::Tensor<Scalar, 3> &> get_multisite_env_var_blk(std::optional<std::vector<size_t>> sites = std::nullopt) const;
    [[nodiscard]] env_pair<Eigen::Tensor<Scalar, 3> &>       get_multisite_env_ene_blk(std::optional<std::vector<size_t>> sites = std::nullopt);
    [[nodiscard]] env_pair<Eigen::Tensor<Scalar, 3> &>       get_multisite_env_var_blk(std::optional<std::vector<size_t>> sites = std::nullopt);

    template<typename T> [[nodiscard]] env_pair<Eigen::Tensor<T, 3>> get_multisite_env_ene_blk_as(std::optional<std::vector<size_t>> sites = std::nullopt) const;
    template<typename T> [[nodiscard]] env_pair<Eigen::Tensor<T, 3>> get_multisite_env_var_blk_as(std::optional<std::vector<size_t>> sites = std::nullopt) const;
    /* clang-format on */

    [[nodiscard]] std::pair<std::vector<size_t>, std::vector<size_t>> get_active_ids() const;

    [[nodiscard]] std::array<long, 3> get_dims_eneL(size_t pos) const;
    [[nodiscard]] std::array<long, 3> get_dims_eneR(size_t pos) const;
    [[nodiscard]] std::array<long, 3> get_dims_varL(size_t pos) const;
    [[nodiscard]] std::array<long, 3> get_dims_varR(size_t pos) const;
};
