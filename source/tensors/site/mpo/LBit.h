#pragma once

#include "math/tenx/fwd_decl.h"
#include "MpoSite.h"
#include <general/sfinae.h>
#include <h5pp/details/h5ppHid.h>

template<typename Scalar>
class LBit : public MpoSite<Scalar> {
    private:
    using RealScalar = typename MpoSite<Scalar>::RealScalar;
    using QuadScalar = std::conditional_t<sfinae::is_std_complex_v<Scalar>, cx128, fp128>;
    using MpoSite<Scalar>::extent2;
    using MpoSite<Scalar>::extent4;
    using MpoSite<Scalar>::all_mpo_parameters_have_been_set;
    using MpoSite<Scalar>::local_energy_upper_bound;
    using MpoSite<Scalar>::global_energy_upper_bound;
    using MpoSite<Scalar>::position;
    using MpoSite<Scalar>::get_position;
    using MpoSite<Scalar>::unique_id;
    using MpoSite<Scalar>::unique_id_sq;
    using MpoSite<Scalar>::mpo_internal;
    using MpoSite<Scalar>::mpo_squared;
    using MpoSite<Scalar>::build_mpo;
    using MpoSite<Scalar>::build_mpo_squared;
    using MpoSite<Scalar>::build_mpo_q;
    using TableMap = typename MpoSite<Scalar>::TableMap;

    h5tb_lbit                    h5tb;
    Eigen::Tensor<Scalar, 4>     get_mpo(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody = std::nullopt,
                                         std::optional<std::vector<size_t>> skip = std::nullopt) const final;
    Eigen::Tensor<QuadScalar, 4> get_mpo_q(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody = std::nullopt,
                                           std::optional<std::vector<size_t>> skip = std::nullopt) const final;

    public:
    LBit(ModelType model_type_, size_t position_);
    [[nodiscard]] std::unique_ptr<MpoSite<Scalar>> clone() const final;
    [[nodiscard]] long                             get_spin_dimension() const final;
    [[nodiscard]] TableMap                         get_parameters() const final;
    [[nodiscard]] std::any                         get_parameter(std::string_view name) const final;

    //    [[nodiscard]] Eigen::MatrixXcd single_site_hamiltonian(size_t position, size_t sites, std::vector<Eigen::MatrixXcd> &SX, std::vector<Eigen::MatrixXcd>
    //    &SY,
    //                                                           std::vector<Eigen::MatrixXcd> &SZ) const final;
    //
    void set_parameter(std::string_view name, std::any value) final;
    void print_parameter_names() const final;
    void print_parameter_values() const final;
    void set_parameters(TableMap &parameters) final;
    void randomize_hamiltonian() final;
    void set_averages(std::vector<TableMap> all_parameters, bool infinite = false) final;

    void save_hamiltonian(h5pp::File &file, std::string_view table_path) const final;
    void load_hamiltonian(const h5pp::File &file, std::string_view model_prefix) final;
};
