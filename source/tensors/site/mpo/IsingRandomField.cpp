#include "IsingRandomField.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <h5pp/h5pp.h>

template class IsingRandomField<fp32>;
template class IsingRandomField<fp64>;
template class IsingRandomField<fp128>;
template class IsingRandomField<cx32>;
template class IsingRandomField<cx64>;
template class IsingRandomField<cx128>;

template<typename Scalar>
IsingRandomField<Scalar>::IsingRandomField(ModelType model_type_, size_t position_) : MpoSite<Scalar>(model_type_, position_) {
    h5tb.param.J1           = settings::model::ising_tf_rf::J1;
    h5tb.param.J2           = settings::model::ising_tf_rf::J2;
    h5tb.param.h_tran       = settings::model::ising_tf_rf::h_tran;
    h5tb.param.h_mean       = settings::model::ising_tf_rf::h_mean;
    h5tb.param.h_wdth       = settings::model::ising_tf_rf::h_wdth;
    h5tb.param.spin_dim     = settings::model::ising_tf_rf::spin_dim;
    h5tb.param.distribution = settings::model::ising_tf_rf::distribution;
    extent4                 = {1, 1, h5tb.param.spin_dim, h5tb.param.spin_dim};
    extent2                 = {h5tb.param.spin_dim, h5tb.param.spin_dim};
    using namespace qm::spin::half;
    qm::spin::half::SX = qm::spin::gen_manybody_spins(sx, 2);
    qm::spin::half::SY = qm::spin::gen_manybody_spins(sy, 2);
    qm::spin::half::SZ = qm::spin::gen_manybody_spins(sz, 2);
    qm::spin::half::II = qm::spin::gen_manybody_spins(id, 2);
    all_mpo_parameters_have_been_set =
        false; // There are no full lattice parameters, but we set it to true here since the model is not supposed to be randomized per site
}

template<typename Scalar>
double IsingRandomField<Scalar>::get_field() const {
    return h5tb.param.h_tran + h5tb.param.h_rand;
}
template<typename Scalar>
double IsingRandomField<Scalar>::get_coupling() const {
    return h5tb.param.J1;
}
template<typename Scalar>
void IsingRandomField<Scalar>::print_parameter_names() const {
    h5tb.print_parameter_names();
}
template<typename Scalar>
void IsingRandomField<Scalar>::print_parameter_values() const {
    h5tb.print_parameter_values();
}

template<typename Scalar>
void IsingRandomField<Scalar>::set_parameters(TableMap &parameters) {
    h5tb.param.J1           = std::any_cast<double>(parameters["J1"]);
    h5tb.param.J2           = std::any_cast<double>(parameters["J2"]);
    h5tb.param.h_tran       = std::any_cast<double>(parameters["h_tran"]);
    h5tb.param.h_mean       = std::any_cast<double>(parameters["h_mean"]);
    h5tb.param.h_wdth       = std::any_cast<double>(parameters["h_wdth"]);
    h5tb.param.h_rand       = std::any_cast<double>(parameters["h_rand"]);
    h5tb.param.spin_dim     = std::any_cast<long>(parameters["spin_dim"]);
    h5tb.param.distribution = std::any_cast<h5pp::vstr_t>(parameters["distribution"]);

    if(h5tb.param.J2 != 0.0) throw except::runtime_error("mpo({}): use of [J2] - Next-nearest neighbor coupling - is not implemented yet", get_position());
    all_mpo_parameters_have_been_set = true;
}

template<typename Scalar>
IsingRandomField<Scalar>::TableMap IsingRandomField<Scalar>::get_parameters() const {
    TableMap parameters;
    parameters["J1"]           = h5tb.param.J1;
    parameters["J2"]           = h5tb.param.J2;
    parameters["h_tran"]       = h5tb.param.h_tran;
    parameters["h_mean"]       = h5tb.param.h_mean;
    parameters["h_wdth"]       = h5tb.param.h_wdth;
    parameters["h_rand"]       = h5tb.param.h_rand;
    parameters["spin_dim"]     = h5tb.param.spin_dim;
    parameters["distribution"] = h5tb.param.distribution;
    return parameters;
}

template<typename Scalar>
std::any IsingRandomField<Scalar>::get_parameter(std::string_view name) const {
    /* clang-format off */
   if     (name == "J1")            return  h5tb.param.J1;
   else if(name == "J2")            return  h5tb.param.J2;
   else if(name == "h_tran")        return  h5tb.param.h_tran;
   else if(name == "h_mean")        return  h5tb.param.h_mean;
   else if(name == "h_wdth")        return  h5tb.param.h_wdth;
   else if(name == "h_rand")        return  h5tb.param.h_rand;
   else if(name == "spin_dim")      return  h5tb.param.spin_dim;
   else if(name == "distribution")  return  h5tb.param.distribution;
    /* clang-format on */
    throw except::logic_error("Invalid parameter name for IsingRandomField model: {}", name);
}

template<typename Scalar>
void IsingRandomField<Scalar>::set_parameter(const std::string_view name, std::any value) {
    /* clang-format off */
    if(name      == "J1")               h5tb.param.J1 = std::any_cast<decltype(h5tb.param.J1)>(value);
    else if(name == "J2")               h5tb.param.J2 = std::any_cast<decltype(h5tb.param.J2)>(value);
    else if(name == "h_tran")           h5tb.param.h_tran = std::any_cast<decltype(h5tb.param.h_tran)>(value);
    else if(name == "h_mean")           h5tb.param.h_mean = std::any_cast<decltype(h5tb.param.h_mean)>(value);
    else if(name == "h_wdth")           h5tb.param.h_wdth = std::any_cast<decltype(h5tb.param.h_wdth)>(value);
    else if(name == "h_rand")           h5tb.param.h_rand = std::any_cast<decltype(h5tb.param.h_rand)>(value);
    else if(name == "spin_dim")         h5tb.param.spin_dim = std::any_cast<decltype(h5tb.param.spin_dim)>(value);
    else if(name == "distribution")     h5tb.param.distribution = std::any_cast<decltype(h5tb.param.distribution)>(value);
    else
        /* clang-format on */
        throw except::logic_error("Invalid parameter name for the IsingRandomField model: {}", name);
    build_mpo();
    build_mpo_squared();
}

/*! Builds the MPO hamiltonian as a rank 4 tensor. Notation following Schollwöck (2010)
 * H = - Σ J sz_{i} sz_{i+1} +  g_{i} sx_{i}
 *
 *  |      I        0   0   |
 *  |     sz        0   0   |
 *  | -(g+r)*sx     sz  I   |
 *
 *        2
 *        |
 *    0---H---1
 *        |
 *        3
 *
 */

// Eigen::Tensor<cx64,2> operator*(const Eigen::Tensor<cx64,2> &, double rhs);
// Eigen::Tensor<cx64,2> operator*(double lhs, const Eigen::Tensor<cx64,2> & t) {
//     return t*t.constant(static_cast<cx64>(lhs));
// }

template<typename Scalar>
Eigen::Tensor<Scalar, 4> IsingRandomField<Scalar>::get_mpo(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody,
                                                           [[maybe_unused]] std::optional<std::vector<size_t>> skip) const {
    tools::log->debug("mpo({}): building tf-rf ising mpo", get_position());
    if(not all_mpo_parameters_have_been_set)
        throw except::runtime_error("mpo({}): can't build mpo: full lattice parameters haven't been set yet.", get_position());

    auto J1 = static_cast<RealScalar>(1.0);
    auto J2 = static_cast<RealScalar>(1.0);
    if(nbody.has_value()) {
        J1 = static_cast<RealScalar>(0.0);
        J2 = static_cast<RealScalar>(0.0);
        for(const auto &n : nbody.value()) {
            if(n == 1) J1 = static_cast<RealScalar>(1.0);
            if(n == 2) J2 = static_cast<RealScalar>(1.0);
        }
    }
    auto id = tenx::asScalarType<Scalar>(qm::spin::half::tensor::id);
    auto sx = tenx::asScalarType<Scalar>(qm::spin::half::tensor::sx);
    auto sz = tenx::asScalarType<Scalar>(qm::spin::half::tensor::sz);

    Eigen::Tensor<Scalar, 4> mpo_build;
    mpo_build.resize(3, 3, h5tb.param.spin_dim, h5tb.param.spin_dim);
    mpo_build.setZero();


    mpo_build.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = id;
    mpo_build.slice(std::array<long, 4>{1, 0, 0, 0}, extent4).reshape(extent2) = sz;
    mpo_build.slice(std::array<long, 4>{2, 0, 0, 0}, extent4).reshape(extent2) = -J1 * static_cast<RealScalar>(get_field()) * sx - energy_shift_per_site * id;
    mpo_build.slice(std::array<long, 4>{2, 1, 0, 0}, extent4).reshape(extent2) = -J2 * static_cast<RealScalar>(get_coupling()) * sz;
    mpo_build.slice(std::array<long, 4>{2, 2, 0, 0}, extent4).reshape(extent2) = id;
    if(tenx::hasNaN(mpo_internal)) {
        print_parameter_names();
        print_parameter_values();
        throw except::runtime_error("mpo({}): found nan", get_position());
    }
    return mpo_build;
}

template<typename Scalar>
void IsingRandomField<Scalar>::randomize_hamiltonian() {
    if(h5tb.param.distribution == "normal") {
        h5tb.param.h_rand = rnd::normal(h5tb.param.h_mean, h5tb.param.h_wdth);
    } else if(h5tb.param.distribution == "lognormal") {
        h5tb.param.h_rand = rnd::log_normal(h5tb.param.h_mean, h5tb.param.h_wdth);
    } else if(h5tb.param.distribution == "uniform") {
        h5tb.param.h_rand = rnd::uniform_double_box(h5tb.param.h_mean - h5tb.param.h_wdth / 2.0, h5tb.param.h_mean + h5tb.param.h_wdth / 2.0);
    } else {
        throw except::runtime_error("wrong distribution [{}]: expected one of normal | lognormal | uniform", h5tb.param.distribution);
    }
    all_mpo_parameters_have_been_set = false;
    mpo_squared                      = std::nullopt;
    unique_id                        = std::nullopt;
    unique_id_sq                     = std::nullopt;
}

template<typename Scalar>
std::unique_ptr<MpoSite<Scalar>> IsingRandomField<Scalar>::clone() const {
    return std::make_unique<IsingRandomField>(*this);
}

template<typename Scalar>
long IsingRandomField<Scalar>::get_spin_dimension() const {
    return h5tb.param.spin_dim;
}

template<typename Scalar>
void IsingRandomField<Scalar>::set_averages([[maybe_unused]] std::vector<TableMap> lattice_parameters, bool infinite) {
    if(not infinite) {
        lattice_parameters.back()["J1"]    = 0.0;
        lattice_parameters.back()["J2"]    = 0.0;
        lattice_parameters.end()[-2]["J2"] = 0.0;
    }
    // Recompute J_avrg and h_avrg from given J_rnrd and h_rnd on all sites
    double J_sum = 0;
    double h_sum = 0;
    for(auto &site_params : lattice_parameters) {
        auto J1_     = std::any_cast<double>(site_params["J1"]);
        auto J2_     = std::any_cast<double>(site_params["J2"]);
        auto h_tran_ = std::any_cast<double>(site_params["h_tran"]);
        auto h_rand_ = std::any_cast<double>(site_params["h_rand"]);
        J_sum += J1_ + J2_;
        h_sum += h_tran_ + h_rand_;
    }

    set_parameters(lattice_parameters[get_position()]);
}

template<typename Scalar>
void IsingRandomField<Scalar>::save_hamiltonian(h5pp::File &file, std::string_view table_path) const {
    if(not file.linkExists(table_path)) file.createTable(h5tb.get_h5_type(), table_path, "Transverse-field Ising");
    file.appendTableRecords(h5tb.param, table_path);
}

template<typename Scalar>
void IsingRandomField<Scalar>::load_hamiltonian(const h5pp::File &file, std::string_view model_prefix) {
    auto ham_table = fmt::format("{}/hamiltonian", model_prefix);
    if(file.linkExists(ham_table)) {
        h5tb.param                       = file.readTableRecords<h5tb_ising_tf_rf::table>(ham_table, position);
        all_mpo_parameters_have_been_set = true;
    } else {
        throw except::runtime_error("Could not load MPO. Table [{}] does not exist", ham_table);
    }

    // Check that we are on the same point of the phase diagram
    using namespace settings::model::ising_tf_rf;
    if(std::abs(h5tb.param.J1 - J1) > 1e-6) throw except::runtime_error("J1_rand {:.16f} != {:.16f} ising_tf_rf::J1_rand", h5tb.param.J1, J1);
    if(std::abs(h5tb.param.J2 - J2) > 1e-6) throw except::runtime_error("J2_rand {:.16f} != {:.16f} ising_tf_rf::J2_rand", h5tb.param.J2, J2);
    if(std::abs(h5tb.param.h_tran - h_tran) > 1e-6) throw except::runtime_error("h_tran {:.16f} != {:.16f} ising_tf_rf::h_tran", h5tb.param.h_tran, h_tran);
    if(std::abs(h5tb.param.h_mean - h_mean) > 1e-6) throw except::runtime_error("h_mean {:.16f} != {:.16f} ising_tf_rf::h_mean", h5tb.param.h_mean, h_mean);
    if(std::abs(h5tb.param.h_wdth - h_wdth) > 1e-6) throw except::runtime_error("h_wdth {:.16f} != {:.16f} ising_tf_rf::h_wdth", h5tb.param.h_wdth, h_wdth);

    build_mpo();
}
