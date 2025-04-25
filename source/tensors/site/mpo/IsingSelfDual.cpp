#include "IsingSelfDual.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <h5pp/h5pp.h>

template class IsingSelfDual<fp32>;
template class IsingSelfDual<fp64>;
template class IsingSelfDual<fp128>;
template class IsingSelfDual<cx32>;
template class IsingSelfDual<cx64>;
template class IsingSelfDual<cx128>;

double delta_to_J_mean(double delta) { return std::min(1.0, std::exp(delta)); }
double delta_to_h_mean(double delta) { return std::min(1.0, std::exp(-delta)); }

template<typename Scalar>
IsingSelfDual<Scalar>::IsingSelfDual(ModelType model_type_, size_t position_) : MpoSite<Scalar>(model_type_, position_) {
    h5tb.param.lambda = settings::model::ising_sdual::lambda;
    h5tb.param.delta  = settings::model::ising_sdual::delta;
    h5tb.param.J_mean = delta_to_J_mean(h5tb.param.delta);
    h5tb.param.h_mean = delta_to_h_mean(h5tb.param.delta);
    // Sanity check on delta, J_mean, h_mean
    double delta_check = std::log(h5tb.param.J_mean) - std::log(h5tb.param.h_mean);
    if(std::abs(h5tb.param.delta - delta_check) > 1e-10)
        throw except::logic_error("error when transforming delta to (J_mean, h_mean): delta {:.12f} != {:.16f} delta_check", h5tb.param.delta, delta_check);

    h5tb.param.spin_dim     = settings::model::ising_sdual::spin_dim;
    h5tb.param.distribution = settings::model::ising_sdual::distribution;
    extent4                 = {1, 1, h5tb.param.spin_dim, h5tb.param.spin_dim};
    extent2                 = {h5tb.param.spin_dim, h5tb.param.spin_dim};
}

template<typename Scalar>
double IsingSelfDual<Scalar>::get_coupling() const {
    return h5tb.param.J_rand;
}
template<typename Scalar>
double IsingSelfDual<Scalar>::get_field() const {
    return h5tb.param.h_rand;
}
template<typename Scalar>
void IsingSelfDual<Scalar>::print_parameter_names() const {
    h5tb.print_parameter_names();
}
template<typename Scalar>
void IsingSelfDual<Scalar>::print_parameter_values() const {
    h5tb.print_parameter_values();
}

template<typename Scalar>
void IsingSelfDual<Scalar>::set_parameters(TableMap &parameters) {
    h5tb.param.J_mean                = std::any_cast<double>(parameters["J_mean"]);
    h5tb.param.J_wdth                = std::any_cast<double>(parameters["J_wdth"]);
    h5tb.param.J_rand                = std::any_cast<double>(parameters["J_rand"]);
    h5tb.param.h_mean                = std::any_cast<double>(parameters["h_mean"]);
    h5tb.param.h_wdth                = std::any_cast<double>(parameters["h_wdth"]);
    h5tb.param.h_rand                = std::any_cast<double>(parameters["h_rand"]);
    h5tb.param.lambda                = std::any_cast<double>(parameters["lambda"]);
    h5tb.param.delta                 = std::any_cast<double>(parameters["delta"]);
    h5tb.param.spin_dim              = std::any_cast<long>(parameters["spin_dim"]);
    h5tb.param.distribution          = std::any_cast<h5pp::vstr_t>(parameters["distribution"]);
    all_mpo_parameters_have_been_set = true;
}

template<typename Scalar>
IsingSelfDual<Scalar>::TableMap IsingSelfDual<Scalar>::get_parameters() const {
    TableMap parameters;
    parameters["J_mean"]       = h5tb.param.J_mean;
    parameters["J_wdth"]       = h5tb.param.J_wdth;
    parameters["J_rand"]       = h5tb.param.J_rand;
    parameters["h_mean"]       = h5tb.param.h_mean;
    parameters["h_wdth"]       = h5tb.param.h_wdth;
    parameters["h_rand"]       = h5tb.param.h_rand;
    parameters["lambda"]       = h5tb.param.lambda;
    parameters["delta"]        = h5tb.param.delta;
    parameters["spin_dim"]     = h5tb.param.spin_dim;
    parameters["distribution"] = h5tb.param.distribution;
    return parameters;
}

template<typename Scalar>
std::any IsingSelfDual<Scalar>::get_parameter(std::string_view name) const {
    /* clang-format off */
    if     (name == "J_mean")        return h5tb.param.J_mean;
    else if(name == "J_wdth")        return h5tb.param.J_wdth;
    else if(name == "J_rand")        return h5tb.param.J_rand;
    else if(name == "h_mean")        return h5tb.param.h_mean;
    else if(name == "h_wdth")        return h5tb.param.h_wdth;
    else if(name == "h_rand")        return h5tb.param.h_rand;
    else if(name == "lambda")        return h5tb.param.lambda;
    else if(name == "delta")         return h5tb.param.delta;
    else if(name == "spin_dim")      return h5tb.param.spin_dim;
    else if(name == "distribution")  return h5tb.param.distribution;
    /* clang-format on */
    throw except::logic_error("Invalid parameter name for IsingSelfDual model: {}", name);
}

template<typename Scalar>
void IsingSelfDual<Scalar>::set_parameter(const std::string_view name, std::any value) {
    /* clang-format off */
    if(name      == "J_mean")           h5tb.param.J_mean = std::any_cast<decltype(h5tb.param.J_mean)>(value);
    else if(name == "J_wdth")           h5tb.param.J_wdth = std::any_cast<decltype(h5tb.param.J_wdth)>(value);
    else if(name == "J_rand")           h5tb.param.J_rand = std::any_cast<decltype(h5tb.param.J_rand)>(value);
    else if(name == "h_mean")           h5tb.param.h_mean = std::any_cast<decltype(h5tb.param.h_mean)>(value);
    else if(name == "h_wdth")           h5tb.param.h_wdth = std::any_cast<decltype(h5tb.param.h_wdth)>(value);
    else if(name == "h_rand")           h5tb.param.h_rand = std::any_cast<decltype(h5tb.param.h_rand)>(value);
    else if(name == "lambda")           h5tb.param.lambda = std::any_cast<decltype(h5tb.param.lambda)>(value);
    else if(name == "delta")            h5tb.param.delta = std::any_cast<decltype(h5tb.param.delta)>(value);
    else if(name == "spin_dim")         h5tb.param.spin_dim = std::any_cast<decltype(h5tb.param.spin_dim)>(value);
    else if(name == "distribution")     h5tb.param.distribution = std::any_cast<decltype(h5tb.param.distribution)>(value);
    else
        /* clang-format on */
        throw except::logic_error("Invalid parameter name for the IsingSelfDual model: {}", name);
    build_mpo();
    build_mpo_squared();
}

/*! Builds the MPO hamiltonian as a rank 4 tensor. Notation following Schollwöck (2010)
 *
 * H = Σ J_{i} σx_{i} σx_{i+1} +  h_{i} σz_{i} + λ (h_mean σz_i σz_{i+1} + J_mean σx_{i} σx_{i+2})
 *
 *  |     I            0          0           0           0   |
 *  |     σx           0          0           0           0   |
 *  |     σz           0          0           0           0   |
 *  |     0            I          0           0           0   |
 *  |   h_rand*σz  J_rand*σx  l*h_mean*σz  l*J_mean*σx    I   |
 *
 *        2
 *        |
 *    0---H---1
 *        |
 *        3
 *
 */
template<typename Scalar>
Eigen::Tensor<Scalar, 4> IsingSelfDual<Scalar>::get_mpo(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody,
                                                        [[maybe_unused]] std::optional<std::vector<size_t>> skip) const {
    tools::log->debug("mpo({}): building ising-selfdual mpo", get_position());
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
    mpo_build.resize(5, 5, h5tb.param.spin_dim, h5tb.param.spin_dim);
    mpo_build.setZero();
    /* clang-format off */
    mpo_build.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = id;
    mpo_build.slice(std::array<long, 4>{1, 0, 0, 0}, extent4).reshape(extent2) = sx;
    mpo_build.slice(std::array<long, 4>{2, 0, 0, 0}, extent4).reshape(extent2) = sz;
    mpo_build.slice(std::array<long, 4>{3, 1, 0, 0}, extent4).reshape(extent2) = id;
    mpo_build.slice(std::array<long, 4>{4, 0, 0, 0}, extent4).reshape(extent2) = J1 * static_cast<RealScalar>(h5tb.param.h_rand) * sz - energy_shift_per_site * id;
    mpo_build.slice(std::array<long, 4>{4, 1, 0, 0}, extent4).reshape(extent2) = J2 * static_cast<RealScalar>(h5tb.param.J_rand) * sx;
    mpo_build.slice(std::array<long, 4>{4, 2, 0, 0}, extent4).reshape(extent2) = J2 * static_cast<RealScalar>(h5tb.param.h_mean * h5tb.param.lambda) * sz;
    mpo_build.slice(std::array<long, 4>{4, 3, 0, 0}, extent4).reshape(extent2) = J2 * static_cast<RealScalar>(h5tb.param.J_mean * h5tb.param.lambda) * sx;
    mpo_build.slice(std::array<long, 4>{4, 4, 0, 0}, extent4).reshape(extent2) = id;
    /* clang-format on */
    if(tenx::hasNaN(mpo_internal)) {
        print_parameter_names();
        print_parameter_values();
        throw except::runtime_error("mpo({}): found nan", get_position());
    }
    return mpo_build;
}

template<typename Scalar>
void IsingSelfDual<Scalar>::randomize_hamiltonian() {
    if(h5tb.param.distribution == "normal") {
        h5tb.param.J_wdth = 1.0;
        h5tb.param.h_wdth = 1.0;
        h5tb.param.J_rand = rnd::normal(h5tb.param.J_mean, h5tb.param.J_wdth);
        h5tb.param.h_rand = rnd::normal(h5tb.param.h_mean, h5tb.param.h_wdth);
    } else if(h5tb.param.distribution == "lognormal") {
        // See http://arxiv.org/abs/1711.00020
        h5tb.param.J_wdth = 1.0;
        h5tb.param.h_wdth = 1.0;
        h5tb.param.J_rand = rnd::log_normal(h5tb.param.J_mean, h5tb.param.J_wdth);
        h5tb.param.h_rand = rnd::log_normal(h5tb.param.h_mean, h5tb.param.h_wdth);
    } else if(h5tb.param.distribution == "uniform") {
        h5tb.param.J_wdth = 2.0 * h5tb.param.J_mean;
        h5tb.param.h_wdth = 2.0 * h5tb.param.h_mean;
        h5tb.param.J_rand = rnd::uniform_double_box(0, h5tb.param.J_wdth);
        h5tb.param.h_rand = rnd::uniform_double_box(0, h5tb.param.h_wdth);
    } else if(h5tb.param.distribution == "constant") {
        h5tb.param.J_wdth = 0.0;
        h5tb.param.h_wdth = 0.0;
        h5tb.param.J_rand = h5tb.param.J_mean;
        h5tb.param.h_rand = h5tb.param.h_mean;
    } else {
        throw except::runtime_error("wrong distribution [{}]: expected one of normal | lognormal | uniform | constant", h5tb.param.distribution);
    }

    all_mpo_parameters_have_been_set = false;
    mpo_squared                      = std::nullopt;
    unique_id                        = std::nullopt;
    unique_id_sq                     = std::nullopt;
}

template<typename Scalar>
std::unique_ptr<MpoSite<Scalar>> IsingSelfDual<Scalar>::clone() const {
    return std::make_unique<IsingSelfDual>(*this);
}

template<typename Scalar>
long IsingSelfDual<Scalar>::get_spin_dimension() const {
    return h5tb.param.spin_dim;
}

template<typename Scalar>
void IsingSelfDual<Scalar>::set_averages(std::vector<TableMap> all_parameters, bool infinite) {
    if(not infinite) { all_parameters.back()["J_rand"] = 0.0; }
    set_parameters(all_parameters[get_position()]);
}

template<typename Scalar>
void IsingSelfDual<Scalar>::save_hamiltonian(h5pp::File &file, std::string_view hamiltonian_table_path) const {
    if(not file.linkExists(hamiltonian_table_path)) file.createTable(h5tb.get_h5_type(), hamiltonian_table_path, "Selfdual Ising");
    file.appendTableRecords(h5tb.param, hamiltonian_table_path);
}

template<typename Scalar>
void IsingSelfDual<Scalar>::load_hamiltonian(const h5pp::File &file, std::string_view model_path) {
    auto ham_table = fmt::format("{}/hamiltonian", model_path);
    if(file.linkExists(ham_table)) {
        h5tb.param                       = file.readTableRecords<h5tb_ising_selfdual::table>(ham_table, position);
        all_mpo_parameters_have_been_set = true;
    } else
        throw except::runtime_error("could not load mpo: table [{}] does not exist", ham_table);

    // Check that we are on the same point of the phase diagram
    using namespace settings::model::ising_sdual;
    if(std::abs(h5tb.param.delta - delta) > 1e-6) throw except::runtime_error("delta  {:.16f} != {:.16f} ising_sdual::delta", h5tb.param.delta, delta);
    if(std::abs(h5tb.param.lambda - lambda) > 1e-6) throw except::runtime_error("lambda {:.16f} != {:.16f} ising_sdual::lambda", h5tb.param.lambda, lambda);
    if(h5tb.param.distribution != distribution)
        throw except::runtime_error("distribution {} != {} ising_sdual::distribution", h5tb.param.distribution, distribution);

    // Sanity check on delta, J_mean, h_mean
    double delta_check = std::log(h5tb.param.J_mean) - std::log(h5tb.param.h_mean);
    if(std::abs(h5tb.param.delta - delta_check) > 1e-10)
        throw except::logic_error("Error when transforming delta to (J_mean, h_mean): delta {:.12f} != {:.16f} delta_check", h5tb.param.delta, delta_check);

    build_mpo();
}
