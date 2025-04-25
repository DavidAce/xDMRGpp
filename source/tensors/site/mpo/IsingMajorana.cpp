#include "IsingMajorana.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <h5pp/h5pp.h>
#include <tools/infinite/mps.h>

template class IsingMajorana<fp32>;
template class IsingMajorana<fp64>;
template class IsingMajorana<fp128>;
template class IsingMajorana<cx32>;
template class IsingMajorana<cx64>;
template class IsingMajorana<cx128>;

// double delta_to_J_mean(double delta) { return ;  }
// double delta_to_h_mean(double delta) { return ;  }
double delta_to_W_J(double delta) { return std::exp(delta / 2.0); }
double delta_to_W_h(double delta) { return std::exp(-delta / 2.0); }

template<typename Scalar>
IsingMajorana<Scalar>::IsingMajorana(ModelType model_type_, size_t position_) : MpoSite<Scalar>(model_type_, position_) {
    h5tb.param.g            = settings::model::ising_majorana::g;
    h5tb.param.delta        = settings::model::ising_majorana::delta;
    h5tb.param.spin_dim     = settings::model::ising_majorana::spin_dim;
    h5tb.param.distribution = settings::model::ising_majorana::distribution;
    extent4                 = {1, 1, h5tb.param.spin_dim, h5tb.param.spin_dim};
    extent2                 = {h5tb.param.spin_dim, h5tb.param.spin_dim};
}

template<typename Scalar>
typename IsingMajorana<Scalar>::RealScalar IsingMajorana<Scalar>::get_coupling() const {
    return static_cast<RealScalar>(h5tb.param.J_rand);
}
template<typename Scalar>
typename IsingMajorana<Scalar>::RealScalar IsingMajorana<Scalar>::get_field() const {
    return static_cast<RealScalar>(h5tb.param.h_rand);
}
template<typename Scalar>
void IsingMajorana<Scalar>::print_parameter_names() const {
    h5tb.print_parameter_names();
}
template<typename Scalar>
void IsingMajorana<Scalar>::print_parameter_values() const {
    h5tb.print_parameter_values();
}

template<typename Scalar>
void IsingMajorana<Scalar>::set_parameters(TableMap &parameters) {
    h5tb.param.g                     = std::any_cast<decltype(h5tb.param.g)>(parameters["g"]);
    h5tb.param.delta                 = std::any_cast<decltype(h5tb.param.delta)>(parameters["delta"]);
    h5tb.param.J_rand                = std::any_cast<decltype(h5tb.param.J_rand)>(parameters["J_rand"]);
    h5tb.param.h_rand                = std::any_cast<decltype(h5tb.param.h_rand)>(parameters["h_rand"]);
    h5tb.param.spin_dim              = std::any_cast<decltype(h5tb.param.spin_dim)>(parameters["spin_dim"]);
    h5tb.param.distribution          = std::any_cast<decltype(h5tb.param.distribution)>(parameters["distribution"]);
    all_mpo_parameters_have_been_set = true;
}

template<typename Scalar>
typename IsingMajorana<Scalar>::TableMap IsingMajorana<Scalar>::get_parameters() const {
    /* clang-format off */
    TableMap parameters;
    parameters["g"]                             = h5tb.param.g;
    parameters["delta"]                         = h5tb.param.delta;
    parameters["J_rand"]                        = h5tb.param.J_rand;
    parameters["h_rand"]                        = h5tb.param.h_rand;
    parameters["spin_dim"]                      = h5tb.param.spin_dim;
    parameters["distribution"]                  = h5tb.param.distribution;
    parameters["local_energy_upper_bound"] = local_energy_upper_bound;
    return parameters;
    /* clang-format on */
}

template<typename Scalar>
std::any IsingMajorana<Scalar>::get_parameter(const std::string_view name) const {
    /* clang-format off */
    if(name == "g")                 return  h5tb.param.g;
    else if(name == "delta")        return  h5tb.param.delta;
    else if(name == "J_rand")       return  h5tb.param.J_rand;
    else if(name == "h_rand")       return  h5tb.param.h_rand;
    else if(name == "spin_dim")     return  h5tb.param.spin_dim;
    else if(name == "distribution") return  h5tb.param.distribution;
    /* clang-format on */
    throw except::logic_error("Invalid parameter name for the IsingMajorana model: {}", name);
}

template<typename Scalar>
void IsingMajorana<Scalar>::set_parameter(const std::string_view name, std::any value) {
    /* clang-format off */
    if(name == "g")                 h5tb.param.g = std::any_cast<decltype(h5tb.param.g)>(value);
    else if(name == "delta")        h5tb.param.delta = std::any_cast<decltype(h5tb.param.delta)>(value);
    else if(name == "J_rand")       h5tb.param.J_rand = std::any_cast<decltype(h5tb.param.J_rand)>(value);
    else if(name == "h_rand")       h5tb.param.h_rand = std::any_cast<decltype(h5tb.param.h_rand)>(value);
    else if(name == "spin_dim")     h5tb.param.spin_dim = std::any_cast<decltype(h5tb.param.spin_dim)>(value);
    else if(name == "distribution") h5tb.param.distribution = std::any_cast<decltype(h5tb.param.distribution)>(value);
    else
        /* clang-format on */
        throw except::logic_error("Invalid parameter name for the IsingMajorana model: {}", name);
    build_mpo();
    build_mpo_squared();
}

/*! Builds the MPO hamiltonian as a rank 4 tensor.
 *
 * H = Σ J_{i} σx_{i} σx_{i+1} + h_{i} σz_{i} + g*(σz_i σz_{i+1} + σx_{i} σx_{i+2})
 *
 *  |    I           0          0          0         0   |
 *  |    σx          0          0          0         0   |
 *  |    σz          0          0          0         0   |
 *  |    0           I          0          0         0   |
 *  | h_rand*σz  J_rand*σx     g*σz      g*σx        I   |
 *
 *        2
 *        |
 *    0---H---1
 *        |
 *        3
 *
 *  We use the definitions from https://link.aps.org/doi/10.1103/PhysRevResearch.4.L032016
 *  We take g > 0 and use box distributions for h_i and J_i in Box(0,W_h|J), where W_h|J are the
 *  disorder strengths of h_i and J_i.
 *  The disorder strengths are set to W_J = 1/W_h = W, giving us a single control parameter
 *  delta = <ln J_i> - <ln h_i> =  2ln(W).
 *
 */
template<typename Scalar>
Eigen::Tensor<Scalar, 4> IsingMajorana<Scalar>::get_mpo(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody,
                                                        [[maybe_unused]] std::optional<std::vector<size_t>> skip) const {
    if constexpr(settings::debug) tools::log->trace("mpo({}): building ising-majorana mpo", get_position());
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
    mpo_build.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = id;
    mpo_build.slice(std::array<long, 4>{1, 0, 0, 0}, extent4).reshape(extent2) = sx;
    mpo_build.slice(std::array<long, 4>{2, 0, 0, 0}, extent4).reshape(extent2) = sz;
    mpo_build.slice(std::array<long, 4>{3, 1, 0, 0}, extent4).reshape(extent2) = id;
    mpo_build.slice(std::array<long, 4>{4, 0, 0, 0}, extent4).reshape(extent2) = J1 * get_field() * sz - energy_shift_per_site * id;
    mpo_build.slice(std::array<long, 4>{4, 1, 0, 0}, extent4).reshape(extent2) = J2 * get_coupling() * sx;
    mpo_build.slice(std::array<long, 4>{4, 2, 0, 0}, extent4).reshape(extent2) = (J2 * static_cast<RealScalar>(h5tb.param.g)) * sz;
    mpo_build.slice(std::array<long, 4>{4, 3, 0, 0}, extent4).reshape(extent2) = (J2 * static_cast<RealScalar>(h5tb.param.g)) * sx;
    mpo_build.slice(std::array<long, 4>{4, 4, 0, 0}, extent4).reshape(extent2) = id;

    if(tenx::hasNaN(mpo_build)) {
        print_parameter_names();
        print_parameter_values();
        throw except::runtime_error("mpo({}): found nan", get_position());
    }
    return mpo_build;
}

template<typename Scalar>
void IsingMajorana<Scalar>::randomize_hamiltonian() {
    if(h5tb.param.distribution != "uniform") throw except::runtime_error("IsingMajorana expects a uniform distribution. Got: {}", h5tb.param.distribution);
    h5tb.param.J_rand = rnd::uniform_double_box(0, delta_to_W_J(h5tb.param.delta));
    h5tb.param.h_rand = rnd::uniform_double_box(0, delta_to_W_h(h5tb.param.delta));

    // Estimate the maximum energy contribution from this site
    local_energy_upper_bound = std::abs(h5tb.param.h_rand) + 2 * std::abs(h5tb.param.g);
    if(get_position() + 1 < settings::model::model_size) local_energy_upper_bound += std::abs(h5tb.param.J_rand) + std::abs(h5tb.param.g);
    if(get_position() + 2 < settings::model::model_size) local_energy_upper_bound += std::abs(h5tb.param.g);

    all_mpo_parameters_have_been_set = false;
    mpo_squared                      = std::nullopt;
    unique_id                        = std::nullopt;
    unique_id_sq                     = std::nullopt;
}

template<typename Scalar>
std::unique_ptr<MpoSite<Scalar>> IsingMajorana<Scalar>::clone() const {
    return std::make_unique<IsingMajorana<Scalar>>(*this);
}

template<typename Scalar>
long IsingMajorana<Scalar>::get_spin_dimension() const {
    return h5tb.param.spin_dim;
}

template<typename Scalar>
void IsingMajorana<Scalar>::set_averages(std::vector<TableMap> all_parameters, bool infinite) {
    if(not infinite) { all_parameters.back()["J_rand"] = 0.0; }
    set_parameters(all_parameters[get_position()]);
    global_energy_upper_bound = 0.0;
    for(const auto &param : all_parameters) {
        if(param.contains("local_energy_upper_bound")) {
            global_energy_upper_bound += std::any_cast<decltype(global_energy_upper_bound)>(param.at("local_energy_upper_bound"));
        }
    }
}

template<typename Scalar>
void IsingMajorana<Scalar>::save_hamiltonian(h5pp::File &file, std::string_view hamiltonian_table_path) const {
    if(not file.linkExists(hamiltonian_table_path)) file.createTable(h5tb.get_h5_type(), hamiltonian_table_path, "Ising-Majorana");
    file.appendTableRecords(h5tb.param, hamiltonian_table_path);
}

template<typename Scalar>
void IsingMajorana<Scalar>::load_hamiltonian(const h5pp::File &file, std::string_view model_path) {
    auto ham_table = fmt::format("{}/hamiltonian", model_path);
    if(file.linkExists(ham_table)) {
        h5tb.param                       = file.readTableRecords<h5tb_ising_majorana::table>(ham_table, position);
        all_mpo_parameters_have_been_set = true;
    } else
        throw except::runtime_error("could not load mpo: table [{}] does not exist", ham_table);

    using namespace settings::model::ising_majorana;
    if(std::abs(h5tb.param.g - g) > 1e-6) throw except::runtime_error("g {:.16f} != {:.16f} ising_majorana::g", h5tb.param.g, g);
    if(std::abs(h5tb.param.delta - delta) > 1e-6) throw except::runtime_error("delta  {:.16f} != {:.16f} ising_majorana::delta", h5tb.param.delta, delta);
    // Estimate the maximum energy contribution from this site
    local_energy_upper_bound = std::abs(h5tb.param.h_rand) + 2 * std::abs(h5tb.param.g);
    if(get_position() + 1 < settings::model::model_size) local_energy_upper_bound += std::abs(h5tb.param.J_rand) + std::abs(h5tb.param.g);
    if(get_position() + 2 < settings::model::model_size) local_energy_upper_bound += std::abs(h5tb.param.g);

    // Calculate the global upper bound
    auto all_param            = file.readTableRecords<std::vector<h5tb_ising_majorana::table>>(ham_table, h5pp::TableSelection::ALL);
    global_energy_upper_bound = 0;
    for(const auto &param : all_param) {
        global_energy_upper_bound += std::abs(param.h_rand) + 2 * std::abs(param.g);
        global_energy_upper_bound += std::abs(h5tb.param.J_rand) + std::abs(h5tb.param.g);
        global_energy_upper_bound += std::abs(h5tb.param.g);
    }
    build_mpo();
}
