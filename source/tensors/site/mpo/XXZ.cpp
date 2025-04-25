#include "XXZ.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <h5pp/h5pp.h>

template class XXZ<fp32>;
template class XXZ<fp64>;
template class XXZ<fp128>;
template class XXZ<cx32>;
template class XXZ<cx64>;
template class XXZ<cx128>;

template<typename Scalar>
XXZ<Scalar>::XXZ(ModelType model_type_, size_t position_) : MpoSite<Scalar>(model_type_, position_) {
    h5tb.param.delta        = settings::model::xxz::delta;
    h5tb.param.spin_dim     = settings::model::xxz::spin_dim;
    h5tb.param.distribution = settings::model::xxz::distribution;
    extent4                 = {1, 1, h5tb.param.spin_dim, h5tb.param.spin_dim};
    extent2                 = {h5tb.param.spin_dim, h5tb.param.spin_dim};
}

template<typename Scalar>
typename XXZ<Scalar>::RealScalar XXZ<Scalar>::get_field() const {
    return static_cast<RealScalar>(h5tb.param.h_rand);
}
template<typename Scalar>
typename XXZ<Scalar>::RealScalar XXZ<Scalar>::get_coupling() const {
    return static_cast<RealScalar>(h5tb.param.delta);
}
template<typename Scalar>
void XXZ<Scalar>::print_parameter_names() const {
    h5tb.print_parameter_names();
}
template<typename Scalar>
void XXZ<Scalar>::print_parameter_values() const {
    h5tb.print_parameter_values();
}

template<typename Scalar>
void XXZ<Scalar>::set_parameters(TableMap &parameters) {
    h5tb.param.delta                 = std::any_cast<decltype(h5tb.param.delta)>(parameters["delta"]);
    h5tb.param.h_rand                = std::any_cast<decltype(h5tb.param.h_rand)>(parameters["h_rand"]);
    h5tb.param.spin_dim              = std::any_cast<decltype(h5tb.param.spin_dim)>(parameters["spin_dim"]);
    h5tb.param.distribution          = std::any_cast<decltype(h5tb.param.distribution)>(parameters["distribution"]);
    all_mpo_parameters_have_been_set = true;
}

template<typename Scalar>
typename XXZ<Scalar>::TableMap XXZ<Scalar>::get_parameters() const {
    /* clang-format off */
    TableMap parameters;
    parameters["delta"]         = h5tb.param.delta;
    parameters["h_rand"]        = h5tb.param.h_rand;
    parameters["spin_dim"]      = h5tb.param.spin_dim;
    parameters["distribution"]  = h5tb.param.distribution;
    return parameters;
    /* clang-format on */
}

template<typename Scalar>
std::any XXZ<Scalar>::get_parameter(const std::string_view name) const {
    /* clang-format off */
    if(name      == "delta")        return  h5tb.param.delta;
    else if(name == "h_rand")       return  h5tb.param.h_rand;
    else if(name == "spin_dim")     return  h5tb.param.spin_dim;
    else if(name == "distribution") return  h5tb.param.distribution;
    /* clang-format on */
    throw except::logic_error("Invalid parameter name for XXZ model: {}", name);
}

template<typename Scalar>
void XXZ<Scalar>::set_parameter(const std::string_view name, std::any value) {
    /* clang-format off */
    if(name      == "delta")         h5tb.param.delta = std::any_cast<decltype(h5tb.param.delta)>(value);
    else if(name == "h_rand")       h5tb.param.h_rand = std::any_cast<decltype(h5tb.param.h_rand)>(value);
    else if(name == "spin_dim")     h5tb.param.spin_dim = std::any_cast<decltype(h5tb.param.spin_dim)>(value);
    else if(name == "distribution") h5tb.param.distribution = std::any_cast<decltype(h5tb.param.distribution)>(value);
    else
        /* clang-format on */
        throw except::logic_error("Invalid parameter name for the XXZ model: {}", name);
    build_mpo();
    build_mpo_squared();
}

/*! Builds the MPO hamiltonian as a rank 4 tensor.
 *
 * H = 1/4 [Σ(σx{i}*σx{i+1} + σy{i}*σy{i+1}) + Δσz{i}*σz{i+1} ] + Σ h{i}σz{i}
 *   = [Σ (1/2)*(σ+{i}*σ-{i+1} + σ-{i}*σ+{i+1}) + (Δ/4)σz{i}*σz{i+1}] + Σ h{i}σz{i}
 *
 * where J = 1 in this case.
 *
 *
 *
 *  |    I           0          0          0         0   |
 *  |    σx/4        0          0          0         0   |
 *  |    σy/4        0          0          0         0   |
 *  |    σz/4        0          0          0         0   |
 *  | h_rand*σz      σx        σy         Δσz        I   |
 *
 *  |    I           0          0          0         0   |
 *  |    σ+/2        0          0          0         0   |
 *  |    σ-/2        0          0          0         0   |
 *  |    σz/4        0          0          0         0   |
 *  | h_rand*σz      σ-        σ+         Δσz        I   |
 *
 *        2
 *        |
 *    0---H---1
 *        |
 *        3
 *
 *  Finite state machine for the MPO (adjacency diagram)
 *
 *         I==[0]--------------------------h_{i}σz_{i}------------[4]==I
 *         |                                                       |
 *         |---σx{i}/4---[1]-----------------σx{i+1}---------------|
 *         |                                                       |
 *         |---σy{i}/4---[2]-----------------σy{i+1}---------------|
 *         |                                                       |
 *         |---σz{i}/4---[3]---------------- Δσz{i+1}--------------|
 *
 * or
 *
 *         I==[0]--------------------------h_{i}σz_{i}------------[4]==I
 *         |                                                       |
 *         |---σ+{i}/2---[1]-----------------σ-{i+1}---------------|
 *         |                                                       |
 *         |---σ-{i}/2---[2]-----------------σ+{i+1}---------------|
 *         |                                                       |
 *         |---σz{i}/4---[3]---------------- Δσz{i+1}--------------|
 *
 *
 *
 */
template<typename Scalar>
Eigen::Tensor<Scalar, 4> XXZ<Scalar>::get_mpo(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody,
                                              [[maybe_unused]] std::optional<std::vector<size_t>> skip) const

{
    using namespace qm::spin::half::tensor;
    if constexpr(settings::debug) tools::log->trace("mpo({}): building XXZ mpo", get_position());
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
    auto sp = tenx::asScalarType<Scalar>(qm::spin::half::tensor::sp);
    auto sm = tenx::asScalarType<Scalar>(qm::spin::half::tensor::sm);
    auto sz = tenx::asScalarType<Scalar>(qm::spin::half::tensor::sz);

    Eigen::Tensor<Scalar, 4> mpo_build;
    mpo_build.resize(5, 5, h5tb.param.spin_dim, h5tb.param.spin_dim);
    mpo_build.setZero();
    mpo_build.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = id;
    mpo_build.slice(std::array<long, 4>{1, 0, 0, 0}, extent4).reshape(extent2) = sp * static_cast<Scalar>(0.50);
    mpo_build.slice(std::array<long, 4>{2, 0, 0, 0}, extent4).reshape(extent2) = sm * static_cast<Scalar>(0.50);
    mpo_build.slice(std::array<long, 4>{3, 0, 0, 0}, extent4).reshape(extent2) = sz * static_cast<Scalar>(0.25);
    mpo_build.slice(std::array<long, 4>{4, 0, 0, 0}, extent4).reshape(extent2) = J1 * get_field() * sz - energy_shift_per_site * id;
    mpo_build.slice(std::array<long, 4>{4, 1, 0, 0}, extent4).reshape(extent2) = J2 * sm;
    mpo_build.slice(std::array<long, 4>{4, 2, 0, 0}, extent4).reshape(extent2) = J2 * sp;
    mpo_build.slice(std::array<long, 4>{4, 3, 0, 0}, extent4).reshape(extent2) = J2 * get_coupling() * sz;
    mpo_build.slice(std::array<long, 4>{4, 4, 0, 0}, extent4).reshape(extent2) = id;

    if(tenx::hasNaN(mpo_build)) {
        print_parameter_names();
        print_parameter_values();
        throw except::runtime_error("mpo({}): found nan", get_position());
    }
    return mpo_build;
}

template<typename Scalar>
void XXZ<Scalar>::randomize_hamiltonian() {
    if(h5tb.param.distribution != "uniform") throw except::runtime_error("XXZ expects a uniform distribution. Got: {}", h5tb.param.distribution);
    h5tb.param.h_rand = rnd::uniform_double_box(settings::model::xxz::h_wdth);

    all_mpo_parameters_have_been_set = false;
    mpo_squared                      = std::nullopt;
    unique_id                        = std::nullopt;
    unique_id_sq                     = std::nullopt;
}

template<typename Scalar>
std::unique_ptr<MpoSite<Scalar>> XXZ<Scalar>::clone() const {
    return std::make_unique<XXZ>(*this);
}

template<typename Scalar>
long XXZ<Scalar>::get_spin_dimension() const {
    return h5tb.param.spin_dim;
}

template<typename Scalar>
void XXZ<Scalar>::set_averages(std::vector<TableMap> all_parameters, bool infinite) {
    if(not infinite) { all_parameters.back()["J_rand"] = 0.0; }
    set_parameters(all_parameters[get_position()]);
    double delta              = h5tb.param.delta;
    double h                  = settings::model::xxz::h_wdth;
    double L                  = safe_cast<double>(all_parameters.size());
    global_energy_upper_bound = (2 + delta) * (L - 1) + h * L;
}

template<typename Scalar>
void XXZ<Scalar>::save_hamiltonian(h5pp::File &file, std::string_view hamiltonian_table_path) const {
    if(not file.linkExists(hamiltonian_table_path)) file.createTable(h5tb.get_h5_type(), hamiltonian_table_path, "XXZ");
    file.appendTableRecords(h5tb.param, hamiltonian_table_path);
}

template<typename Scalar>
void XXZ<Scalar>::load_hamiltonian(const h5pp::File &file, std::string_view model_path) {
    auto ham_table = fmt::format("{}/hamiltonian", model_path);
    if(file.linkExists(ham_table)) {
        h5tb.param                       = file.readTableRecords<h5tb_xxz::table>(ham_table, position);
        all_mpo_parameters_have_been_set = true;
    } else
        throw except::runtime_error("could not load mpo: table [{}] does not exist", ham_table);

    using namespace settings::model::xxz;
    if(std::abs(h5tb.param.delta - delta) > 1e-6) throw except::runtime_error("delta  {:.16f} != {:.16f} XXZ<Scalar>::delta", h5tb.param.delta, delta);

    local_energy_upper_bound = 2 + h5tb.param.delta + h5tb.param.delta;

    // Calculate the global upper bound
    auto all_param            = file.readTableRecords<std::vector<h5tb_xxz::table>>(ham_table, h5pp::TableSelection::ALL);
    global_energy_upper_bound = 0;
    for(const auto &param : all_param) {
        global_energy_upper_bound += std::abs(param.h_rand);
        global_energy_upper_bound += 2 + std::abs(param.delta);
    }

    build_mpo();
}
