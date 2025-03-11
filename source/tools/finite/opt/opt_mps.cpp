#include "../opt_mps.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/sfinae.h"
#include "io/fmt_custom.h"
#include "math/float.h"
#include "math/num.h"

using namespace tools::finite::opt;

opt_mps::opt_mps(std::string_view name_, const Eigen::Tensor<cx64, 3> &tensor_, const std::vector<size_t> &sites_, double eshift_, double energy_shifted_,
                 std::optional<double> variance_, double overlap_, size_t length_)
    : name(name_), tensor(tensor_), sites(sites_), eshift(eshift_), energy_shifted(energy_shifted_), energy(energy_shifted_ + eshift_), variance(variance_),
      overlap(overlap_), length(length_) {
    norm   = get_vector().norm();
    iter   = 0;
    num_mv = 0;
    time   = 0;
}

opt_mps::opt_mps(std::string_view name_, const Eigen::Tensor<cx64, 3> &tensor_, const std::vector<size_t> &sites_, double energy_, double variance_,
                 double overlap_, size_t length_, size_t iter_, size_t counter_, size_t time_)
    : name(name_), tensor(tensor_), sites(sites_), energy(energy_), variance(variance_), overlap(overlap_), length(length_), iter(iter_), num_mv(counter_),
      time(time_) {
    norm = get_vector().norm();
}

bool opt_mps::is_initialized() const { return tensor and name and sites and energy and variance and length; }

std::string_view opt_mps::get_name() const {
    if(name)
        return name.value();
    else
        throw except::runtime_error("opt_mps: name not set");
}

const Eigen::Tensor<cx64, 3> &opt_mps::get_tensor() const {
    if(tensor)
        return tensor.value();
    else
        throw except::runtime_error("opt_mps: tensor not set");
}

template<typename T>
Eigen::Tensor<T, 3> opt_mps::get_tensor_as() const {
    if constexpr(std::is_same_v<T, fp32> or std::is_same_v<T, fp64>) {
        return get_tensor().real().template cast<T>();
    } else if constexpr(std::is_same_v<T, cx32>) {
        return get_tensor().template cast<T>();
    } else if constexpr(std::is_same_v<T, cx64>) {
        return get_tensor();
    }
    throw except::runtime_error("get_tensor(): invalid type <{}>", sfinae::type_name<T>());
}

template Eigen::Tensor<fp32, 3> opt_mps::get_tensor_as() const;
template Eigen::Tensor<fp64, 3> opt_mps::get_tensor_as() const;
template Eigen::Tensor<cx32, 3> opt_mps::get_tensor_as() const;
template Eigen::Tensor<cx64, 3> opt_mps::get_tensor_as() const;

const Eigen::Tensor<cx64, 2> &opt_mps::get_bond() const {
    if(bond)
        return bond.value();
    else
        throw except::runtime_error("opt_mps: bond not set");
}
Eigen::Map<const Eigen::VectorXcd> opt_mps::get_vector() const { return Eigen::Map<const Eigen::VectorXcd>(get_tensor().data(), get_tensor().size()); }

Eigen::Map<const Eigen::VectorXd> opt_mps::get_vector_cplx_as_2xreal() const {
    if(not tensor) throw except::runtime_error("opt_mps: tensor not set");
    return {reinterpret_cast<const double *>(tensor.value().data()), 2 * tensor.value().size()};
}

Eigen::VectorXd opt_mps::get_vector_cplx_as_1xreal() const {
    if(not tensor) throw except::runtime_error("opt_mps: tensor not set");
    return Eigen::Map<const Eigen::VectorXcd>(tensor.value().data(), tensor.value().size()).real();
}

template<OptType optType>
[[nodiscard]] Eigen::VectorXd opt_mps::get_initial_state_with_lagrange_multiplier() const {
    if(not tensor) throw except::runtime_error("opt_mps: tensor not set");
    auto             size_old = tensor.value().size();
    auto             size_new = size_old + 2; // Add 2 for both lagrange multipliers (we have 2 constraints)
    Eigen::VectorXcd tensor_extended(size_new);
    tensor_extended.topRows(size_old) = Eigen::Map<const Eigen::VectorXcd>(tensor.value().data(), tensor.value().size());
    tensor_extended.bottomRows(2)[0]  = 0; // Norm
    tensor_extended.bottomRows(2)[1]  = 0; // Residual norm
    if constexpr(optType == OptType::REAL) { return Eigen::Map<const Eigen::VectorXcd>(tensor_extended.data(), tensor_extended.size()).real(); }
    if constexpr(optType == OptType::CPLX) {
        return Eigen::Map<Eigen::VectorXd>(reinterpret_cast<double *>(tensor_extended.data()), 2 * tensor_extended.size());
    }
}
template Eigen::VectorXd opt_mps::get_initial_state_with_lagrange_multiplier<OptType::REAL>() const;
template Eigen::VectorXd opt_mps::get_initial_state_with_lagrange_multiplier<OptType::CPLX>() const;

template<OptType optType>
[[nodiscard]] std::vector<double> opt_mps::get_stl_initial_state_with_lagrange_multiplier() const {
    if(not tensor) throw except::runtime_error("opt_mps: tensor not set");
    auto             size_old = tensor.value().size();
    auto             size_new = size_old + 2; // Add 2 for both lagrange multipliers (we have 2 constraints)
    Eigen::VectorXcd tensor_extended(size_new);
    tensor_extended.topRows(size_old) = Eigen::Map<const Eigen::VectorXcd>(tensor.value().data(), tensor.value().size());
    tensor_extended.bottomRows(2)[0]  = 0; // Norm
    tensor_extended.bottomRows(2)[1]  = 0; // Residual norm
    if constexpr(optType == OptType::REAL) {
        Eigen::VectorXd tensor_map = Eigen::Map<const Eigen::VectorXcd>(tensor_extended.data(), tensor_extended.size()).real();
        return {tensor_map.begin(), tensor_map.end()};
    }
    if constexpr(optType == OptType::CPLX) {
        auto tensor_map = Eigen::Map<Eigen::VectorXd>(reinterpret_cast<double *>(tensor_extended.data()), 2 * tensor_extended.size());
        return {tensor_map.begin(), tensor_map.end()};
    }
}
template std::vector<double> opt_mps::get_stl_initial_state_with_lagrange_multiplier<OptType::REAL>() const;
template std::vector<double> opt_mps::get_stl_initial_state_with_lagrange_multiplier<OptType::CPLX>() const;

const std::vector<size_t> &opt_mps::get_sites() const {
    if(not sites) throw except::runtime_error("opt_mps: sites not set");
    return sites.value();
}

double opt_mps::get_energy() const {
    if(energy)
        return energy.value();
    else
        throw except::runtime_error("opt_mps: energy not set");
}

double opt_mps::get_eshift() const {
    if(eshift)
        return eshift.value();
    else
        throw except::runtime_error("opt_mps: eshift not set");
}

double opt_mps::get_energy_per_site() const { return get_energy() / static_cast<double>(get_length()); }

double opt_mps::get_energy_shifted() const {
    if(energy_shifted)
        return energy_shifted.value();
    else
        throw except::runtime_error("opt_mps: energy_shifted not set");
}
double opt_mps::get_hsquared() const {
    if(hsquared)
        return hsquared.value();
    else
        throw except::runtime_error("opt_mps: hsquared not set");
}
double opt_mps::get_variance() const {
    if(variance)
        return variance.value();
    else
        throw except::runtime_error("opt_mps: variance not set");
}

size_t opt_mps::get_iter() const {
    if(iter)
        return iter.value();
    else
        return 0;
}

size_t opt_mps::get_op() const {
    if(num_op)
        return num_op.value();
    else
        return 0;
}

size_t opt_mps::get_mv() const {
    if(num_mv)
        return num_mv.value();
    else
        return 0;
}

size_t opt_mps::get_pc() const {
    if(num_pc)
        return num_pc.value();
    else
        return 0;
}

double opt_mps::get_time() const {
    if(time)
        return time.value();
    else
        return 0.0;
}

double opt_mps::get_time_mv() const {
    if(time_mv)
        return time_mv.value();
    else
        return 0.0;
}

double opt_mps::get_time_pc() const {
    if(time_pc)
        return time_pc.value();
    else
        return 0.0;
}

double opt_mps::get_delta_f() const {
    if(delta_f)
        return delta_f.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

double opt_mps::get_grad_max() const {
    if(grad_max)
        return grad_max.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

double opt_mps::get_grad_tol() const {
    if(grad_tol)
        return grad_tol.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

double opt_mps::get_relchange() const {
    if(relchange)
        return relchange.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

long opt_mps::get_eigs_idx() const {
    if(eigs_idx)
        return eigs_idx.value();
    else
        return -1;
}

long opt_mps::get_eigs_nev() const {
    if(eigs_nev)
        return eigs_nev.value();
    else
        return -1;
}

long opt_mps::get_eigs_ncv() const {
    if(eigs_ncv)
        return eigs_ncv.value();
    else
        return -1;
}

double opt_mps::get_eigs_tol() const {
    if(eigs_tol)
        return eigs_tol.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}
double opt_mps::get_eigs_rnorm() const {
    if(eigs_rnorm)
        return eigs_rnorm.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

double opt_mps::get_eigs_eigval() const {
    if(eigs_eigval)
        return eigs_eigval.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

std::string_view opt_mps::get_eigs_ritz() const {
    if(eigs_ritz)
        return eigs_ritz.value();
    else
        return "--";
}

cx64 opt_mps::get_eigs_shift() const {
    if(eigs_shift)
        return eigs_shift.value();
    else
        return cx64(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
}

double opt_mps::get_rnorm_H() const {
    if(rnorm_H)
        return rnorm_H.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

double opt_mps::get_rnorm_H2() const {
    if(rnorm_H2)
        return rnorm_H2.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

double opt_mps::get_overlap() const {
    if(overlap)
        return overlap.value();
    else
        throw except::runtime_error("opt_mps: overlap not set");
}

double opt_mps::get_alpha() const {
    if(alpha)
        return alpha.value();
    else
        return std::numeric_limits<double>::quiet_NaN();
}

double opt_mps::get_norm() const {
    if(norm)
        return norm.value();
    else
        throw except::runtime_error("opt_mps: norm not set");
}

size_t opt_mps::get_length() const {
    if(length)
        return length.value();
    else
        throw except::runtime_error("opt_mps: length not set");
}

OptSolver opt_mps::get_optsolver() const {
    if(optSolver)
        return optSolver.value();
    else
        throw except::runtime_error("opt_mps: optSolver not set");
}

OptAlgo opt_mps::get_optalgo() const {
    if(optAlgo)
        return optAlgo.value();
    else
        throw except::runtime_error("opt_mps: optAlgo not set");
}

OptExit opt_mps::get_optexit() const {
    if(optExit)
        return optExit.value();
    else
        throw except::runtime_error("opt_mps: optMode not set");
}

long opt_mps::get_bond_lim() const {
    if(bond_lim)
        return bond_lim.value();
    else
        throw except::runtime_error("opt_mps: bond_lim not set");
}

double opt_mps::get_trnc_lim() const {
    if(trnc_lim)
        return trnc_lim.value();
    else
        throw except::runtime_error("opt_mps: trnc_lim not set");
}

void opt_mps::clear() { tensor = std::nullopt; }
void opt_mps::normalize() {
    if(not tensor) throw except::runtime_error("opt_mps: tensor not set");
    Eigen::Map<Eigen::VectorXcd> vector(tensor.value().data(), tensor.value().size());
    vector.normalize();
    norm = vector.norm();
}

void opt_mps::set_name(std::string_view name_) { name = name_; }
void opt_mps::set_tensor(const Eigen::Tensor<cx64, 3> &tensor_) {
    tensor = tensor_;
    norm   = get_vector().norm();
}
void opt_mps::set_tensor(const Eigen::VectorXcd &vector, const Eigen::DSizes<long, 3> &dims) {
    tensor = Eigen::TensorMap<const Eigen::Tensor<const cx64, 3>>(vector.data(), dims);
    norm   = get_vector().norm();
}

void opt_mps::set_bond(const Eigen::MatrixXcd &matrix) {
    bond = Eigen::TensorMap<const Eigen::Tensor<const cx64, 2>>(matrix.data(), matrix.rows(), matrix.cols());
}
void opt_mps::set_bond(const Eigen::Tensor<cx64, 2> &bond_) { bond = bond_; }

void opt_mps::set_sites(const std::vector<size_t> &sites_) { sites = sites_; }
void opt_mps::set_energy_shifted(double energy_shifted_) {
    energy_shifted = energy_shifted_;
    if(energy and not eshift) eshift = energy.value() - energy_shifted.value();
    if(eshift and not energy) energy = energy_shifted.value() + eshift.value();
}
void opt_mps::set_eshift(double energy_shift_) {
    eshift = energy_shift_;
    if(energy and not energy_shifted) energy_shifted = energy.value() - eshift.value();
    if(energy_shifted and not energy) energy = energy_shifted.value() + eshift.value();
}

void opt_mps::set_energy(double energy_) {
    energy = energy_;
    if(eshift and not energy_shifted) energy_shifted = energy.value() - eshift.value();
    if(energy_shifted and not eshift) eshift = energy.value() - energy_shifted.value();
}
void opt_mps::set_energy_per_site(double energy_per_site_) { set_energy(energy_per_site_ * static_cast<double>(get_length())); }
void opt_mps::set_hsquared(double hsquared_) { hsquared = hsquared_; }
void opt_mps::set_variance(double variance_) { variance = variance_; }
void opt_mps::set_rnorm_H1(const double rnorm_) { rnorm_H = rnorm_; }
void opt_mps::set_rnorm_H2(const double rnorm_) { rnorm_H2 = rnorm_; }
void opt_mps::set_overlap(double overlap_) { overlap = overlap_; }
void opt_mps::set_alpha(std::optional<double> alpha_) { alpha = alpha_; }
void opt_mps::set_length(size_t length_) { length = length_; }
void opt_mps::set_iter(size_t iter_) { iter = iter_; }
void opt_mps::set_op(size_t op_) { num_op = op_; }
void opt_mps::set_mv(size_t mv_) { num_mv = mv_; }
void opt_mps::set_pc(size_t pc_) { num_pc = pc_; }
void opt_mps::set_time(double time_) { time = time_; }
void opt_mps::set_time_mv(double time_mv_) { time_mv = time_mv_; }
void opt_mps::set_time_pc(double time_pc_) { time_pc = time_pc_; }
void opt_mps::set_delta_f(double delta_f_) { delta_f = delta_f_; }
void opt_mps::set_grad_tol(double grad_tol_) { grad_tol = grad_tol_; }
void opt_mps::set_grad_max(double grad_max_) { grad_max = grad_max_; }
void opt_mps::set_relchange(double relative_change_) { relchange = relative_change_; }
void opt_mps::set_eigs_idx(long idx_) { eigs_idx = idx_; }
void opt_mps::set_eigs_nev(long nev_) { eigs_nev = nev_; }
void opt_mps::set_eigs_ncv(long ncv_) { eigs_ncv = ncv_; }
void opt_mps::set_eigs_tol(double tol_) { eigs_tol = tol_; }
void opt_mps::set_eigs_rnorm(double rnorm_) { eigs_rnorm = rnorm_; }
void opt_mps::set_eigs_eigval(double eigval_) { eigs_eigval = eigval_; }
void opt_mps::set_eigs_ritz(std::string_view ritz_) { eigs_ritz = std::string(ritz_); }
void opt_mps::set_eigs_shift(const cx64 shift_) { eigs_shift = shift_; }

void opt_mps::set_tensor_cplx(const double *data, const Eigen::DSizes<long, 3> &dims) {
    // Here we set a complex tensor from double data.
    // In the real representation, a pair of doubles corresponds to a single complex<double>
    // We expect the size of "dims" to corresponds to the number of complex numbers in the data buffer
    tensor = Eigen::TensorMap<const Eigen::Tensor<const cx64, 3>>(reinterpret_cast<const cx64 *>(data), dims);
    norm   = get_vector().norm();
}
void opt_mps::set_tensor_real(const double *data, const Eigen::DSizes<long, 3> &dims) {
    // Here we set a complex tensor from double data, but this time
    // only the real part is present in the data buffer, so 1 double corresponds to 1 std::complex<double>.
    // Simplest solution is to cast each element to complex before copy.
    tensor = Eigen::TensorMap<const Eigen::Tensor<const fp64, 3>>(data, dims).cast<cx64>();
    norm   = get_vector().norm();
}

void opt_mps::set_optsolver(OptSolver optSolver_) { optSolver = optSolver_; }
void opt_mps::set_optalgo(OptAlgo optAlgo_) { optAlgo = optAlgo_; }
void opt_mps::set_optexit(OptExit optExit_) { optExit = optExit_; }
void opt_mps::set_bond_limit(long bond_) { bond_lim = bond_; }
void opt_mps::set_trnc_limit(double trnc_) { trnc_lim = trnc_; }

void opt_mps::validate_basis_vector() const {
    std::string error_msg;
    error_msg.reserve(128);
    /* clang-format off */
    if(not name)            error_msg += "\t name           \n";
    if(not tensor)          error_msg += "\t tensor         \n";
    if(not sites)           error_msg += "\t sites          \n";
    if(not energy_shifted)  error_msg += "\t energy_shifted  \n";
    if(not eshift)          error_msg += "\t eshift         \n";
    if(not energy)          error_msg += "\t energy         \n";
    if(not overlap)         error_msg += "\t overlap        \n";
    if(not norm)            error_msg += "\t norm           \n";
    if(not length)          error_msg += "\t length         \n";
    if(not iter)            error_msg += "\t iter           \n";
    if(not num_mv)          error_msg += "\t num_mv         \n";
    if(not time)            error_msg += "\t time           \n";
    if constexpr (settings::debug){
        if(has_nan()) throw except::runtime_error("opt_mps error: mps has nan's");
    }
    /* clang-format on */
    if(not error_msg.empty()) { throw except::runtime_error("opt_mps error: Missing fields:\n{}", error_msg); }
}

void opt_mps::validate_initial_mps() const {
    std::string error_msg;
    error_msg.reserve(128);
    /* clang-format off */
    if(not name)            error_msg += "\t name           \n";
    if(not tensor)          error_msg += "\t tensor         \n";
    if(not sites)           error_msg += "\t sites          \n";
    if(not energy)          error_msg += "\t energy         \n";
    if(not eshift)          error_msg += "\t eshift         \n";
    if(not energy_shifted)  error_msg += "\t energy_shifted  \n";
    if(not overlap)         error_msg += "\t overlap        \n";
    if(not norm)            error_msg += "\t norm           \n";
    if(not length)          error_msg += "\t length         \n";
    if constexpr (settings::debug){
        if(has_nan()) throw except::runtime_error("opt_mps error: initial mps has nan's");
    }
    /* clang-format on */
    if(not error_msg.empty()) { throw except::runtime_error("opt_mps error: Missing fields in initial mps:\n{}", error_msg); }
}

void opt_mps::validate_result() const {
    std::string error_msg;
    error_msg.reserve(128);
    /* clang-format off */
    if(not name)            error_msg += "\t name           \n";
    if(not tensor)          error_msg += "\t tensor         \n";
    if(not sites)           error_msg += "\t sites          \n";
    if(not energy)          error_msg += "\t energy         \n";
    if(not eshift)          error_msg += "\t eshift         \n";
    if(not energy_shifted)  error_msg += "\t energy_shifted \n";
    if(not variance)        error_msg += "\t variance       \n";
    if(not overlap)         error_msg += "\t overlap        \n";
    if(not norm)            error_msg += "\t norm           \n";
    if(not length)          error_msg += "\t length         \n";
    if(not iter)            error_msg += "\t iter           \n";
    if(not num_mv)          error_msg += "\t num_mv         \n";
    if(not time)            error_msg += "\t time           \n";
    if constexpr (settings::debug){
        if(has_nan()) throw except::runtime_error("opt_mps error: mps has nan's");
    }
    /* clang-format on */
    if(not error_msg.empty()) { throw except::runtime_error("opt_mps error: Missing fields:\n{}", error_msg); }
}

bool opt_mps::operator<(const opt_mps &rhs) const {
    if(overlap and rhs.overlap) {
        if(this->get_overlap() < rhs.get_overlap())
            return true;
        else if(this->get_overlap() > rhs.get_overlap())
            return false;
    }
    // If we reached this point then the overlaps are equal (probably 0)
    // To disambiguate we can use the variance
    if(this->variance and rhs.variance) return std::abs(this->get_variance()) < std::abs(rhs.get_variance());

    // If we reached this point then the overlaps are equal (probably 0) and there are no variances defined
    // To disambiguate we can use the eigenvalue, which should
    // be spread around 0 as well (because we use shifted energies)
    // Eigenvalues nearest 0 are near the target energy, and while
    // not strictly relevant, it's probably the best we can do here.
    // Of course this is only valid if the eigenvalues are defined
    if(this->energy_shifted and rhs.energy_shifted) return std::abs(this->get_energy_shifted()) < std::abs(rhs.get_energy_shifted());

    // If we have reched this point the overlaps, variances and eigenvalues are not defined.
    // There is probably a logical bug somewhere
    fmt::print("Checking that this opt_mps is valid\n");
    this->validate_basis_vector();
    fmt::print("Checking that rhs opt_mps is valid\n");
    rhs.validate_basis_vector();
    return true;
}

bool opt_mps::operator>(const opt_mps &rhs) const { return not(*this < rhs); }

bool opt_mps::has_nan() const { return get_vector().hasNaN(); }