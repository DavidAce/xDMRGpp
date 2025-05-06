#pragma once

#include "../../opt_mps.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/sfinae.h"
#include "io/fmt_custom.h"
#include "math/float.h"
#include "math/num.h"

using namespace tools::finite::opt;



template<typename Scalar>
bool opt_mps<Scalar>::is_initialized() const {
    return tensor and name and sites and energy and variance and length;
}

template<typename Scalar> void opt_mps<Scalar>::normalize() {
    if(not tensor) throw except::runtime_error("opt_mps: tensor not set");
    Eigen::Map<VectorType<Scalar>> vector(tensor.value().data(), tensor.value().size());
    vector.normalize();
    norm = vector.norm();
}
template<typename Scalar> void opt_mps<Scalar>::validate_basis_vector() const {
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

template<typename Scalar> void opt_mps<Scalar>::validate_initial_mps() const {
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

template<typename Scalar> void opt_mps<Scalar>::validate_result() const {
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

template<typename Scalar> bool opt_mps<Scalar>::operator<(const opt_mps &rhs) const {
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

template<typename Scalar> bool opt_mps<Scalar>::operator>(const opt_mps &rhs) const { return not(*this < rhs); }
template<typename Scalar> bool opt_mps<Scalar>::has_nan() const { return get_vector().hasNaN(); }