#pragma once
#include "../MpoSite.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "io/fmt_f128_t.h"
#include "math/cast.h"
#include "math/float.h"
#include "math/hash.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "math/tenx/sfinae.h"
#include "MpoSite.tmpl.h"
#include "qm/qm.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <config/settings.h>
#include <general/sfinae.h>
#include <h5pp/h5pp.h>
#include <utility>

template<typename Scalar>
MpoSite<Scalar>::MpoSite(ModelType model_type_, size_t position_) : model_type(model_type_), position(position_) {}

template<typename Scalar>
void MpoSite<Scalar>::build_mpo() {
    mpo_internal   = get_mpo(energy_shift_mpo);
    mpo_internal   = get_parity_shifted_mpo(mpo_internal);
    mpo_internal   = apply_edge_left(mpo_internal, get_MPO_edge_left(mpo_internal));
    mpo_internal   = apply_edge_right(mpo_internal, get_MPO_edge_right(mpo_internal));
    unique_id      = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
void MpoSite<Scalar>::build_mpo_q() {
    mpo_internal_q = get_mpo_q(energy_shift_mpo);
    mpo_internal_q = get_parity_shifted_mpo(mpo_internal_q);
    mpo_internal_q = apply_edge_left(mpo_internal_q, get_MPO_edge_left(mpo_internal_q));
    mpo_internal_q = apply_edge_right(mpo_internal_q, get_MPO_edge_right(mpo_internal_q));
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
void MpoSite<Scalar>::build_mpo_squared() {
    mpo_squared    = get_non_compressed_mpo_squared();
    mpo_squared    = get_parity_shifted_mpo_squared(mpo_squared.value());
    mpo_squared    = apply_edge_left(mpo_squared.value(), get_MPO2_edge_left<Scalar>());
    mpo_squared    = apply_edge_right(mpo_squared.value(), get_MPO2_edge_right<Scalar>());
    unique_id_sq   = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> MpoSite<Scalar>::get_non_compressed_mpo_squared() const {
    if constexpr(settings::debug) tools::log->trace("mpo({}): building mpo²", get_position());
    Eigen::Tensor<Scalar, 4> mpo = get_mpo(energy_shift_mpo);
    Eigen::Tensor<Scalar, 4> mpo2;
    {
        auto d0 = mpo.dimension(0) * mpo.dimension(0);
        auto d1 = mpo.dimension(1) * mpo.dimension(1);
        auto d2 = mpo.dimension(2);
        auto d3 = mpo.dimension(3);
        mpo2    = mpo.contract(mpo.conjugate(), tenx::idx({3}, {2})).shuffle(tenx::array6{0, 3, 1, 4, 2, 5}).reshape(tenx::array4{d0, d1, d2, d3});
    }
    return mpo2;
}

template<typename Scalar>
void MpoSite<Scalar>::set_mpo(const Eigen::Tensor<Scalar, 4> &mpo) {
    mpo_internal   = mpo;
    unique_id      = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
void MpoSite<Scalar>::set_mpo_squared(const Eigen::Tensor<Scalar, 4> &mpo_sq) {
    mpo_squared    = mpo_sq;
    unique_id_sq   = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
void MpoSite<Scalar>::clear_mpo() {
    mpo_internal   = Eigen::Tensor<Scalar, 4>();
    mpo_internal_q = Eigen::Tensor<QuadScalar, 4>();
    unique_id      = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
void MpoSite<Scalar>::clear_mpo_squared() {
    mpo_squared    = std::nullopt;
    unique_id_sq   = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
bool MpoSite<Scalar>::has_mpo() const {
    bool has_mpo_internal = mpo_internal.size() != 0;
    // bool has_mpo_internal_t = mpo_internal_t.size() != 0;
    return has_mpo_internal;
}
template<typename Scalar>
bool MpoSite<Scalar>::has_mpo_squared() const {
    return mpo_squared.has_value();
}

template<typename Scalar>
Eigen::Tensor<typename MpoSite<Scalar>::QuadScalar, 4> MpoSite<Scalar>::get_mpo_q(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody,
                                                                                  std::optional<std::vector<size_t>> skip) const {
    // tools::log->trace("MpoSite<Scalar>::get_mpo_q(): Pointless upcast {} -> {}", sfinae::type_name<cx64>(), sfinae::type_name<cx128>());
    auto ereal = energy_shift_per_site; // cx64(static_cast<fp64>(energy_shift_per_site.real(), static_cast<fp64>(energy_shift_per_site.imag())));
    auto mpo   = get_mpo(ereal, nbody, skip);
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        return mpo.unaryExpr([](auto z) { return std::complex(static_cast<fp128>(std::real(z)), static_cast<fp128>(std::imag(z))); });

    } else {
        return mpo.unaryExpr([](auto z) { return static_cast<fp128>(z); });
    }
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 4> &MpoSite<Scalar>::MPO() const {
    if(all_mpo_parameters_have_been_set) {
        return mpo_internal;
    } else {
        throw std::runtime_error("All MPO parameters haven't been set yet.");
    }
}

template<typename Scalar>
const Eigen::Tensor<typename MpoSite<Scalar>::QuadScalar, 4> &MpoSite<Scalar>::MPO_q() const {
    if(all_mpo_parameters_have_been_set) {
        return mpo_internal_q;
    } else {
        throw std::runtime_error("All MPO parameters haven't been set yet.");
    }
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> MpoSite<Scalar>::MPO_energy_shifted_view(Scalar energy_shift_per_site) const {
    if(has_mpo() and all_mpo_parameters_have_been_set and energy_shift_per_site == energy_shift_mpo) return mpo_internal;
    auto mpo_build = get_mpo(energy_shift_per_site);
    mpo_build      = get_parity_shifted_mpo(mpo_build);
    mpo_build      = apply_edge_left(mpo_build, get_MPO_edge_left(mpo_build));
    mpo_build      = apply_edge_right(mpo_build, get_MPO_edge_right(mpo_build));
    return mpo_build;
}
template<typename Scalar>
Eigen::Tensor<typename MpoSite<Scalar>::QuadScalar, 4> MpoSite<Scalar>::MPO_energy_shifted_view_q(Scalar energy_shift_per_site) const {
    if(has_mpo() and all_mpo_parameters_have_been_set and energy_shift_per_site == energy_shift_mpo) return mpo_internal_q;
    auto mpo_build = get_mpo_q(energy_shift_per_site);
    mpo_build      = get_parity_shifted_mpo(mpo_build);
    mpo_build      = apply_edge_left(mpo_build, get_MPO_edge_left(mpo_build));
    mpo_build      = apply_edge_right(mpo_build, get_MPO_edge_right(mpo_build));
    return mpo_build;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> MpoSite<Scalar>::MPO_nbody_view(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const {
    auto mpo_build = get_mpo(energy_shift_mpo, nbody, skip);
    mpo_build      = get_parity_shifted_mpo(mpo_build);
    mpo_build      = apply_edge_left(mpo_build, get_MPO_edge_left(mpo_build));
    mpo_build      = apply_edge_right(mpo_build, get_MPO_edge_right(mpo_build));
    return mpo_build;
    // return get_mpo(energy_shift_mpo, nbody, skip);
}

template<typename Scalar>
Eigen::Tensor<typename MpoSite<Scalar>::QuadScalar, 4> MpoSite<Scalar>::MPO_nbody_view_q(std::optional<std::vector<size_t>> nbody,
                                                                                         std::optional<std::vector<size_t>> skip) const {
    auto mpo_build = get_mpo_q(energy_shift_mpo, nbody, skip);
    mpo_build      = get_parity_shifted_mpo(mpo_build);
    mpo_build      = apply_edge_left(mpo_build, get_MPO_edge_left(mpo_build));
    mpo_build      = apply_edge_right(mpo_build, get_MPO_edge_right(mpo_build));
    return mpo_build;
    // return get_mpo_q(energy_shift_mpo, nbody, skip);
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 4> &MpoSite<Scalar>::MPO2() const {
    if(has_mpo_squared() and all_mpo_parameters_have_been_set)
        return mpo_squared.value();
    else
        throw std::runtime_error("MPO squared has not been set.");
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> MpoSite<Scalar>::MPO2_nbody_view(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const {
    if(not nbody) return MPO2();
    auto mpo1 = MPO_nbody_view(nbody, std::move(skip));
    auto dim0 = mpo1.dimension(0) * mpo1.dimension(0);
    auto dim1 = mpo1.dimension(1) * mpo1.dimension(1);
    auto dim2 = mpo1.dimension(2);
    auto dim3 = mpo1.dimension(3);
    return mpo1.contract(mpo1, tenx::idx({3}, {2})).shuffle(tenx::array6{0, 3, 1, 4, 2, 5}).reshape(tenx::array4{dim0, dim1, dim2, dim3});
}

template<typename Scalar>
bool MpoSite<Scalar>::is_real() const {
    if constexpr(!settings::debug) {
        if(is_real_cached.has_value()) return is_real_cached.value();
    }
    bool is_real_mpo  = tenx::isReal(MPO());
    bool is_real_mpo2 = mpo_squared.has_value() ? tenx::isReal(mpo_squared.value()) : is_real_mpo;
    is_real_cached    = is_real_mpo && is_real_mpo2;
    return is_real_cached.value();
}

template<typename Scalar>
bool MpoSite<Scalar>::has_nan() const {
    if constexpr(!settings::debug) {
        if(has_nan_cached.value()) return has_nan_cached.value();
    }
    for(auto &param : get_parameters()) {
        if(param.second.type() == typeid(double))
            if(std::isnan(std::any_cast<double>(param.second))) {
                has_nan_cached = true;
                return true;
            }
        if(param.second.type() == typeid(long double))
            if(std::isnan(std::any_cast<long double>(param.second))) {
                has_nan_cached = true;
                return true;
            }
#if defined(DMRG_USE_QUADMATH) || defined(DMRG_USE_FLOAT128)
        if(param.second.type() == typeid(fp128)) {
            using std::isnan;
            if(isnan(std::any_cast<fp128>(param.second))) {
                has_nan_cached = true;
                return true;
            }
        }
#endif
    }
    bool has_nan_mpo  = tenx::hasNaN(mpo_internal);
    bool has_nan_mpo2 = mpo_squared.has_value() ? tenx::hasNaN(mpo_squared.value()) : has_nan_mpo;
    has_nan_cached    = has_nan_mpo && has_nan_mpo2;
    return has_nan_cached.value();
}

template<typename Scalar>
void MpoSite<Scalar>::assert_validity() const {
    for(auto &param : get_parameters()) {
        if(param.second.type() == typeid(double)) {
            if(std::isnan(std::any_cast<double>(param.second))) {
                print_parameter_names();
                print_parameter_values();
                throw except::runtime_error("Param [{}] = {}", param.first, std::any_cast<double>(param.second));
            }
        }
        if(param.second.type() == typeid(long double)) {
            if(std::isnan(std::any_cast<long double>(param.second))) {
                print_parameter_names();
                print_parameter_values();
                throw except::runtime_error("Param [{}] = {}", param.first, std::any_cast<long double>(param.second));
            }
        }
#if defined(DMRG_USE_QUADMATH) || defined(DMRG_USE_FLOAT128)
        if(param.second.type() == typeid(fp128)) {
            using std::isnan;
            if(isnan(std::any_cast<fp128>(param.second))) {
                print_parameter_names();
                print_parameter_values();
                throw except::runtime_error("Param [{}] = {}", param.first, f128_t(std::any_cast<fp128>(param.second)));
            }
        }
#endif
    }
    if(tenx::hasNaN(mpo_internal)) throw except::runtime_error("MPO has NAN on position {}", get_position());
    if(not tenx::isReal(mpo_internal)) throw except::runtime_error("MPO has IMAG on position {}", get_position());
    if(mpo_squared) {
        if(tenx::hasNaN(mpo_squared.value())) throw except::runtime_error("MPO2 squared has NAN on position {}", get_position());
        if(not tenx::isReal(mpo_squared.value())) throw except::runtime_error("MPO2 squared has IMAG on position {}", get_position());
    }
}

template<typename Scalar>
void MpoSite<Scalar>::set_position(size_t position_) {
    position = position_;
}

template<typename Scalar>
std::vector<std::string> MpoSite<Scalar>::get_parameter_names() const {
    std::vector<std::string> parameter_names;
    for(auto &item : get_parameters()) parameter_names.push_back(item.first);
    return parameter_names;
}
template<typename Scalar>
std::vector<std::any> MpoSite<Scalar>::get_parameter_values() const {
    std::vector<std::any> parameter_values;
    for(auto &item : get_parameters()) parameter_values.push_back(item.second);
    return parameter_values;
}

template<typename Scalar>
size_t MpoSite<Scalar>::get_position() const {
    if(position) {
        return position.value();
    } else {
        throw std::runtime_error("Position of MPO has not been set");
    }
}

template<typename Scalar>
bool MpoSite<Scalar>::has_energy_shifted_mpo() const {
    return energy_shift_mpo != RealScalar(0.0);
}
template<typename Scalar>
bool MpoSite<Scalar>::has_parity_shifted_mpo() const {
    return !parity_shift_axus_mpo.empty() and parity_shift_sign_mpo != 0;
}
template<typename Scalar>
bool MpoSite<Scalar>::has_parity_shifted_mpo2() const {
    return !parity_shift_axus_mpo2.empty() and parity_shift_sign_mpo2 != 0;
}

template<typename Scalar>
bool MpoSite<Scalar>::has_compressed_mpo_squared() const {
    // When H² = mpo*mpo is compressed, we typically find that the virtual bonds
    // have become smaller than they would otherwise. We can simply check that if they are smaller.
    /*           2
     *           |
     *      0---H²---1
     *           |
     *           3
     */
    if(not has_mpo_squared()) return false;
    auto        mpo    = get_mpo(energy_shift_mpo);
    const auto &mpo_sq = MPO2();
    auto        dp     = parity_shift_sign_mpo2 == 0 ? 0 : 2;
    auto        d0     = mpo.dimension(0) * mpo.dimension(0) + dp;
    auto        d1     = mpo.dimension(1) * mpo.dimension(1) + dp;
    if(get_position() == 0)
        return mpo_sq.dimension(1) < d1;
    else if(get_position() + 1 == settings::model::model_size)
        return mpo_sq.dimension(0) < d0;
    else
        return mpo_sq.dimension(0) < d0 or mpo_sq.dimension(1) < d1;
}

template<typename Scalar>
Scalar MpoSite<Scalar>::get_energy_shift_mpo() const {
    return energy_shift_mpo;
}
template<typename Scalar>
void MpoSite<Scalar>::set_energy_shift_mpo(Scalar site_energy) {
    if(energy_shift_mpo != site_energy) {
        energy_shift_mpo = site_energy;
        clear_mpo();
        clear_mpo_squared();
    }
}

template<typename Scalar>
[[nodiscard]] double MpoSite<Scalar>::get_global_energy_upper_bound() const {
    return global_energy_upper_bound;
}
template<typename Scalar>
[[nodiscard]] double MpoSite<Scalar>::get_local_energy_upper_bound() const {
    return local_energy_upper_bound;
}

template<typename Scalar>
void MpoSite<Scalar>::set_parity_shift_mpo(OptRitz ritz, int sign, std::string_view axis) {
    bool unset = ritz == OptRitz::NONE or sign == 0 or axis.empty();
    if(unset) {
        bool clear            = parity_shift_ritz_mpo != OptRitz::NONE or parity_shift_sign_mpo != 0 or !parity_shift_axus_mpo.empty();
        parity_shift_ritz_mpo = OptRitz::NONE;
        parity_shift_sign_mpo = 0;
        parity_shift_axus_mpo = {};
        if(clear) {
            clear_mpo();
            clear_mpo_squared();
        }
        return;
    }

    if(not qm::spin::half::is_valid_axis(axis)) {
        tools::log->warn("MpoSite[{}]::set_parity_shift_mpo: invalid axis {} | expected one of {}", get_position(), axis, qm::spin::half::valid_axis_str);
        return;
    }
    if(std::abs(sign) != 1) sign = qm::spin::half::get_sign(axis);
    if(std::abs(sign) != 1) {
        tools::log->warn("MpoSite<Scalar>::set_parity_shift_mpo: wrong sign value [{}] | expected -1 or 1", sign);
        return;
    }
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(ritz != parity_shift_ritz_mpo or sign != parity_shift_sign_mpo or axus != parity_shift_axus_mpo) {
        tools::log->trace("MpoSite[{}]::set_parity_shift_mpo: {} {}{}", get_position(), enum2sv(ritz), fmt::format("{:+}", sign).front(), axus);
        parity_shift_ritz_mpo = ritz;
        parity_shift_sign_mpo = sign;
        parity_shift_axus_mpo = axus;
        clear_mpo();
        clear_mpo_squared();
    }
}

template<typename Scalar>
std::tuple<OptRitz, int, std::string_view> MpoSite<Scalar>::get_parity_shift_mpo() const {
    return {parity_shift_ritz_mpo, parity_shift_sign_mpo, parity_shift_axus_mpo};
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> MpoSite<Scalar>::get_parity_shifted_mpo_squared(const Eigen::Tensor<Scalar, 4> &mpo_build) const {
    if(std::abs(parity_shift_sign_mpo2) != 1.0) return mpo_build;
    if(parity_shift_axus_mpo2.empty()) return mpo_build;
    // This redefines H² --> H² + Q(σ), where
    //      * Q(σ) = 0.5 * ( I - q*prod(σ) )
    //      * σ is a pauli matrix (usually σ^z)
    //      * 0.5 is a scalar that we multiply on the left edge as well.
    //      * q == parity_shift_sign_mpo2 is the sign of the parity sector we want to shift away from the target sector we are interested in.
    //        Often this is done to resolve a degeneracy. We multiply q just once on the left edge.
    // We add (I - q*prod(σ)) along the diagonal of the MPO
    //        MPO = |1  0  0  0|
    //              |h² 1  0  0|
    //              |0  0  1  0|
    //              |0  0  0  σ|
    //
    // Example 1: Let q == +1 (e.g. because target_axis=="+z"), then
    //            Q(σ)|ψ+⟩ = 0.5*(1 - q *(+1)) |ψ+⟩ = 0|ψ+⟩
    //            Q(σ)|ψ-⟩ = 0.5*(1 - q *(-1)) |ψ-⟩ = 1|ψ-⟩
    //
    // Example 2: For xDMRG we can add the projection on H² directly, as
    //                  H² --> (H² + Q(σ))
    //            Then, if q == +1 we get and
    //                  (H² + Q(σ)) |ψ+⟩ = (E + 0.5(1-1)) |ψ+⟩ = (E² + 0) |ψ+⟩ <--- target state
    //                  (H² + Q(σ)) |ψ-⟩ = (E + 0.5(1+1)) |ψ-⟩ = (E² + 1) |ψ-⟩
    //            If q == -1 instead we get
    //                  (H² + Q(σ)) |ψ+⟩ = (E + 0.5(1+1)) |ψ+⟩ = (E² + 1) |ψ+⟩
    //                  (H² + Q(σ)) |ψ-⟩ = (E + 0.5(1-1)) |ψ-⟩ = (E² + 0) |ψ-⟩ <--- target state
    //  Note:
    //  1) Var(H) is typically a number close to 0, so 1 adds a very large gap.
    //  2) We for xDMRG we could in principle add the projection on the mpo for H instead by defining
    //              H  -->  (H + iQ(σ))
    //              H² --> H^† H  = H² + Q(σ)² = H² + Q(σ)
    //     but this introduces imaginaries in all the MPOs which gives us a performance
    //     penalty by forcing us to use complex versions of all the expensive operations.
    //  3) For ground state DMRG (fDMRG) we can add the projection on H directly, as
    //              H --> (H - r*Q(σ))
    //     where r is the ritz (SR: r=-1, LR:r=+1) such that for q=+1 and  r = -1 (ground state search of + sector)
    //              (H + Q(σ)) |ψ+⟩ = (σ² + 0.5(1-1)) |ψ+⟩ = (E + 0) |ψ+⟩
    //              (H + Q(σ)) |ψ-⟩ = (σ² + 0.5(1+1)) |ψ-⟩ = (E + 1) |ψ-⟩

    auto d0 = mpo_build.dimension(0);
    auto d1 = mpo_build.dimension(1);
    auto d2 = mpo_build.dimension(2);
    auto d3 = mpo_build.dimension(3);
    using namespace qm::spin::half::tensor;
    auto pl = qm::spin::half::tensor::get_pauli(parity_shift_axus_mpo2);

    Eigen::Tensor<Scalar, 4> mpo2_with_parity_shift_op(d0 + 2, d1 + 2, d2, d3);
    mpo2_with_parity_shift_op.setZero();
    mpo2_with_parity_shift_op.slice(tenx::array4{0, 0, 0, 0}, mpo_build.dimensions())             = mpo_build;
    mpo2_with_parity_shift_op.slice(tenx::array4{d0, d1, 0, 0}, extent4).reshape(extent2)         = tenx::asScalarType<Scalar>(id);
    mpo2_with_parity_shift_op.slice(tenx::array4{d0 + 1, d1 + 1, 0, 0}, extent4).reshape(extent2) = tenx::asScalarType<Scalar>(pl);
    return mpo2_with_parity_shift_op;
}

template<typename Scalar>
void MpoSite<Scalar>::set_parity_shift_mpo_squared(int sign, std::string_view axis) {
    if(not qm::spin::half::is_valid_axis(axis)) {
        tools::log->warn("MpoSite[{}]::set_parity_shift_mpo_squared: invalid axis {} | expected one of {}", get_position(), axis,
                         qm::spin::half::valid_axis_str);
        return;
    }
    if(std::abs(sign) != 1) sign = qm::spin::half::get_sign(axis);
    if(std::abs(sign) != 1) {
        tools::log->warn("MpoSite<Scalar>::set_parity_shift_mpo2: wrong sign value [{}] | expected -1 or 1", sign);
        return;
    }
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(sign != parity_shift_sign_mpo2 or axus != parity_shift_axus_mpo2) {
        if constexpr(settings::debug)
            tools::log->trace("MpoSite[{}]::set_parity_shift_mpo_squared: {}{}", get_position(), fmt::format("{:+}", sign).front(), axus);
        parity_shift_sign_mpo2 = sign;
        parity_shift_axus_mpo2 = axus;
        clear_mpo_squared();
    }
}

template<typename Scalar>
std::pair<int, std::string_view> MpoSite<Scalar>::get_parity_shift_mpo_squared() const {
    return {parity_shift_sign_mpo2, parity_shift_axus_mpo2};
}

template<typename Scalar>
long MpoSite<Scalar>::size() const {
    return mpo_internal.size();
}
template<typename Scalar>
std::array<long, 4> MpoSite<Scalar>::dimensions() const {
    return mpo_internal.dimensions();
}

template<typename Scalar>
void MpoSite<Scalar>::print_parameter_names() const {
    for(auto &item : get_parameters()) fmt::print("{:<16}", item.first);
    fmt::print("\n");
}

template<typename Scalar>
void MpoSite<Scalar>::print_parameter_values() const {
    for(auto &item : get_parameters()) {
        if(item.second.type() == typeid(int)) fmt::print("{:<16}", std::any_cast<int>(item.second));
        if(item.second.type() == typeid(bool)) fmt::print("{:<16}", std::any_cast<bool>(item.second));
        if(item.second.type() == typeid(size_t)) fmt::print("{:<16}", std::any_cast<size_t>(item.second));
        if(item.second.type() == typeid(std::string)) fmt::print("{:<16}", std::any_cast<std::string>(item.second));
        if(item.second.type() == typeid(double)) fmt::print("{:<16.12f}", std::any_cast<double>(item.second));
    }
    fmt::print("\n");
}

template<typename Scalar>
void MpoSite<Scalar>::save_mpo(h5pp::File &file, std::string_view mpo_prefix) const {
    std::string dataset_name = fmt::format("{}/H_{}", mpo_prefix, get_position());
    file.writeDataset(MPO_energy_shifted_view(Scalar{}), dataset_name, H5D_layout_t::H5D_CONTIGUOUS);
    file.writeAttribute(get_position(), dataset_name, "position");
    for(auto &params : get_parameters()) {
        if(params.second.type() == typeid(double)) file.writeAttribute(std::any_cast<double>(params.second), dataset_name, params.first);
        if(params.second.type() == typeid(size_t)) file.writeAttribute(std::any_cast<size_t>(params.second), dataset_name, params.first);
        if(params.second.type() == typeid(uint64_t)) file.writeAttribute(std::any_cast<uint64_t>(params.second), dataset_name, params.first);
        if(params.second.type() == typeid(int)) file.writeAttribute(std::any_cast<int>(params.second), dataset_name, params.first);
        if(params.second.type() == typeid(bool)) file.writeAttribute(std::any_cast<bool>(params.second), dataset_name, params.first);
        if(params.second.type() == typeid(std::string)) file.writeAttribute(std::any_cast<std::string>(params.second), dataset_name, params.first);
    }
}

template<typename Scalar>
void MpoSite<Scalar>::load_mpo(const h5pp::File &file, std::string_view mpo_prefix) {
    std::string mpo_dset = fmt::format("{}/H_{}", mpo_prefix, get_position());
    TableMap    map;
    if(file.linkExists(mpo_dset)) {
        auto param_names = file.getAttributeNames(mpo_dset);
        for(auto &param_name : param_names) {
            auto param_type = file.getTypeInfoAttribute(mpo_dset, param_name);
            if(param_type.cppTypeIndex) {
                if(param_type.cppTypeIndex.value() == typeid(double)) map[param_name] = file.readAttribute<double>(mpo_dset, param_name);
                if(param_type.cppTypeIndex.value() == typeid(size_t)) map[param_name] = file.readAttribute<size_t>(mpo_dset, param_name);
                if(param_type.cppTypeIndex.value() == typeid(uint64_t)) map[param_name] = file.readAttribute<uint64_t>(mpo_dset, param_name);
                if(param_type.cppTypeIndex.value() == typeid(int)) map[param_name] = file.readAttribute<int>(mpo_dset, param_name);
                if(param_type.cppTypeIndex.value() == typeid(bool)) map[param_name] = file.readAttribute<bool>(mpo_dset, param_name);
                if(param_type.cppTypeIndex.value() == typeid(std::string)) map[param_name] = file.readAttribute<std::string>(mpo_dset, param_name);
            }
        }
        set_parameters(map);
        build_mpo();
        if(tenx::VectorMap(MPO()) != tenx::VectorCast(file.readDataset<Eigen::Tensor<Scalar, 4>>(mpo_dset)))
            throw std::runtime_error("Built MPO does not match the MPO on file");
    } else {
        throw except::runtime_error("Could not load MPO. Dataset [{}] does not exist", mpo_dset);
    }
}

template<typename Scalar>
std::size_t MpoSite<Scalar>::get_unique_id() const {
    if(unique_id) return unique_id.value();
    unique_id = hash::hash_buffer(MPO().data(), safe_cast<size_t>(MPO().size()));
    return unique_id.value();
}

template<typename Scalar>
std::size_t MpoSite<Scalar>::get_unique_id_sq() const {
    if(unique_id_sq) return unique_id_sq.value();
    unique_id_sq = hash::hash_buffer(MPO2().data(), safe_cast<size_t>(MPO2().size()));
    return unique_id_sq.value();
}
