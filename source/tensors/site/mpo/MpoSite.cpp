#include "MpoSite.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "io/fmt_f128_t.h"
#include "math/cast.h"
#include "math/float.h"
#include "math/hash.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "math/tenx/sfinae.h"
#include "qm/qm.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <config/settings.h>
#include <general/sfinae.h>
#include <h5pp/h5pp.h>
#include <utility>

template class MpoSite<cx64>;
template class MpoSite<cx128>;

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
    mpo_internal_q = Eigen::Tensor<cx128, 4>();
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
Eigen::Tensor<cx128, 4> MpoSite<Scalar>::get_mpo_q(Scalar energy_shift_per_site, std::optional<std::vector<size_t>> nbody,
                                                   std::optional<std::vector<size_t>> skip) const {
    // tools::log->trace("MpoSite<Scalar>::get_mpo_q(): Pointless upcast {} -> {}", sfinae::type_name<cx64>(), sfinae::type_name<cx128>());
    auto ereal = energy_shift_per_site; // cx64(static_cast<fp64>(energy_shift_per_site.real(), static_cast<fp64>(energy_shift_per_site.imag())));
    auto mpo   = get_mpo(ereal, nbody, skip);
    return mpo.unaryExpr([](auto z) { return std::complex<fp128>(static_cast<fp128>(z.real()), static_cast<fp128>(z.imag())); });
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
const Eigen::Tensor<cx128, 4> &MpoSite<Scalar>::MPO_q() const {
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
Eigen::Tensor<cx128, 4> MpoSite<Scalar>::MPO_energy_shifted_view_q(Scalar energy_shift_per_site) const {
    if(has_mpo() and all_mpo_parameters_have_been_set and energy_shift_per_site == energy_shift_mpo) return mpo_internal_q;
    auto mpo_build = get_mpo_q(energy_shift_per_site);
    mpo_build      = get_parity_shifted_mpo(mpo_build);
    mpo_build      = apply_edge_left(mpo_build, get_MPO_edge_left(mpo_build));
    mpo_build      = apply_edge_right(mpo_build, get_MPO_edge_right(mpo_build));
    return mpo_build;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::MPO_energy_shifted_view_as(Scalar energy_shift_per_site) const {
    static_assert(sfinae::is_any_v<T, fp32, fp64, fp128, cx32, cx64, cx128>);
    if constexpr(tenx::sfinae::is_single_prec_v<T> or tenx::sfinae::is_double_prec_v<T>) {
        return tenx::asScalarType<T>(MPO_energy_shifted_view(energy_shift_per_site));
    } else if constexpr(tenx::sfinae::is_quadruple_prec_v<T>) {
        return tenx::asScalarType<T>(MPO_energy_shifted_view_q(energy_shift_per_site));
    } else
        throw except::runtime_error("MPO_energy_shifted_view_as(): invalid type <{}>", sfinae::type_name<T>());
}

template Eigen::Tensor<fp32, 4>  MpoSite<>::MPO_energy_shifted_view_as(cx64 energy_shift_per_site) const;
template Eigen::Tensor<fp64, 4>  MpoSite<>::MPO_energy_shifted_view_as(cx64 energy_shift_per_site) const;
template Eigen::Tensor<fp128, 4> MpoSite<>::MPO_energy_shifted_view_as(cx64 energy_shift_per_site) const;
template Eigen::Tensor<cx32, 4>  MpoSite<>::MPO_energy_shifted_view_as(cx64 energy_shift_per_site) const;
template Eigen::Tensor<cx64, 4>  MpoSite<>::MPO_energy_shifted_view_as(cx64 energy_shift_per_site) const;
template Eigen::Tensor<cx128, 4> MpoSite<>::MPO_energy_shifted_view_as(cx64 energy_shift_per_site) const;

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
Eigen::Tensor<cx128, 4> MpoSite<Scalar>::MPO_nbody_view_q(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const {
    auto mpo_build = get_mpo_q(energy_shift_mpo, nbody, skip);
    mpo_build      = get_parity_shifted_mpo(mpo_build);
    mpo_build      = apply_edge_left(mpo_build, get_MPO_edge_left(mpo_build));
    mpo_build      = apply_edge_right(mpo_build, get_MPO_edge_right(mpo_build));
    return mpo_build;
    // return get_mpo_q(energy_shift_mpo, nbody, skip);
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const {
    static_assert(sfinae::is_any_v<T, fp32, fp64, fp128, cx32, cx64, cx128>);
    if constexpr(tenx::sfinae::is_single_prec_v<T> or tenx::sfinae::is_double_prec_v<T>) {
        return tenx::asScalarType<T>(MPO_nbody_view(nbody, skip));
    } else if constexpr(tenx::sfinae::is_quadruple_prec_v<T>) {
        return tenx::asScalarType<T>(MPO_nbody_view_q(nbody, skip));
    } else
        throw except::logic_error("MPO_nbody_view_as(): invalid type <{}>", sfinae::type_name<T>());
}
template Eigen::Tensor<fp32, 4>  MpoSite<>::MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<fp64, 4>  MpoSite<>::MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<fp128, 4> MpoSite<>::MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<cx32, 4>  MpoSite<>::MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<cx64, 4>  MpoSite<>::MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<cx128, 4> MpoSite<>::MPO_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;

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
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::MPO2_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const {
    static_assert(sfinae::is_any_v<T, fp32, fp64, cx32, cx64>);
    return tenx::asScalarType<T>(MPO2_nbody_view(nbody, skip));
}
template Eigen::Tensor<fp32, 4> MpoSite<>::MPO2_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<fp64, 4> MpoSite<>::MPO2_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<cx32, 4> MpoSite<>::MPO2_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;
template Eigen::Tensor<cx64, 4> MpoSite<>::MPO2_nbody_view_as(std::optional<std::vector<size_t>> nbody, std::optional<std::vector<size_t>> skip) const;

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
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::get_parity_shifted_mpo(const Eigen::Tensor<T, 4> &mpo_build) const {
    if(std::abs(parity_shift_sign_mpo) != 1) return mpo_build;
    if(parity_shift_axus_mpo.empty()) return mpo_build;
    // This redefines H --> H - r*Q(σ), where
    //      * Q(σ) = 0.5 * ( I - q*prod(σ) )
    //      * σ is a pauli matrix (usually σ^z)
    //      * 0.5 is a scalar that we multiply on the left edge as well.
    //      * r is the the shift direction depending on the ritz (target energy): ground state energy (r = -1, SR) or maximum energy state (r = +1, LR).
    //        We multiply r just once on the left edge.
    //      * q == parity_shift_sign_mpo is the sign of the parity sector we want to shift away from the target sector we are interested in.
    //        Often this is done to resolve a degeneracy. We multiply q just once on the left edge.
    // We add (I - q*prod(σ)) along the diagonal of the MPO
    //        MPO = |1  0  0  0|
    //              |h  1  0  0|
    //              |0  0  1  0|
    //              |0  0  0  σ|
    //
    // Example 1: Let q == +1 (e.g. because target_axis=="+z"), then
    //            Q(σ)|ψ+⟩ = 0.5*(1 - q *(+1)) |ψ+⟩ = 0|ψ+⟩
    //            Q(σ)|ψ-⟩ = 0.5*(1 - q *(-1)) |ψ-⟩ = 1|ψ-⟩
    //
    // Example 2: For fDMRG with r == -1 (ritz == SR: min energy state)  we can add the projection on H directly, as
    //                  H --> (H - r*Q(σ))
    //            Then, if q == +1 we get and
    //                  (H - r*Q(σ)) |ψ+⟩ = (E + 0.5(1-1)) |ψ+⟩ = (E + 0) |ψ+⟩ <--- min energy state
    //                  (H - r*Q(σ)) |ψ-⟩ = (E + 0.5(1+1)) |ψ-⟩ = (E + 1) |ψ-⟩
    //            If q == -1 instead we get
    //                  (H - r*Q(σ)) |ψ+⟩ = (E + 0.5(1+1)) |ψ+⟩ = (E + 1) |ψ+⟩
    //                  (H - r*Q(σ)) |ψ-⟩ = (E + 0.5(1-1)) |ψ-⟩ = (E + 0) |ψ-⟩ <--- min energy state
    //
    // Example 3: For fDMRG with r == +1 (ritz == LR: max energy state) we can add the projection on H directly, as
    //                  H --> (H - r*Q(σ))
    //            Then, if q == +1 we get
    //                  (H - r*Q(σ)) |ψ+⟩ = (E - 0.5(1-1)) |ψ+⟩ = (E - 0) |ψ+⟩ <--- max energy state
    //                  (H - r*Q(σ)) |ψ-⟩ = (E - 0.5(1+1)) |ψ-⟩ = (E - 1) |ψ-⟩
    //            If q == -1 instead we get
    //                  (H - r*Q(σ)) |ψ+⟩ = (E - 0.5(1+1)) |ψ+⟩ = (E - 1) |ψ+⟩
    //                  (H - r*Q(σ)) |ψ-⟩ = (E - 0.5(1-1)) |ψ-⟩ = (E - 0) |ψ-⟩ <--- max energy state
    //
    //
    auto d0 = mpo_build.dimension(0);
    auto d1 = mpo_build.dimension(1);
    auto d2 = mpo_build.dimension(2);
    auto d3 = mpo_build.dimension(3);
    auto id = qm::spin::half::id;
    auto pl = qm::spin::half::get_pauli(parity_shift_axus_mpo);

    Eigen::Tensor<T, 4> mpo_with_parity_shift_op(d0 + 2, d1 + 2, d2, d3);
    mpo_with_parity_shift_op.setZero();
    mpo_with_parity_shift_op.slice(tenx::array4{0, 0, 0, 0}, mpo_build.dimensions())             = mpo_build;
    mpo_with_parity_shift_op.slice(tenx::array4{d0, d1, 0, 0}, extent4).reshape(extent2)         = tenx::TensorMap(id).cast<T>();
    mpo_with_parity_shift_op.slice(tenx::array4{d0 + 1, d1 + 1, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(pl).cast<T>();
    return mpo_with_parity_shift_op;
}
template Eigen::Tensor<cx64, 4>  MpoSite<>::get_parity_shifted_mpo(const Eigen::Tensor<cx64, 4> &mpo_build) const;
template Eigen::Tensor<cx128, 4> MpoSite<>::get_parity_shifted_mpo(const Eigen::Tensor<cx128, 4> &mpo_build) const;

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
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO_edge_left(const Eigen::Tensor<T, 4> &mpo) const {
    using Real = typename Eigen::NumTraits<T>::Real;
    if(mpo.size() == 0) throw except::runtime_error("mpo({}): can't build the left edge: mpo has not been built yet", get_position());
    auto                ldim = mpo.dimension(0);
    Eigen::Tensor<T, 1> ledge(ldim);
    if(ldim == 1) {
        // Thin edge (it was probably already applied to the left-most MPO)
        ledge.setConstant(cx64(1.0, 0.0));
    } else {
        ledge.setZero();
        if(parity_shift_sign_mpo == 0) {
            /*
             *  MPO = |1 0|
             *        |h 1|
             *  So the left edge picks out the row for h
             */
            ledge(ldim - 1) = 1;
        } else {
            /*
             *  MPO = |1  0  0  0|
             *        |h  1  0  0|
             *        |0  0  1  0|
             *        |0  0  0  σ|
             *  So the left edge picks out the row for h, as well as 1 and σ along the diagonal
             *  We also put the signs r,q and factor 0.5 if this is the first site.
             */
            // This redefines H --> H - r*Q(σ), where
            //      * Q(σ) = 0.5 * ( I - q*prod(σ) )
            //      * σ is a pauli matrix (usually σ^z)
            //      * 0.5 is a scalar that we multiply on the left edge as well.
            //      * r is the shift direction depending on the ritz (target energy): ground state energy (r = -1, SR) or maximum energy state (r = +1, LR).
            //        We multiply r just once on the left edge.
            //      * q == parity_shift_sign_mpo is the sign of the parity sector we want to shift away from the target sector we are interested in.
            //        Often this is done to resolve a degeneracy. We multiply q just once on the left edge.
            double a = 1.0; // The shift amount. Use a large amount to make sure the target energy in that axis is actually extremal
            double q = 1.0; // The parity sign to shift
            double r = 1.0; // The energy shift direction (-1 is down, +1 is up)

            if(position == 0) {
                a = global_energy_upper_bound;
                q = -parity_shift_sign_mpo;
                switch(parity_shift_ritz_mpo) {
                    case OptRitz::SR: r = -1.0; break;
                    case OptRitz::LR: r = +1.0; break;
                    case OptRitz::SM: r = -1.0; break;
                    case OptRitz::NONE: r = 0.0; break;
                    // case OptRitz::LM: r = -1.0; break; // TODO this one probably doesn't make sense
                    default: throw except::runtime_error("expected ritz SR or LR, got: {}", enum2sv(parity_shift_ritz_mpo));
                }
            }

            if(get_position() == 0)
                tools::log->trace("Shifting MPO energy of the {:+}{} parity sector by E -> E{:+.8f} (estimated Emax)", q, parity_shift_axus_mpo, a);
            ledge(ldim - 3) = static_cast<Real>(1.0); // The bottom left corner
            ledge(ldim - 2) = static_cast<Real>(-a * r * 0.5);
            ledge(ldim - 1) = static_cast<Real>(-a * r * 0.5 * q);
        }
    }
    return ledge;
}
template Eigen::Tensor<cx64, 1>  MpoSite<>::get_MPO_edge_left(const Eigen::Tensor<cx64, 4> &mpo) const;
template Eigen::Tensor<cx128, 1> MpoSite<>::get_MPO_edge_left(const Eigen::Tensor<cx128, 4> &mpo) const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO_edge_right(const Eigen::Tensor<T, 4> &mpo) const {
    using Real = typename Eigen::NumTraits<T>::Real;
    if(mpo.size() == 0) throw except::runtime_error("mpo({}): can't build the right edge: mpo has not been built yet", get_position());
    auto                rdim = mpo.dimension(1);
    Eigen::Tensor<T, 1> redge(rdim);
    if(rdim == 1) {
        // Thin edge (it was probably already applied to the right-most MPO
        redge.setConstant(T(static_cast<Real>(1.0), static_cast<Real>(0.0)));
    } else {
        redge.setZero();

        if(parity_shift_sign_mpo == 0) {
            /*
             *  MPO = |1 0|
             *        |h 1|
             *  So the right edge picks out the column for h
             */
            redge(0) = 1; // The bottom left corner
        } else {
            /*
             *  MPO = |1  0  0  0|
             *        |h  1  0  0|
             *        |0  0  1  0|
             *        |0  0  0  σ|
             *  So the right edge picks out the column for h, as well as 1 and σ along the diagonal
             */
            redge(0)        = static_cast<Real>(1.0); // The bottom left corner of the original non-parity-shifted mpo
            redge(rdim - 2) = static_cast<Real>(1.0);
            redge(rdim - 1) = static_cast<Real>(1.0);
        }
    }
    return redge;
}

template Eigen::Tensor<cx64, 1>  MpoSite<>::get_MPO_edge_right(const Eigen::Tensor<cx64, 4> &mpo) const;
template Eigen::Tensor<cx128, 1> MpoSite<>::get_MPO_edge_right(const Eigen::Tensor<cx128, 4> &mpo) const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::apply_edge_left(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeL) const {
    if(mpo.dimension(0) == 1 or get_position() != 0) return mpo;
    if(mpo.dimension(0) != edgeL.dimension(0))
        throw except::logic_error("apply_edge_left: dimension mismatch: mpo {} | edgeL {}", mpo.dimensions(), edgeL.dimensions());
    auto  tmp     = mpo;
    auto  dim     = tmp.dimensions();
    auto &threads = tenx::threads::get();
    tmp.resize(tenx::array4{1, dim[1], dim[2], dim[3]});
    tmp.device(*threads->dev) = edgeL.reshape(tenx::array2{1, edgeL.size()}).contract(mpo, tenx::idx({1}, {0}));
    return tmp;
}
template Eigen::Tensor<cx64, 4>  MpoSite<>::apply_edge_left(const Eigen::Tensor<cx64, 4> &mpo, const Eigen::Tensor<cx64, 1> &edgeL) const;
template Eigen::Tensor<cx128, 4> MpoSite<>::apply_edge_left(const Eigen::Tensor<cx128, 4> &mpo, const Eigen::Tensor<cx128, 1> &edgeL) const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::apply_edge_right(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeR) const {
    if(mpo.dimension(1) == 1 or get_position() + 1 != settings::model::model_size) return mpo;
    if(mpo.dimension(1) != edgeR.dimension(0))
        throw except::logic_error("apply_edge_right: dimension mismatch: mpo {} | edgeR {}", mpo.dimensions(), edgeR.dimensions());
    auto  tmp     = mpo;
    auto  dim     = tmp.dimensions();
    auto &threads = tenx::threads::get();
    tmp.resize(tenx::array4{dim[0], 1, dim[2], dim[3]});
    tmp.device(*threads->dev) = mpo.contract(edgeR.reshape(tenx::array2{edgeR.size(), 1}), tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
    return tmp;
}
template Eigen::Tensor<cx64, 4>  MpoSite<>::apply_edge_right(const Eigen::Tensor<cx64, 4> &mpo, const Eigen::Tensor<cx64, 1> &edgeR) const;
template Eigen::Tensor<cx128, 4> MpoSite<>::apply_edge_right(const Eigen::Tensor<cx128, 4> &mpo, const Eigen::Tensor<cx128, 1> &edgeR) const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO_edge_left() const {
    if constexpr(tenx::sfinae::is_single_prec_v<T> or tenx::sfinae::is_double_prec_v<T>)
        return tenx::asScalarType<T>(get_MPO_edge_left(mpo_internal));
    else if constexpr(tenx::sfinae::is_quadruple_prec_v<T>)
        return tenx::asScalarType<T>(get_MPO_edge_left(mpo_internal_q));
    else {
        static_assert(sfinae::invalid_type_v<T>);
        throw std::logic_error("Invalid type");
    }
}
template Eigen::Tensor<fp32, 1>  MpoSite<>::get_MPO_edge_left() const;
template Eigen::Tensor<fp64, 1>  MpoSite<>::get_MPO_edge_left() const;
template Eigen::Tensor<fp128, 1> MpoSite<>::get_MPO_edge_left() const;
template Eigen::Tensor<cx32, 1>  MpoSite<>::get_MPO_edge_left() const;
template Eigen::Tensor<cx64, 1>  MpoSite<>::get_MPO_edge_left() const;
template Eigen::Tensor<cx128, 1> MpoSite<>::get_MPO_edge_left() const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO_edge_right() const {
    if constexpr(tenx::sfinae::is_single_prec_v<T> or tenx::sfinae::is_double_prec_v<T>)
        return tenx::asScalarType<T>(get_MPO_edge_right(mpo_internal));
    else if constexpr(tenx::sfinae::is_quadruple_prec_v<T>)
        return tenx::asScalarType<T>(get_MPO_edge_right(mpo_internal_q));
    else {
        static_assert(sfinae::invalid_type_v<T>);
        throw std::logic_error("Invalid type");
    }
}
template Eigen::Tensor<fp32, 1>  MpoSite<>::get_MPO_edge_right() const;
template Eigen::Tensor<fp64, 1>  MpoSite<>::get_MPO_edge_right() const;
template Eigen::Tensor<fp128, 1> MpoSite<>::get_MPO_edge_right() const;
template Eigen::Tensor<cx32, 1>  MpoSite<>::get_MPO_edge_right() const;
template Eigen::Tensor<cx64, 1>  MpoSite<>::get_MPO_edge_right() const;
template Eigen::Tensor<cx128, 1> MpoSite<>::get_MPO_edge_right() const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO2_edge_left() const {
    using Real = typename Eigen::NumTraits<T>::Real;
    if(mpo_squared.has_value()) {
        auto ldim = mpo_squared->dimension(0);
        if(ldim == 1) {
            // Thin edge (it was probably already applied to the right-most MPO
            auto ledge2 = Eigen::Tensor<T, 1>(ldim);
            ledge2.setConstant(T(static_cast<Real>(1.0), static_cast<Real>(0.0)));
            return ledge2;
        }
    }
    /* Start by making a left edge that would fit a raw mpo
     *  MPO = |1 0|
     *        |h 1|
     *  The left edge should pick out the last row
     */
    auto mpo1  = get_mpo(cx64(0.0, 0.0));
    auto d0    = mpo1.dimension(0);
    auto ledge = Eigen::Tensor<T, 1>(d0);
    ledge.setZero();
    ledge(d0 - 1) = 1;
    auto ledge2   = ledge.contract(ledge, tenx::idx()).reshape(tenx::array1{d0 * d0});
    if(std::abs(parity_shift_sign_mpo2) != 1.0) return ledge2;

    double q = 1.0;
    double a = 1.0;                               // The shift amount (select 1.0 to shift up by 1.0)
    if(position == 0) q = parity_shift_sign_mpo2; // Selects the opposite sector sign (only needed on one MPO)
    auto ledge2_with_shift = Eigen::Tensor<T, 1>(d0 * d0 + 2);
    ledge2_with_shift.setZero();
    ledge2_with_shift.slice(tenx::array1{0}, ledge2.dimensions()) = ledge2;
    ledge2_with_shift(d0 * d0 + 0)                                = static_cast<Real>(a * 0.5);        // 1.0;
    ledge2_with_shift(d0 * d0 + 1)                                = static_cast<Real>(a * 0.5 * (-q)); // 1.0;

    return ledge2_with_shift;
}
template Eigen::Tensor<cx64, 1>  MpoSite<>::get_MPO2_edge_left() const;
template Eigen::Tensor<cx128, 1> MpoSite<>::get_MPO2_edge_left() const;

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO2_edge_right() const {
    if(mpo_squared.has_value()) {
        auto rdim = mpo_squared->dimension(1);
        if(rdim == 1) {
            // Thin edge (it was probably already applied to the right-most MPO
            auto redge2 = Eigen::Tensor<T, 1>(rdim);
            redge2.setConstant(1.0);
            return redge2;
        }
    }
    /* Start by making a right edge that would fit a raw mpo
     *  MPO = |1 0|
     *        |h 1|
     *  The right edge should pick out the first column
     */
    auto mpo1  = get_mpo(cx64(0.0, 0.0));
    auto d0    = mpo1.dimension(1);
    auto redge = Eigen::Tensor<T, 1>(d0);
    redge.setZero();
    redge(0)    = 1;
    auto redge2 = redge.contract(redge.conjugate(), tenx::idx()).reshape(tenx::array1{d0 * d0});
    if(std::abs(parity_shift_sign_mpo2) != 1.0) return redge2;
    auto redge2_with_shift = Eigen::Tensor<T, 1>(d0 * d0 + 2);
    redge2_with_shift.setZero();
    redge2_with_shift.slice(tenx::array1{0}, redge2.dimensions()) = redge2;
    redge2_with_shift(d0 * d0 + 0)                                = static_cast<fp128>(1.0); // 0.5;
    redge2_with_shift(d0 * d0 + 1)                                = static_cast<fp128>(1.0); // 0.5 * q;
    return redge2_with_shift;
}
template Eigen::Tensor<cx64, 1>  MpoSite<>::get_MPO2_edge_right() const;
template Eigen::Tensor<cx128, 1> MpoSite<>::get_MPO2_edge_right() const;

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

//
// const std::any &class_model_base::find_val(const Parameters &parameters, std::string_view key) const {
//    for(auto &param : parameters) {
//        if(key == param.first) return param.second;
//    }
//    throw std::runtime_error("No parameter named [" + std::string(key) + "]");
//}
//
// std::any &class_model_base::find_val(Parameters &parameters, std::string_view key) const {
//    for(auto &param : parameters) {
//        if(key == param.first) return param.second;
//    }
//    throw std::runtime_error("No parameter named [" + std::string(key) + "]");
//}
//
//

template<typename Scalar>
void MpoSite<Scalar>::save_mpo(h5pp::File &file, std::string_view mpo_prefix) const {
    std::string dataset_name = fmt::format("{}/H_{}", mpo_prefix, get_position());
    file.writeDataset(MPO_energy_shifted_view(cx64(0.0, 0.0)), dataset_name, H5D_layout_t::H5D_CONTIGUOUS);
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
