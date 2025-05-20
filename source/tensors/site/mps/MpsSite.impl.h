#pragma once
#include "math/tenx.h"
// tenx must appear first
#include "config/debug.h"
#include "config/enums.h"
#include "math/float.h"
#include "math/hash.h"
#include "math/linalg/tensor/to_string.h"
#include "math/num.h"
#include "MpsSite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include <fmt/ranges.h>
#include <general/sfinae.h>
#include <utility>

namespace settings {
    inline constexpr bool debug_merge       = false;
    inline constexpr bool debug_apply_mpo   = false;
    inline constexpr bool verbose_merge     = false;
    inline constexpr bool verbose_apply_mpo = false;
}

template<typename Scalar>
MpsSite<Scalar>::MpsSite() = default;

// We need to define the destructor and other special functions
// because we enclose data in unique_ptr for this pimpl idiom.
// Otherwise, unique_ptr will forcibly inline its own default deleter.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
template<typename Scalar>
MpsSite<Scalar>::~MpsSite() = default; // default dtor
template<typename Scalar>
MpsSite<Scalar>::MpsSite(MpsSite &&other) noexcept = default; // default move ctor
template<typename Scalar>
MpsSite<Scalar> &MpsSite<Scalar>::operator=(MpsSite &&other) noexcept = default; // default move assign
template<typename Scalar>
MpsSite<Scalar>::MpsSite(const MpsSite &other) = default;
template<typename Scalar>
MpsSite<Scalar> &MpsSite<Scalar>::operator=(const MpsSite &other) = default;

template<typename Scalar>
bool MpsSite<Scalar>::isCenter() const {
    return LC.has_value();
}
template<typename Scalar>
bool MpsSite<Scalar>::has_L() const {
    return L.has_value();
}
template<typename Scalar>
bool MpsSite<Scalar>::has_M() const {
    return M.has_value();
}
template<typename Scalar>
bool MpsSite<Scalar>::has_LC() const {
    return LC.has_value();
}

template<typename Scalar>
Eigen::DSizes<long, 3> MpsSite<Scalar>::dimensions() const {
    return Eigen::DSizes<long, 3>{spin_dim(), get_chiL(), get_chiR()};
}

template<typename Scalar>
bool MpsSite<Scalar>::is_real() const {
    if(std::is_floating_point_v<Scalar>) return true;
    if constexpr(settings::debug) {
        return tenx::isReal(get_M_bare()) and tenx::isReal(get_L());
    } else {
        is_real_cached = is_real_cached.value_or(tenx::isReal(get_M_bare()) and tenx::isReal(get_L()));
        return is_real_cached.value();
    }
}
template<typename Scalar>
bool MpsSite<Scalar>::has_nan() const {
    if constexpr(settings::debug) {
        return tenx::hasNaN(get_M_bare()) or tenx::hasNaN(get_L());
    } else {
        has_nan_cached = has_nan_cached.value_or(tenx::hasNaN(get_M_bare()) or tenx::hasNaN(get_L()));
        return has_nan_cached.value();
    }
}
template<typename Scalar>
bool MpsSite<Scalar>::is_normalized(RealScalar prec) const {
    if constexpr(!settings::debug) {
        if(is_norm_cached.has_value()) return is_norm_cached.value();
    }
    constexpr auto eps  = std::numeric_limits<RealScalar>::epsilon();
    const auto     size = static_cast<RealScalar>(get_M_bare().size());
    prec                = std::max(prec, eps * std::sqrt(size)*100);
    auto t_dbg          = tid::tic_token("is_normalized", tid::level::highest);
    if(isCenter() or get_label() == "AC") {
        auto norm      = tools::common::contraction::contract_mps_norm(get_M());
        tools::log->info("AC norm: {:.16f} (error {:.4e}", fp(norm), fp(norm - Scalar{1}));
        is_norm_cached = std::abs(norm - Scalar{1}) <= prec;
        return is_norm_cached.value();
    }
    if(get_label() == "A") {
        auto id        = tools::common::contraction::contract_mps_partial<std::array{0l, 1l}>(get_M_bare());
        is_norm_cached = tenx::isIdentity(id, prec);
        return is_norm_cached.value();
    }
    if(get_label() == "B") {
        auto id        = tools::common::contraction::contract_mps_partial<std::array{0l, 2l}>(get_M_bare());
        is_norm_cached = tenx::isIdentity(id, prec);
        return is_norm_cached.value();
    }
    throw except::runtime_error("MpsSite<Scalar>::is_identity: unexpected label: {}", get_label());
}

template<typename Scalar>
void MpsSite<Scalar>::assert_validity() const {
    if(LC and get_label() != "AC")
        throw except::runtime_error("MpsSite<Scalar>::assert_validity: Found LC at pos {} with label {}", get_position(), get_label());
    if(not LC and get_label() == "AC")
        throw except::runtime_error("MpsSite<Scalar>::assert_validity: Missing LC at pos {} with label {}", get_position(), get_label());
    if(has_nan()) throw except::runtime_error("MpsSite<Scalar>::assert_validity(): MPS (M or L) at position {} has NaN's", get_position());
    assert_dimensions();
}

template<typename Scalar>
void MpsSite<Scalar>::assert_dimensions() const {
    if(get_label() == "B" and get_chiR() != get_L().dimension(0))
        throw except::runtime_error("MpsSite: Assert failed: Dimensions for B and L are incompatible at site {}: B {} | L {}", get_position(),
                                    get_M_bare().dimensions(), get_L().dimensions());
    if(get_label() == "A" and get_chiL() != get_L().dimension(0))
        throw except::runtime_error("MpsSite: Assert failed: Dimensions for A and L are incompatible at site {}: A {} | L {}", get_position(),
                                    get_M_bare().dimensions(), get_L().dimensions());
    if(get_label() == "AC") {
        if(get_chiL() != get_L().dimension(0))
            throw except::runtime_error("MpsSite: Assert failed: Dimensions for AC and L are incompatible at site {}: AC {} | L {}", get_position(),
                                        get_M_bare().dimensions(), get_L().dimensions());
        if(get_chiR() != get_LC().dimension(0))
            throw except::runtime_error("MpsSite: Assert failed: Dimensions for AC and L are incompatible at site {}: AC {} | LC{}", get_position(),
                                        get_M_bare().dimensions(), get_L().dimensions());
    }
}

template<typename Scalar>
void MpsSite<Scalar>::assert_normalized(RealScalar prec) const {
    if(not is_normalized(prec))
        throw except::runtime_error("MpsSite<Scalar>::assert_normalized({0:.2e}): {1}^dagger {1} is not normalized at pos {2}", fp(prec), get_label(),
                                    get_position());
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 3> &MpsSite<Scalar>::get_M_bare() const {
    if(M) {
        if(M.value().size() == 0) throw except::runtime_error("MpsSite<Scalar>::get_M_bare(): M has size 0 at position {}", get_position());
        return M.value();
    } else
        throw except::runtime_error("MpsSite<Scalar>::get_M_bare(): M has not been set at position {}", get_position());
}
template<typename Scalar>
const Eigen::Tensor<Scalar, 3> &MpsSite<Scalar>::get_M() const {
    //    auto t_get = tid::tic_scope("get_M", tid::level::highest);
    if(isCenter()) {
        if(LC.value().dimension(0) != get_M_bare().dimension(2))
            throw except::runtime_error("MpsSite<Scalar>::get_M(): M and LC dim mismatch: {} != {} at position {}", get_M_bare().dimension(2),
                                        LC.value().dimension(0), get_position());
        if(MC) {
            if(MC.value().size() == 0) throw except::runtime_error("MpsSite<Scalar>::get_M(): MC has size 0 at position {}", get_position());
            return MC.value();
        } else {
            MC = Eigen::Tensor<Scalar, 3>(spin_dim(), get_chiL(), get_chiR());
            tools::common::contraction::contract_mps_bnd(MC.value(), get_M_bare(), get_LC());
            if(MC->size() == 0) throw except::runtime_error("MpsSite<Scalar>::get_M(): built MC with size 0 at position {}", get_position());
            return MC.value();
        }
    } else
        return get_M_bare();
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 1> &MpsSite<Scalar>::get_L() const {
    if(L) {
        if constexpr(settings::debug) {
            if(L->size() == 0) throw except::runtime_error("MpsSite<Scalar>::get_L(): L has size 0 at position {} | label {}", get_position(), get_label());
            if(get_label() != "B" and L->size() != get_chiL())
                throw except::runtime_error("MpsSite<Scalar>::get_L(): L.size() {} != chiL {} at position {} | M dims {} | label {}", L->size(), get_chiL(),
                                            get_position(), dimensions(), get_label());
            if(get_label() == "B" and L->size() != get_chiR())
                throw except::runtime_error("MpsSite<Scalar>::get_L(): L.size() {} != chiR {} at position {} | M dims {} | label {}", L->size(), get_chiR(),
                                            get_position(), dimensions(), get_label());
        }
        return L.value();
    } else
        throw except::runtime_error("MpsSite<Scalar>::get_L(): L has not been set at position {}", get_position());
}
template<typename Scalar>
const Eigen::Tensor<Scalar, 1> &MpsSite<Scalar>::get_LC() const {
    if(isCenter()) {
        if(LC->dimension(0) != get_M_bare().dimension(2))
            throw except::runtime_error("MpsSite<Scalar>::get_LC(): M dimensions {} are incompatible with LC dimensions {} at position {}",
                                        get_M_bare().dimensions(), LC.value().dimensions(), get_position());
        if(LC.value().size() == 0) throw except::runtime_error("MpsSite<Scalar>::get_LC(): LC has size 0 at position {}", get_position());
        return LC.value();
    } else
        throw except::runtime_error("MpsSite<Scalar>::get_LC(): Site at position {} is not a center", get_position());
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3> &MpsSite<Scalar>::get_M_bare() {
    return const_cast<Eigen::Tensor<Scalar, 3> &>(std::as_const(*this).get_M_bare());
}
template<typename Scalar>
Eigen::Tensor<Scalar, 3> &MpsSite<Scalar>::get_M() {
    return const_cast<Eigen::Tensor<Scalar, 3> &>(std::as_const(*this).get_M());
}
template<typename Scalar>
Eigen::Tensor<Scalar, 1> &MpsSite<Scalar>::get_L() {
    return const_cast<Eigen::Tensor<Scalar, 1> &>(std::as_const(*this).get_L());
}
template<typename Scalar>
Eigen::Tensor<Scalar, 1> &MpsSite<Scalar>::get_LC() {
    return const_cast<Eigen::Tensor<Scalar, 1> &>(std::as_const(*this).get_LC());
}

template<typename Scalar>
double MpsSite<Scalar>::get_truncation_error() const {
    return truncation_error;
}
template<typename Scalar>
double MpsSite<Scalar>::get_truncation_error_LC() const {
    return truncation_error_LC;
}
template<typename Scalar>
double MpsSite<Scalar>::get_truncation_error_last() const {
    return truncation_error_last;
}
template<typename Scalar>
std::string_view MpsSite<Scalar>::get_label() const {
    if(label.empty()) throw except::runtime_error("No label found at position {}", get_position());
    return label;
}
template<typename Scalar>
std::string MpsSite<Scalar>::get_tag() const {
    return fmt::format("{}[{}]", get_label(), get_position());
}

template<typename Scalar>
std::tuple<long, long, long> MpsSite<Scalar>::get_dims() const {
    return {spin_dim(), get_chiL(), get_chiR()};
}
template<typename Scalar>
long MpsSite<Scalar>::spin_dim() const {
    return get_M_bare().dimension(0);
}
template<typename Scalar>
long MpsSite<Scalar>::get_chiL() const {
    return get_M_bare().dimension(1);
}
template<typename Scalar>
long MpsSite<Scalar>::get_chiR() const {
    return get_M_bare().dimension(2);
}

template<typename Scalar>
void MpsSite<Scalar>::set_position(const size_t position_) {
    position = position_;
    MC.reset();
}

template<typename Scalar>
void MpsSite<Scalar>::set_truncation_error(double error /* Negative is ignored */) {
    if(error >= 0.0) {
        // tools::log->warn("Setting truncation error on site {}: {:8.5e}", get_tag(), error);
        truncation_error      = error;
        truncation_error_last = error;
    }
}

template<typename Scalar>
void MpsSite<Scalar>::set_truncation_error_LC(double error /* Negative is ignored */) {
    if(error >= 0.0) {
        // tools::log->warn("Setting truncation error on site {}: {:8.5e} (LC)", get_tag(), error);
        truncation_error_LC   = error;
        truncation_error_last = error;
    }
}

template<typename Scalar>
void MpsSite<Scalar>::set_label(std::string_view label_) {
    // If we are flipping the kind of site from B to non B (or vice-versa), then the L matrix
    // should be removed since it changes side.
    if(not label.empty() and not label_.empty() and label.front() != label_.front()) unset_L();
    label = label_;
}
template<typename Scalar>
void MpsSite<Scalar>::unset_LC() {
    LC             = std::nullopt;
    MC             = std::nullopt;
    unique_id      = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
    is_norm_cached = std::nullopt;
    if(label == "AC") label = "A";
}

template<typename Scalar>
void MpsSite<Scalar>::unset_L() {
    L              = std::nullopt;
    unique_id      = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
    is_norm_cached = std::nullopt;
}

template<typename Scalar>
void MpsSite<Scalar>::unset_truncation_error() {
    truncation_error = -1.0;
}

template<typename Scalar>
void MpsSite<Scalar>::unset_truncation_error_LC() {
    truncation_error_LC = -1.0;
}

template<typename Scalar>
void MpsSite<Scalar>::fuse_mps(const MpsSite &other) {
    //    auto t_fuse = tid::tic_scope("fuse", tid::level::highest);
    // This operation is done when merging mps after an svd split
    auto tag  = get_tag();       // tag, (example: A[3])
    auto otag = other.get_tag(); // other tag, (example: AC[3])

    if(get_position() != other.get_position()) throw except::runtime_error("MpsSite({})::fuse_mps: position mismatch {}", tag, otag);

    if(not other.has_M()) throw except::runtime_error("MpsSite({})::fuse_mps: Got other mps {} with undefined M", tag, otag);
    if constexpr(settings::verbose_merge) {
        if(other.has_L())
            tools::log->trace("MpsSite({})::fuse_mps: Merging {} | M {} | L {}", tag, otag, other.dimensions(), other.get_L().dimensions());
        else
            tools::log->trace("MpsSite({})::fuse_mps: Merging {} | M {} | L nullopt", tag, otag, other.dimensions());
    }
    // Copy the A/AC/B label
    set_label(other.get_label());

    // We have to copy the bare "M", i.e. not MC, which would include LC.
    set_M(other.get_M_bare());

    // If we are flipping the kind of site from B to non B (or vice versa), then the L matrix
    // should be removed since it changes side.
    if(get_label().front() != other.get_label().front()) unset_L();

    // We should only copy the L matrix if it exists.
    // If it does not exist it may just not have been set.
    // For instance, when splitting a multisite tensor into its contituent mps sites,
    // the edge mps sites have empty L matrices on them because these are not updated.
    // This is intended, but should be checked for.
    // Note that if there is no L on the other mps, there has to exist a previous one
    // on this mps -- we can't leave here without having defined L.

    if(other.has_L())
        set_L(other.get_L(), other.get_truncation_error());
    else if(not has_L()) {
        // Now we have a situation where no L was given, and no L is found on this site.
        // For edge-sites this is simple: then the mps has edge-dimension == 1, so the only
        // way to have a normalized L is to set it to 1. Otherwise, this is a failure.
        auto one = Eigen::Tensor<Scalar, 1>(1).setConstant(RealScalar(1.0));
        if((other.get_label() != "B" and get_chiL() == 1) or (other.get_label() == "B" and get_chiR() == 1))
            set_L(one);
        else
            throw except::runtime_error("MpsSite({})::fuse_mps: Got other mps {} without an L, and there is no preexisting L", tag, otag);
    }

    // Copy the center LC if it's in other
    if(other.isCenter()) {
        if(get_M_bare().dimension(2) != other.get_LC().dimension(0))
            throw except::runtime_error(
                "MpsSite({})::fuse_mps: Got other mps {} with center of wrong dimension. M dims {} are not compatible with its LC dim {}", tag, otag,
                get_M_bare().dimensions(), other.get_LC().dimensions());
        set_LC(other.get_LC(), other.get_truncation_error_LC());
    }
}

template<typename Scalar>
void MpsSite<Scalar>::apply_mpo(const Eigen::Tensor<Scalar, 4> &mpo, bool adjoint) {
    auto t_mpo = tid::tic_token("apply_mpo", tid::level::higher);
    tools::log->trace("MpsSite({})::apply_mpo: Applying mpo (dims {})", get_tag(), mpo.dimensions());
    if constexpr(settings::verbose_apply_mpo) {
        tools::log->trace("L({}):\n{}\n", get_position(), linalg::tensor::to_string(get_L(), 16, 18));
        tools::log->trace("M({}):\n{}\n", get_position(), linalg::tensor::to_string(get_M_bare(), 4, 6));
        tools::log->trace("mpo({}):\n{}\n", get_position(), linalg::tensor::to_string(mpo, 4, 6));
    }
    long mpoDimL = mpo.dimension(0);
    long mpoDimR = mpo.dimension(1);
    //    if(mpoDimL != mpoDimR) throw except::logic_error("MpsSite({})::apply_mpo: Can't apply mpo's with different L/R dims: not implemented yet", get_tag());
    long bcastDimL = get_label() == "B" ? mpoDimR : mpoDimL;

    Eigen::Tensor<Scalar, 1> L_temp = tenx::broadcast(get_L(), {bcastDimL});
    tenx::normalize(L_temp);

    // Sanity check to make sure we are not running into this bug:
    //      https://gitlab.com/libeigen/eigen/-/issues/2351
    if(tenx::hasNaN(L_temp)) throw std::runtime_error("L_temp has nan");
    if(bcastDimL >= 2 and get_L().size() == 1 and L_temp.size() > 1 and std::abs(L_temp[0] - RealScalar{1.0}) < RealScalar(1e-10)) {
        throw std::runtime_error("L_temp is wrong: This may be due to the broadcasting bug: https://gitlab.com/libeigen/eigen/-/issues/2351");
    }
    auto                    &threads = tenx::threads::get();
    Eigen::Tensor<Scalar, 3> M_bare_temp(tenx::array3{spin_dim(), get_chiL() * mpoDimL, get_chiR() * mpoDimR});
    if(adjoint) {
        M_bare_temp.device(*threads->dev) = get_M_bare()
                                                .contract(mpo.conjugate(), tenx::idx({0}, {2}))
                                                .shuffle(tenx::array5{4, 0, 2, 1, 3})
                                                .reshape(tenx::array3{spin_dim(), get_chiL() * mpoDimL, get_chiR() * mpoDimR});
    } else {
        M_bare_temp.device(*threads->dev) = get_M_bare()
                                                .contract(mpo, tenx::idx({0}, {3}))
                                                .shuffle(tenx::array5{4, 0, 2, 1, 3})
                                                .reshape(tenx::array3{spin_dim(), get_chiL() * mpoDimL, get_chiR() * mpoDimR});
    }
    if(isCenter()) {
        Eigen::Tensor<Scalar, 1> LC_temp = tenx::broadcast(get_LC(), {mpoDimR});
        tenx::normalize(LC_temp);
        set_LC(LC_temp);
    }
    set_L(L_temp);
    set_M(M_bare_temp);

    if constexpr(settings::verbose_apply_mpo) {
        tools::log->trace("L({}) after:\n{}\n", get_position(), linalg::tensor::to_string(get_L(), 4, 6));
        tools::log->trace("M({}) after:\n{}\n", get_position(), linalg::tensor::to_string(get_M_bare(), 4, 6));
        tools::log->trace("mpo({}) after:\n{}\n", get_position(), linalg::tensor::to_string(mpo, 4, 6));
    }
}

template<typename Scalar>
void MpsSite<Scalar>::apply_mpo(const Eigen::Tensor<Scalar, 2> &mpo, bool adjoint) {
    auto  t_mpo       = tid::tic_token("apply_mpo", tid::level::higher);
    auto  dim0        = adjoint ? mpo.dimension(0) : mpo.dimension(1);
    auto  M_bare_temp = Eigen::Tensor<Scalar, 3>(dim0, get_chiL(), get_chiR());
    auto &threads     = tenx::threads::get();
    if(adjoint) {
        M_bare_temp.device(*threads->dev) = mpo.conjugate().contract(get_M_bare(), tenx::idx({0}, {0}));
    } else {
        M_bare_temp.device(*threads->dev) = mpo.contract(get_M_bare(), tenx::idx({1}, {0}));
    }
    set_M(M_bare_temp);
}

template<typename Scalar>
void MpsSite<Scalar>::stash_S(const std::pair<Eigen::Tensor<Scalar, 1>, double> &S_and_error, size_t dst) const {
    S_stash = stash<Eigen::Tensor<Scalar, 1>>{S_and_error.first, S_and_error.second, dst};
}

template<typename Scalar>
void MpsSite<Scalar>::stash_C(const std::pair<Eigen::Tensor<Scalar, 1>, double> &C_and_error, size_t dst) const {
    C_stash = stash<Eigen::Tensor<Scalar, 1>>{C_and_error.first, C_and_error.second, dst};
}

template<typename Scalar>
std::optional<stash<Eigen::Tensor<Scalar, 3>>> &MpsSite<Scalar>::get_U_stash() const {
    return U_stash;
}
template<typename Scalar>
std::optional<stash<Eigen::Tensor<Scalar, 1>>> &MpsSite<Scalar>::get_S_stash() const {
    return S_stash;
}
template<typename Scalar>
std::optional<stash<Eigen::Tensor<Scalar, 1>>> &MpsSite<Scalar>::get_C_stash() const {
    return C_stash;
}
template<typename Scalar>
std::optional<stash<Eigen::Tensor<Scalar, 3>>> &MpsSite<Scalar>::get_V_stash() const {
    return V_stash;
}

template<typename Scalar>
void MpsSite<Scalar>::drop_stash() const {
    if constexpr(settings::debug) {
        if(U_stash) tools::log->trace("MpsSite({})::drop_stash: U_stash", get_tag());
        if(S_stash) tools::log->trace("MpsSite({})::drop_stash: S_stash", get_tag());
        if(C_stash) tools::log->trace("MpsSite({})::drop_stash: C_stash", get_tag());
        if(V_stash) tools::log->trace("MpsSite({})::drop_stash: V_stash", get_tag());
    }
    U_stash = std::nullopt;
    S_stash = std::nullopt;
    C_stash = std::nullopt;
    V_stash = std::nullopt;
}

template<typename Scalar>
void MpsSite<Scalar>::drop_stashed_errors() const {
    if constexpr(settings::debug) {
        if(U_stash.has_value() and U_stash->error > 0) tools::log->trace("MpsSite({})::drop_stashed_errors: U_stash error: {:.3e}", get_tag(), U_stash->error);
        if(S_stash.has_value() and S_stash->error > 0) tools::log->trace("MpsSite({})::drop_stashed_errors: S_stash error: {:.3e}", get_tag(), S_stash->error);
        if(C_stash.has_value() and C_stash->error > 0) tools::log->trace("MpsSite({})::drop_stashed_errors: C_stash error: {:.3e}", get_tag(), C_stash->error);
        if(V_stash.has_value() and V_stash->error > 0) tools::log->trace("MpsSite({})::drop_stashed_errors: V_stash error: {:.3e}", get_tag(), V_stash->error);
    }
    if(U_stash and U_stash->error > 0) U_stash->error = -1.0;
    if(S_stash and S_stash->error > 0) S_stash->error = -1.0;
    if(C_stash and C_stash->error > 0) C_stash->error = -1.0;
    if(V_stash and V_stash->error > 0) V_stash->error = -1.0;
}

template<typename Scalar>
void MpsSite<Scalar>::take_stash(const MpsSite &other) {
    if(other.V_stash and other.V_stash->pos_dst == get_position()) {
        /* Left-to-right move.
         * In this case there is a "V" in "other" that should be absorbed into this site from the left.
         *
         *   1 --[New "M"]-- 2       1 --[V]-- 2   1 --[this "M"]-- 2
         *          |            =        |                |
         *          0                 0(dim=1)             0
         *
         *  In addition, if this is an A site we expect an "S" to come along, which didn't become an LC
         *  for the site on the left. Presumably the true LC is on some site further to the right.
         *  Here we simply set it as the new L of this site.
         */
        //        auto t_stash = tid::tic_token("take_stash_V", tid::level::highest);
        if constexpr(settings::verbose_merge)
            tools::log->trace("MpsSite({})::take_stash: Taking V stash from {} | dims {} -> {}", get_tag(), other.get_tag(), dimensions(),
                              other.V_stash->data.dimensions());

        if(get_position() != other.get_position() + 1)
            throw except::logic_error("MpsSite({})::take_stash: Found V stash with destination {} the wrong position {}", get_tag(), other.V_stash->pos_dst,
                                      other.get_tag());

        auto &V = other.V_stash->data;
        if(V.dimension(0) != 1)
            throw except::logic_error("MpsSite({})::take_stash: Failed to take V stash from {}: "
                                      "V has invalid dimensions {}. Dim at idx 0 should be == 1",
                                      get_tag(), other.get_tag(), V.dimensions());
        if(V.dimension(2) != get_chiL())
            throw except::logic_error("MpsSite({})::take_stash: Failed to take V stash from {}: "
                                      "V dimensions {} | M dimensions {} |  Expected V(2) == M(1)",
                                      get_tag(), other.get_tag(), V.dimensions(), dimensions());
        auto VM = tools::common::contraction::contract_mps_mps(V, get_M_bare());
        set_M(VM);
        other.V_stash = std::nullopt; // Stash has been consumed
    }
    if(other.U_stash and other.U_stash->pos_dst == get_position()) {
        /* Right-to-left move.
         * In this case there should be a U and an S in "other".
         * U should be absorbed into this site from the right.
         * S becomes the new LC.
         *
         *    1 --[New "M"]-- 2       1 --[this "M"]-- 2   1 --[U]-- 2
         *           |            =          |                  |
         *           0                       0               0(dim=1)
         *
         *  In addition, if this is a B site we expect an "S" to come along.
         *  Here we simply set it as the new L of this site.
         */

        //        auto t_stash = tid::tic_token("take_stash_U", tid::level::highest);
        if constexpr(settings::verbose_merge)
            tools::log->trace("MpsSite({})::take_stash: Taking U stash from {} | dims {} -> {}", get_tag(), other.get_tag(), dimensions(),
                              other.U_stash->data.dimensions());

        if(get_position() != other.get_position() - 1)
            throw except::logic_error("MpsSite({})::take_stash: Found U stash with destination {} the wrong position {}", get_tag(), other.U_stash->pos_dst,
                                      other.get_tag());

        auto &U = other.U_stash->data;
        if(U.dimension(0) != 1)
            throw except::logic_error("MpsSite({})::take_stash: Failed to take U stash from {}: "
                                      "U has invalid dimensions {}. Dim at idx 0 should be == 1",
                                      get_tag(), other.get_tag(), U.dimensions());

        if(U.dimension(1) != get_chiR())
            throw except::logic_error("MpsSite({})::take_stash: Failed to take V stash from {}: "
                                      "M dimensions {} | U dimensions {} |  Expected M(2) == U(1)",
                                      get_tag(), other.get_tag(), dimensions(), U.dimensions());

        auto MU = tools::common::contraction::contract_mps_mps(get_M_bare(), U);
        set_M(MU);
        other.U_stash = std::nullopt;
    }
    if(other.S_stash and other.S_stash->pos_dst == get_position()) {
        /* We get this stash when
         *      - This is an A-site and AC is further to the right: Then we get both V and S to absorb on this site.
         *      - This is a B-site and AC is further to the left: Then we get both U and S to absorb on this site.
         *      - This is being transformed from AC to a B-site. Then the old LC matrix is inherited as an L matrix.
         *      - We are doing subspace expansion to left or right. Then we get U or V, together with an S to insert into this site.
         */
        //        auto t_stash = tid::tic_token("take_stash_S", tid::level::highest);
        if constexpr(settings::verbose_merge)
            tools::log->trace("MpsSite({})::take_stash: Taking S stash from {} | L dim {} -> {} | err {:.3e} -> {:.3e}", get_tag(), other.get_tag(), L->size(),
                              other.S_stash->data.size(), get_truncation_error(), other.S_stash->error);
        set_L(other.S_stash->data, other.S_stash->error);
        other.S_stash = std::nullopt;
    }
    if(other.C_stash and other.C_stash->pos_dst == get_position()) {
        /* We get this stash when
         *      - This is the right-most last A-site and supposed to become an AC: Then we get S, V and LC to absorb on this site.
         *      - This is being transformed from a B to an AC-site. Then the LC was just created in an SVD.
         *      - We are doing subspace expansion to the left. Then we get U, together with a C to insert into this AC site.
         */
        //        auto t_stash = tid::tic_token("take_stash_C", tid::level::highest);
        if constexpr(settings::verbose_merge) tools::log->trace("MpsSite({})::take_stash: Taking C stash from {}", get_tag(), other.get_tag());
        if(label == "B")
            tools::log->warn("MpsSite({})::take_stash: Taking C_stash to set LC on B-site | LC dim {} -> {} | err {:.3e} -> {:.3e}", get_tag(),
                             isCenter() ? LC->size() : -1l, other.C_stash->data.size(), isCenter() ? get_truncation_error_LC() : 0, other.C_stash->error);
        set_LC(other.C_stash->data, other.C_stash->error);
        other.C_stash = std::nullopt;
    }
}

template<typename Scalar>
void MpsSite<Scalar>::convert_AL_to_A(const Eigen::Tensor<Scalar, 1> &LR) {
    if(label != "AL") throw except::runtime_error("Label error: [{}] | expected [AL]", label);
    if(get_chiR() != LR.dimension(0)) throw except::runtime_error("MpsSite<Scalar>::convert_AL_to_A: chiR {} != LR.dim(0) {}", get_chiR(), LR.dimension(0));
    if(settings::verbose_merge) tools::log->trace("MpsSite({})::convert_AL_to_A: multipliying inverse L", get_tag());
    // Eigen::Tensor<cx64, 3> tmp = get_M_bare().contract(tenx::asDiagonalInversed(LR), tenx::idx({2}, {0}));
    Eigen::Tensor<Scalar, 3> tmp = tenx::asDiagonalInverseContract(LR, get_M_bare(), 2);
    set_label("A");
    set_M(tmp);
}
template<typename Scalar>
void MpsSite<Scalar>::convert_LB_to_B(const Eigen::Tensor<Scalar, 1> &LL) {
    if(label != "LB") throw except::runtime_error("Label error: [{}] | expected [AL]", label);
    if(get_chiL() != LL.dimension(0)) throw except::runtime_error("MpsSite<Scalar>::convert_LB_to_B: chiL {} != LL.dim(0) {}", get_chiL(), LL.dimension(0));
    if(settings::verbose_merge) tools::log->trace("MpsSite({})::convert_LB_to_B: multipliying inverse L", get_tag());
    // Eigen::Tensor<cx64, 3> tmp = tenx::asDiagonalInversed(LL).contract(get_M_bare(), tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2});
    Eigen::Tensor<Scalar, 3> tmp = tenx::asDiagonalInverseContract(LL, get_M_bare(), 1);
    set_label("B");
    set_M(tmp);
}

template<typename Scalar>
std::size_t MpsSite<Scalar>::get_unique_id() const {
    if(unique_id) return unique_id.value();
    unique_id = hash::hash_buffer(get_M().data(), safe_cast<size_t>(get_M().size()));
    unique_id = hash::hash_buffer(get_L().data(), safe_cast<size_t>(get_L().size()), unique_id.value());
    //    if(isCenter()) unique_id = hash::hash_buffer(get_LC().data(), safe_cast<size_t>(get_LC().size()), unique_id.value());
    return unique_id.value();
}
