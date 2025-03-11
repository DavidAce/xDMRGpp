#include "math/tenx.h"
// tenx must appear first
#include "config/debug.h"
#include "config/enums.h"
#include "math/float.h"
#include "math/hash.h"
#include "math/linalg/tensor.h"
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

MpsSite::MpsSite() = default;

template<typename T3, typename T1>
requires is_valid_tensor3<T3> && is_valid_tensor1<T1>
MpsSite::MpsSite(const T3 &M_, const T1 &L_, size_t pos, double error, std::string_view label_) {
    set_position(pos);
    set_label(label_);
    set_M(M_);
    set_L(L_);
    set_truncation_error(error);
}
/* clang-format off */
template MpsSite::MpsSite(const Eigen::Tensor<fp64, 3> &M_, const Eigen::Tensor<fp64, 1> &L_, size_t pos, double error, std::string_view label_);
template MpsSite::MpsSite(const Eigen::Tensor<cx64, 3> &M_, const Eigen::Tensor<cx64, 1> &L_, size_t pos, double error, std::string_view label_);
template MpsSite::MpsSite(const Eigen::Tensor<cx64, 3> &M_, const Eigen::Tensor<fp64, 1> &L_, size_t pos, double error, std::string_view label_);
template MpsSite::MpsSite(const Eigen::Tensor<fp64, 3> &M_, const Eigen::Tensor<cx64, 1> &L_, size_t pos, double error, std::string_view label_);
template MpsSite::MpsSite(const Eigen::TensorMap<Eigen::Tensor<fp64, 3>> &M_, const Eigen::TensorMap<Eigen::Tensor<fp64, 1>> &L_, size_t pos, double error, std::string_view label_);
template MpsSite::MpsSite(const Eigen::TensorMap<Eigen::Tensor<cx64, 3>> &M_, const Eigen::TensorMap<Eigen::Tensor<cx64, 1>> &L_, size_t pos, double error, std::string_view label_);
template MpsSite::MpsSite(const Eigen::TensorMap<Eigen::Tensor<cx64, 3>> &M_, const Eigen::TensorMap<Eigen::Tensor<fp64, 1>> &L_, size_t pos, double error, std::string_view label_);
template MpsSite::MpsSite(const Eigen::TensorMap<Eigen::Tensor<fp64, 3>> &M_, const Eigen::TensorMap<Eigen::Tensor<cx64, 1>> &L_, size_t pos, double error, std::string_view label_);
/* clang-format on */

template<typename T3>
requires is_valid_tensor3<T3>
MpsSite::MpsSite(const T3 &M_, size_t pos, std::string_view label_) {
    set_position(pos);
    set_label(label_);
    set_M(M_);
}
template MpsSite::MpsSite(const Eigen::Tensor<fp64, 3> &M_, size_t pos, std::string_view label_);
template MpsSite::MpsSite(const Eigen::Tensor<cx64, 3> &M_, size_t pos, std::string_view label_);
template MpsSite::MpsSite(const Eigen::TensorMap<Eigen::Tensor<fp64, 3>> &M_, size_t pos, std::string_view label_);
template MpsSite::MpsSite(const Eigen::TensorMap<Eigen::Tensor<cx64, 3>> &M_, size_t pos, std::string_view label_);

MpsSite::MpsSite(const Eigen::Tensor<cx64, 3> &M_, const std::optional<Eigen::Tensor<cx64, 1>> &L_, size_t pos, double error, std::string_view label_) {
    set_position(pos);
    set_label(label_);
    set_M(M_);
    if(L_) set_L(L_.value());
    set_truncation_error(error);
}

MpsSite::MpsSite(const Eigen::Tensor<fp64, 3> &M_, const std::optional<Eigen::Tensor<fp64, 1>> &L_, size_t pos, double error, std::string_view label_) {
    set_position(pos);
    set_label(label_);
    set_M(M_);
    if(L_) set_L(L_.value());
    set_truncation_error(error);
}

// We need to define the destructor and other special functions
// because we enclose data in unique_ptr for this pimpl idiom.
// Otherwise, unique_ptr will forcibly inline its own default deleter.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
MpsSite::~MpsSite()                                   = default; // default dtor
MpsSite::MpsSite(MpsSite &&other) noexcept            = default; // default move ctor
MpsSite &MpsSite::operator=(MpsSite &&other) noexcept = default; // default move assign
MpsSite::MpsSite(const MpsSite &other)                = default;
MpsSite &MpsSite::operator=(const MpsSite &other)     = default;

bool MpsSite::isCenter() const { return LC.has_value(); }

bool MpsSite::has_L() const { return L.has_value(); }
bool MpsSite::has_M() const { return M.has_value(); }
bool MpsSite::has_LC() const { return LC.has_value(); }

Eigen::DSizes<long, 3> MpsSite::dimensions() const { return Eigen::DSizes<long, 3>{spin_dim(), get_chiL(), get_chiR()}; }

bool MpsSite::is_real() const { return tenx::isReal(get_M_bare()) and tenx::isReal(get_L()); }
bool MpsSite::has_nan() const { return tenx::hasNaN(get_M_bare()) or tenx::hasNaN(get_L()); }
bool MpsSite::is_normalized(double prec) const {
    auto t_dbg = tid::tic_token("is_normalized", tid::level::highest);
    if(isCenter() or get_label() == "AC") {
        cx64 norm = tools::common::contraction::contract_mps_norm(get_M());
        return std::abs(norm - 1.0) <= prec;
    }
    if(get_label() == "A") {
        auto id = tools::common::contraction::contract_mps_partial<std::array{0l, 1l}>(get_M_bare());
        return tenx::isIdentity(id, prec);
    }
    if(get_label() == "B") {
        auto id    = tools::common::contraction::contract_mps_partial<std::array{0l, 2l}>(get_M_bare());
        auto is_id = tenx::isIdentity(id, prec);
        if(not is_id) tools::log->error("B^dagger B:\n{}\n", linalg::tensor::to_string(id, 16));
        return is_id;
    }
    throw except::runtime_error("MpsSite::is_identity: unexpected label: {}", get_label());
}

void MpsSite::assert_validity() const {
    if(LC and get_label() != "AC") throw except::runtime_error("MpsSite::assert_validity: Found LC at pos {} with label {}", get_position(), get_label());
    if(not LC and get_label() == "AC") throw except::runtime_error("MpsSite::assert_validity: Missing LC at pos {} with label {}", get_position(), get_label());
    if(has_nan()) throw except::runtime_error("MpsSite::assert_validity(): MPS (M or L) at position {} has NaN's", get_position());
    assert_dimensions();
}

void MpsSite::assert_dimensions() const {
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

void MpsSite::assert_normalized(double prec) const {
    if(not is_normalized(prec))
        throw except::runtime_error("MpsSite::assert_normalized({0:.2e}): {1}^dagger {1} is not normalized at pos {2}", prec, get_label(), get_position());
}

const Eigen::Tensor<cx64, 3> &MpsSite::get_M_bare() const {
    if(M) {
        if(M.value().size() == 0) throw except::runtime_error("MpsSite::get_M_bare(): M has size 0 at position {}", get_position());
        return M.value();
    } else
        throw except::runtime_error("MpsSite::get_M_bare(): M has not been set at position {}", get_position());
}
const Eigen::Tensor<cx64, 3> &MpsSite::get_M() const {
    //    auto t_get = tid::tic_scope("get_M", tid::level::highest);
    if(isCenter()) {
        if(LC.value().dimension(0) != get_M_bare().dimension(2))
            throw except::runtime_error("MpsSite::get_M(): M and LC dim mismatch: {} != {} at position {}", get_M_bare().dimension(2), LC.value().dimension(0),
                                        get_position());
        if(MC) {
            if(MC.value().size() == 0) throw except::runtime_error("MpsSite::get_M(): MC has size 0 at position {}", get_position());
            return MC.value();
        } else {
            MC = Eigen::Tensor<cx64, 3>(spin_dim(), get_chiL(), get_chiR());
            tools::common::contraction::contract_mps_bnd(MC.value(), get_M_bare(), get_LC());
            if(MC->size() == 0) throw except::runtime_error("MpsSite::get_M(): built MC with size 0 at position {}", get_position());
            return MC.value();
        }
    } else
        return get_M_bare();
}

const Eigen::Tensor<cx64, 1> &MpsSite::get_L() const {
    if(L) {
        if constexpr(settings::debug) {
            if(L->size() == 0) throw except::runtime_error("MpsSite::get_L(): L has size 0 at position {} | label {}", get_position(), get_label());
            if(get_label() != "B" and L->size() != get_chiL())
                throw except::runtime_error("MpsSite::get_L(): L.size() {} != chiL {} at position {} | M dims {} | label {}", L->size(), get_chiL(),
                                            get_position(), dimensions(), get_label());
            if(get_label() == "B" and L->size() != get_chiR())
                throw except::runtime_error("MpsSite::get_L(): L.size() {} != chiR {} at position {} | M dims {} | label {}", L->size(), get_chiR(),
                                            get_position(), dimensions(), get_label());
        }
        return L.value();
    } else
        throw except::runtime_error("MpsSite::get_L(): L has not been set at position {}", get_position());
}
const Eigen::Tensor<cx64, 1> &MpsSite::get_LC() const {
    if(isCenter()) {
        if(LC->dimension(0) != get_M_bare().dimension(2))
            throw except::runtime_error("MpsSite::get_LC(): M dimensions {} are incompatible with LC dimensions {} at position {}", get_M_bare().dimensions(),
                                        LC.value().dimensions(), get_position());
        if(LC.value().size() == 0) throw except::runtime_error("MpsSite::get_LC(): LC has size 0 at position {}", get_position());
        return LC.value();
    } else
        throw except::runtime_error("MpsSite::get_LC(): Site at position {} is not a center", get_position());
}

Eigen::Tensor<cx64, 3> &MpsSite::get_M_bare() { return const_cast<Eigen::Tensor<cx64, 3> &>(std::as_const(*this).get_M_bare()); }
Eigen::Tensor<cx64, 3> &MpsSite::get_M() { return const_cast<Eigen::Tensor<cx64, 3> &>(std::as_const(*this).get_M()); }
Eigen::Tensor<cx64, 1> &MpsSite::get_L() { return const_cast<Eigen::Tensor<cx64, 1> &>(std::as_const(*this).get_L()); }
Eigen::Tensor<cx64, 1> &MpsSite::get_LC() { return const_cast<Eigen::Tensor<cx64, 1> &>(std::as_const(*this).get_LC()); }

template<typename T>
Eigen::Tensor<T, 3> MpsSite::get_M_bare_as() const {
    return tenx::asScalarType<T>(get_M_bare());
    // if constexpr(std::is_same_v<T, fp32> or std::is_same_v<T, fp64>) {
    //     assert(is_real());
    //     return get_M_bare().real().template cast<T>();
    // } else if constexpr(std::is_same_v<T, cx32>) {
    //     return get_M_bare().template cast<T>();
    // } else if constexpr(std::is_same_v<T, cx64>) {
    //     return get_M_bare();
    // }
    // throw except::runtime_error("get_M_bare_as(): invalid type <{}>", sfinae::type_name<T>());
}
template Eigen::Tensor<fp32, 3> MpsSite::get_M_bare_as() const;
template Eigen::Tensor<fp64, 3> MpsSite::get_M_bare_as() const;
template Eigen::Tensor<cx32, 3> MpsSite::get_M_bare_as() const;
template Eigen::Tensor<cx64, 3> MpsSite::get_M_bare_as() const;

template<typename T>
[[nodiscard]] Eigen::Tensor<T, 3> MpsSite::get_M_as() const {
    if constexpr(std::is_same_v<T, fp32> or std::is_same_v<T, fp64>) {
        assert(is_real());
        return get_M().real().template cast<T>();
    } else if constexpr(std::is_same_v<T, cx32>) {
        return get_M().template cast<T>();

    } else if constexpr(std::is_same_v<T, cx64>) {
        return get_M();
    }
    throw except::runtime_error("get_M_as(): invalid type <{}>", sfinae::type_name<T>());
}
template Eigen::Tensor<fp32, 3> MpsSite::get_M_as() const;
template Eigen::Tensor<fp64, 3> MpsSite::get_M_as() const;
template Eigen::Tensor<cx32, 3> MpsSite::get_M_as() const;
template Eigen::Tensor<cx64, 3> MpsSite::get_M_as() const;

template<typename T>
[[nodiscard]] Eigen::Tensor<T, 1> MpsSite::get_L_as() const {
    if constexpr(std::is_same_v<T, fp32> or std::is_same_v<T, fp64>) {
        assert(is_real());
        return get_L().real().template cast<T>();
    }
    if constexpr(std::is_same_v<T, cx32>) {
        Eigen::Tensor<T, 1> res = get_L().template cast<T>();
        // auto               &L = get_L();
        // Eigen::Tensor<T, 1> res(L.dimensions());
        // res.real() = L.real().template cast<T>();
        // res.imag() = L.imag().template cast<T>();
        return res;
    }
    if constexpr(std::is_same_v<T, cx64>) { return get_L(); }
    throw except::runtime_error("get_L_as(): invalid type <{}>", sfinae::type_name<T>());
}
template Eigen::Tensor<fp32, 1> MpsSite::get_L_as() const;
template Eigen::Tensor<fp64, 1> MpsSite::get_L_as() const;
template Eigen::Tensor<cx32, 1> MpsSite::get_L_as() const;
template Eigen::Tensor<cx64, 1> MpsSite::get_L_as() const;

template<typename T>
[[nodiscard]] Eigen::Tensor<T, 1> MpsSite::get_LC_as() const {
    if constexpr(std::is_same_v<T, fp32> or std::is_same_v<T, fp64>) {
        assert(is_real());
        return get_LC().real().template cast<T>();
    }
    if constexpr(std::is_same_v<T, cx32>) {
        Eigen::Tensor<T, 1> res = get_LC().template cast<T>();
        // auto               &get = get_LC();
        // Eigen::Tensor<T, 1> res(get.dimensions());
        // res.real() = get.real().template cast<T>();
        // res.imag() = get.imag().template cast<T>();
        return res;
    }
    if constexpr(std::is_same_v<T, cx64>) { return get_L(); }
    throw except::runtime_error("get_LC_as(): invalid type <{}>", sfinae::type_name<T>());
}
template Eigen::Tensor<fp32, 1> MpsSite::get_LC_as() const;
template Eigen::Tensor<fp64, 1> MpsSite::get_LC_as() const;
template Eigen::Tensor<cx32, 1> MpsSite::get_LC_as() const;
template Eigen::Tensor<cx64, 1> MpsSite::get_LC_as() const;

double           MpsSite::get_truncation_error() const { return truncation_error; }
double           MpsSite::get_truncation_error_LC() const { return truncation_error_LC; }
std::string_view MpsSite::get_label() const {
    if(label.empty()) throw except::runtime_error("No label found at position {}", get_position());
    return label;
}
std::string MpsSite::get_tag() const { return fmt::format("{}[{}]", get_label(), get_position()); }

std::tuple<long, long, long> MpsSite::get_dims() const { return {spin_dim(), get_chiL(), get_chiR()}; }
long                         MpsSite::spin_dim() const { return get_M_bare().dimension(0); }
long                         MpsSite::get_chiL() const { return get_M_bare().dimension(1); }
long                         MpsSite::get_chiR() const { return get_M_bare().dimension(2); }

void MpsSite::set_position(const size_t position_) {
    position = position_;
    MC.reset();
}

template<typename T>
T MpsSite::get_position() const {
    if(position) {
        return safe_cast<T>(position.value());
    } else {
        throw std::runtime_error("MpsSite::get_position(): Position hasn't been set on mps site.");
    }
}

template size_t             MpsSite::get_position<size_t>() const;
template long               MpsSite::get_position<long>() const;
template int                MpsSite::get_position<int>() const;
template unsigned long long MpsSite::get_position<unsigned long long>() const; // hsize_t from hdf5

template<typename T>
[[nodiscard]] bool MpsSite::is_at_position(T pos) const {
    return std::cmp_equal(get_position(), pos);
}

template bool MpsSite::is_at_position(size_t pos) const;
template bool MpsSite::is_at_position(unsigned pos) const;
template bool MpsSite::is_at_position(long pos) const;
template bool MpsSite::is_at_position(int pos) const;
template bool MpsSite::is_at_position(unsigned long long pos) const; // hsize_t from hdf5

void MpsSite::set_mps(const Eigen::Tensor<cx64, 3> &M_, const Eigen::Tensor<cx64, 1> &L_, double error, std::string_view label_) {
    // M has to be a "bare" matrix, i.e. not an MC which would include LC.
    set_label(label_);
    set_M(M_);
    set_L(L_);
    set_truncation_error(error);
}
void MpsSite::set_mps(const Eigen::Tensor<cx64, 3> &M_, const Eigen::Tensor<fp64, 1> &L_, double error, std::string_view label_) {
    set_mps(M_, Eigen::Tensor<cx64, 1>(L_.cast<cx64>()), error, label_);
}

template<typename T3>
requires is_valid_tensor3<T3>
void MpsSite::set_M(const T3 &M_) {
    // M has to be a "bare" matrix, i.e. not an MC which would include LC.
    if(position) {
        M = M_.template cast<cx64>();
        MC.reset();
        unique_id = std::nullopt;
    } else
        throw std::runtime_error("MpsSite::set_M(const Eigen::Tensor<cx64, 3> &): Can't set M: Position hasn't been set yet");
}
template void MpsSite::set_M(const Eigen::Tensor<fp32, 3> &M_);
template void MpsSite::set_M(const Eigen::Tensor<fp64, 3> &M_);
template void MpsSite::set_M(const Eigen::Tensor<cx32, 3> &M_);
template void MpsSite::set_M(const Eigen::Tensor<cx64, 3> &M_);
template void MpsSite::set_M(const Eigen::TensorMap<Eigen::Tensor<fp32, 3>> &M_);
template void MpsSite::set_M(const Eigen::TensorMap<Eigen::Tensor<fp64, 3>> &M_);
template void MpsSite::set_M(const Eigen::TensorMap<Eigen::Tensor<cx32, 3>> &M_);
template void MpsSite::set_M(const Eigen::TensorMap<Eigen::Tensor<cx64, 3>> &M_);

template<typename T1>
requires is_valid_tensor1<T1>
void MpsSite::set_L(const T1 &L_, double error /* Negative is ignored */) {
    if constexpr(settings::debug) {
        using RealScalar = typename Eigen::NumTraits<typename T1::Scalar>::Real;
        auto norm        = tenx::VectorMap(L_).norm();
        if(std::abs(norm - 1.0) > static_cast<RealScalar>(1e-8)) tools::log->warn("MpsSite::set_L(): Norm of L is too far from unity: {:.16f}", norm);
    }

    if(position) {
        set_truncation_error(error);
        L         = L_.template cast<cx64>();
        unique_id = std::nullopt;
    } else
        throw std::runtime_error("Can't set L: Position hasn't been set yet");
}
template void MpsSite::set_L(const Eigen::Tensor<fp64, 1> &LC_, double error);
template void MpsSite::set_L(const Eigen::Tensor<cx64, 1> &LC_, double error);
template void MpsSite::set_L(const Eigen::TensorMap<Eigen::Tensor<fp64, 1>> &LC_, double error);
template void MpsSite::set_L(const Eigen::TensorMap<Eigen::Tensor<cx64, 1>> &LC_, double error);

// void MpsSite::set_L(const Eigen::Tensor<fp64, 1> &L_, double error) { set_L(Eigen::Tensor<cx64, 1>(L_.cast<cx64>()), error); }
void MpsSite::set_L(const std::pair<Eigen::Tensor<cx64, 1>, double> &L_and_error) { set_L(L_and_error.first, L_and_error.second); }
void MpsSite::set_L(const std::pair<Eigen::Tensor<fp64, 1>, double> &L_and_error) { set_L(L_and_error.first, L_and_error.second); }

template<typename T1>
requires is_valid_tensor1<T1>
void MpsSite::set_LC(const T1 &LC_, double error /* Negative is ignored */) {
    if constexpr(settings::debug) {
        using RealScalar = typename Eigen::NumTraits<typename T1::Scalar>::Real;
        auto norm        = tenx::VectorMap(LC_).norm();
        if(std::abs(norm - 1) > static_cast<RealScalar>(1e-8)) tools::log->warn("MpsSite::set_LC(): Norm of LC is too far from unity: {:.16f}", norm);
        //            throw except::runtime_error(MpsSite::set_LC(): Can't set L: Norm of LC is too far from unity: {:.16f}", norm);
        if(position and position.value() == 0 and get_label() == "B") {
            throw except::logic_error("Only an A-site can become an AC site (not really true though");
        }
    }
    if(position) {
        set_label("AC");
        set_truncation_error_LC(error);
        LC = LC_.template cast<cx64>();
        MC.reset();
        unique_id = std::nullopt;
    } else
        throw std::runtime_error("Can't set LC: Position hasn't been set yet");
}
template void MpsSite::set_LC(const Eigen::Tensor<fp32, 1> &LC_, double error);
template void MpsSite::set_LC(const Eigen::Tensor<fp64, 1> &LC_, double error);
template void MpsSite::set_LC(const Eigen::Tensor<cx32, 1> &LC_, double error);
template void MpsSite::set_LC(const Eigen::Tensor<cx64, 1> &LC_, double error);
template void MpsSite::set_LC(const Eigen::TensorMap<Eigen::Tensor<fp32, 1>> &LC_, double error);
template void MpsSite::set_LC(const Eigen::TensorMap<Eigen::Tensor<fp64, 1>> &LC_, double error);
template void MpsSite::set_LC(const Eigen::TensorMap<Eigen::Tensor<cx32, 1>> &LC_, double error);
template void MpsSite::set_LC(const Eigen::TensorMap<Eigen::Tensor<cx64, 1>> &LC_, double error);

// void MpsSite::set_LC(const Eigen::Tensor<fp64, 1> &LC_, double error) { set_LC(Eigen::Tensor<cx64, 1>(LC_.cast<cx64>()), error); }
void MpsSite::set_LC(const std::pair<Eigen::Tensor<cx64, 1>, double> &LC_and_error) { set_LC(LC_and_error.first, LC_and_error.second); }
void MpsSite::set_LC(const std::pair<Eigen::Tensor<fp64, 1>, double> &LC_and_error) { set_LC(LC_and_error.first, LC_and_error.second); }

void MpsSite::set_truncation_error(double error /* Negative is ignored */) {
    if(error >= 0.0) {
        //        tools::log->warn("Setting truncation error on site {}: {:8.5e}", get_tag(), error);
        truncation_error = error;
    }
}

void MpsSite::set_truncation_error_LC(double error /* Negative is ignored */) {
    if(error >= 0.0) {
        //        tools::log->warn("Setting truncation error on site {}: {:8.5e} (LC)", get_tag(), error);
        truncation_error_LC = error;
    }
}

void MpsSite::set_label(std::string_view label_) {
    // If we are flipping the kind of site from B to non B (or vice-versa), then the L matrix
    // should be removed since it changes side.
    if(not label.empty() and not label_.empty() and label.front() != label_.front()) unset_L();
    label = label_;
}
void MpsSite::unset_LC() {
    LC        = std::nullopt;
    MC        = std::nullopt;
    unique_id = std::nullopt;
    if(label == "AC") label = "A";
}

void MpsSite::unset_L() {
    L         = std::nullopt;
    unique_id = std::nullopt;
}

void MpsSite::unset_truncation_error() { truncation_error = -1.0; }

void MpsSite::unset_truncation_error_LC() { truncation_error_LC = -1.0; }

void MpsSite::fuse_mps(const MpsSite &other) {
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
        auto one = Eigen::Tensor<cx64, 1>(1).setConstant(cx64(1.0, 0.0));
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

void MpsSite::apply_mpo(const Eigen::Tensor<cx64, 4> &mpo, bool adjoint) {
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

    Eigen::Tensor<cx64, 1> L_temp = tenx::broadcast(get_L(), {bcastDimL});
    tenx::normalize(L_temp);

    // Sanity check to make sure we are not running into this bug:
    //      https://gitlab.com/libeigen/eigen/-/issues/2351
    if(tenx::hasNaN(L_temp)) throw std::runtime_error("L_temp has nan");
    if(bcastDimL >= 2 and get_L().size() == 1 and L_temp.size() > 1 and std::abs(L_temp[0] - 1.0) < 1e-10) {
        for(long i = 0; i < L_temp.size(); i++) std::printf("(%.16f, %.16f)\n", L_temp[i].real(), L_temp[i].imag());
        throw std::runtime_error("L_temp is wrong: This may be due to the broadcasting bug: https://gitlab.com/libeigen/eigen/-/issues/2351");
    }
    auto                  &threads = tenx::threads::get();
    Eigen::Tensor<cx64, 3> M_bare_temp(tenx::array3{spin_dim(), get_chiL() * mpoDimL, get_chiR() * mpoDimR});
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
        Eigen::Tensor<cx64, 1> LC_temp = tenx::broadcast(get_LC(), {mpoDimR});
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

void MpsSite::apply_mpo(const Eigen::Tensor<cx64, 2> &mpo, bool adjoint) {
    auto  t_mpo       = tid::tic_token("apply_mpo", tid::level::higher);
    auto  dim0        = adjoint ? mpo.dimension(0) : mpo.dimension(1);
    auto  M_bare_temp = Eigen::Tensor<cx64, 3>(dim0, get_chiL(), get_chiR());
    auto &threads     = tenx::threads::get();
    if(adjoint) {
        M_bare_temp.device(*threads->dev) = mpo.conjugate().contract(get_M_bare(), tenx::idx({0}, {0}));
    } else {
        M_bare_temp.device(*threads->dev) = mpo.contract(get_M_bare(), tenx::idx({1}, {0}));
    }
    set_M(M_bare_temp);
}

template<typename T3>
requires is_valid_tensor3<T3>
void MpsSite::stash_U(const T3 &U, size_t dst) const {
    U_stash = stash<Eigen::Tensor<cx64, 3>>{.data = U.template cast<cx64>(), .error = 0, .pos_dst = dst};
}
template void MpsSite::stash_U(const Eigen::Tensor<cx64, 3> &U, size_t dst) const;
template void MpsSite::stash_U(const Eigen::Tensor<fp64, 3> &U, size_t dst) const;
template void MpsSite::stash_U(const Eigen::TensorMap<Eigen::Tensor<cx64, 3>> &U, size_t dst) const;
template void MpsSite::stash_U(const Eigen::TensorMap<Eigen::Tensor<fp64, 3>> &U, size_t dst) const;

template<typename T1>
requires is_valid_tensor1<T1>
void MpsSite::stash_S(const T1 &S, double error, size_t dst) const {
    S_stash = stash<Eigen::Tensor<cx64, 1>>{.data = S.template cast<cx64>(), .error = error, .pos_dst = dst};
}
template void MpsSite::stash_S(const Eigen::Tensor<cx64, 1> &S, double error, size_t dst) const;
template void MpsSite::stash_S(const Eigen::Tensor<fp64, 1> &S, double error, size_t dst) const;
template void MpsSite::stash_S(const Eigen::TensorMap<Eigen::Tensor<cx64, 1>> &S, double error, size_t dst) const;
template void MpsSite::stash_S(const Eigen::TensorMap<Eigen::Tensor<fp64, 1>> &S, double error, size_t dst) const;

template<typename T1>
requires is_valid_tensor1<T1>
void MpsSite::stash_C(const T1 &C, double error, size_t dst) const {
    C_stash = stash<Eigen::Tensor<cx64, 1>>{.data = C.template cast<cx64>(), .error = error, .pos_dst = dst};
}
template void MpsSite::stash_C(const Eigen::Tensor<cx64, 1> &S, double error, size_t dst) const;
template void MpsSite::stash_C(const Eigen::Tensor<fp64, 1> &S, double error, size_t dst) const;
template void MpsSite::stash_C(const Eigen::TensorMap<Eigen::Tensor<cx64, 1>> &S, double error, size_t dst) const;
template void MpsSite::stash_C(const Eigen::TensorMap<Eigen::Tensor<fp64, 1>> &S, double error, size_t dst) const;

void MpsSite::stash_S(const std::pair<Eigen::Tensor<cx64, 1>, double> &S_and_error, size_t dst) const {
    S_stash = stash<Eigen::Tensor<cx64, 1>>{S_and_error.first, S_and_error.second, dst};
}
void MpsSite::stash_C(const std::pair<Eigen::Tensor<cx64, 1>, double> &C_and_error, size_t dst) const {
    C_stash = stash<Eigen::Tensor<cx64, 1>>{C_and_error.first, C_and_error.second, dst};
}

template<typename T3>
requires is_valid_tensor3<T3>
void MpsSite::stash_V(const T3 &V, size_t dst) const {
    V_stash = stash<Eigen::Tensor<cx64, 3>>{.data = V.template cast<cx64>(), .error = 0, .pos_dst = dst};
}
template void MpsSite::stash_V(const Eigen::Tensor<cx64, 3> &V, size_t dst) const;
template void MpsSite::stash_V(const Eigen::Tensor<fp64, 3> &V, size_t dst) const;
template void MpsSite::stash_V(const Eigen::TensorMap<Eigen::Tensor<cx64, 3>> &V, size_t dst) const;
template void MpsSite::stash_V(const Eigen::TensorMap<Eigen::Tensor<fp64, 3>> &V, size_t dst) const;

std::optional<stash<Eigen::Tensor<cx64, 3>>> &MpsSite::get_U_stash() const { return U_stash; }
std::optional<stash<Eigen::Tensor<cx64, 1>>> &MpsSite::get_S_stash() const { return S_stash; }
std::optional<stash<Eigen::Tensor<cx64, 1>>> &MpsSite::get_C_stash() const { return C_stash; }
std::optional<stash<Eigen::Tensor<cx64, 3>>> &MpsSite::get_V_stash() const { return V_stash; }

void MpsSite::drop_stash() const {
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

void MpsSite::drop_stashed_errors() const {
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

void MpsSite::take_stash(const MpsSite &other) {
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
        auto VM = tools::common::contraction::contract_mps_mps_temp(V, get_M_bare());
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

        auto MU = tools::common::contraction::contract_mps_mps_temp(get_M_bare(), U);
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

void MpsSite::convert_AL_to_A(const Eigen::Tensor<cx64, 1> &LR) {
    if(label != "AL") throw except::runtime_error("Label error: [{}] | expected [AL]", label);
    if(get_chiR() != LR.dimension(0)) throw except::runtime_error("MpsSite::convert_AL_to_A: chiR {} != LR.dim(0) {}", get_chiR(), LR.dimension(0));
    if(settings::verbose_merge) tools::log->trace("MpsSite({})::convert_AL_to_A: multipliying inverse L", get_tag());
    Eigen::Tensor<cx64, 3> tmp = get_M_bare().contract(tenx::asDiagonalInversed(LR), tenx::idx({2}, {0}));
    set_label("A");
    set_M(tmp);
}
void MpsSite::convert_LB_to_B(const Eigen::Tensor<cx64, 1> &LL) {
    if(label != "LB") throw except::runtime_error("Label error: [{}] | expected [AL]", label);
    if(get_chiL() != LL.dimension(0)) throw except::runtime_error("MpsSite::convert_LB_to_B: chiL {} != LL.dim(0) {}", get_chiL(), LL.dimension(0));
    if(settings::verbose_merge) tools::log->trace("MpsSite({})::convert_LB_to_B: multipliying inverse L", get_tag());
    Eigen::Tensor<cx64, 3> tmp = tenx::asDiagonalInversed(LL).contract(get_M_bare(), tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2});
    set_label("B");
    set_M(tmp);
}

std::size_t MpsSite::get_unique_id() const {
    if(unique_id) return unique_id.value();
    unique_id = hash::hash_buffer(get_M().data(), safe_cast<size_t>(get_M().size()));
    unique_id = hash::hash_buffer(get_L().data(), safe_cast<size_t>(get_L().size()), unique_id.value());
    //    if(isCenter()) unique_id = hash::hash_buffer(get_LC().data(), safe_cast<size_t>(get_LC().size()), unique_id.value());
    return unique_id.value();
}
