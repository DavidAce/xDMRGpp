

#include "qm/gate.h"
#include "config/enums.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "io/fmt_custom.h"
#include "io/fmt_f128_t.h"
#include "math/float.h"
#include "math/num.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tools/common/log.h"
#include <fmt/ranges.h>
#include <set>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>

namespace settings {
    inline constexpr bool debug_gates   = false;
    inline constexpr bool verbose_gates = false;
}

template<typename T>
std::vector<T> subset(const std::vector<T> &vec, size_t idx_start, size_t num) {
    if(idx_start + num > vec.size()) throw except::range_error("Vector subset start {} num {} out of range for vector of size {}", idx_start, num, vec.size());
    auto vec_bgn = vec.begin() + safe_cast<long>(idx_start);
    auto vec_end = vec_bgn + safe_cast<long>(num);
    return std::vector<T>(vec_bgn, vec_end);
}

template<auto N, typename T, auto M>
std::array<T, N> subset(const std::array<T, M> &arr, size_t idx_start) {
    if(idx_start + N > arr.size()) throw except::range_error("Vector subset start {} num {} out of range for vector of size {}", idx_start, N, arr.size());
    auto             vec_bgn = arr.begin() + idx_start;
    auto             vec_end = vec_bgn + N;
    std::array<T, N> res{};
    std::copy(vec_bgn, vec_end, res.begin());
    return res;
}

template<typename T>
T prod(const std::vector<T> &vec, size_t idx_start, size_t num, T seed = 1) {
    auto sub = subset(vec, idx_start, num);
    return std::accumulate(sub.begin(), sub.end(), std::multiplies(seed));
}

template<auto N, typename T, auto M>
T prod(const std::array<T, M> &arr, size_t idx_start, T seed = 1) {
    auto sub = subset<N>(arr, idx_start);
    T    acc = seed;
    for(auto &s : sub) acc *= s;
    return acc;
}

template<typename T>
std::vector<T> concat(const std::vector<T> &v1, const std::vector<T> &v2) {
    auto res = v1;
    res.insert(res.end(), v2.begin(), v2.end());
    return res;
}

template<typename T, auto N, auto M>
constexpr auto concat(const std::array<T, N> &a1, const std::array<T, M> &a2) {
    std::array<T, N + M> result{};
    std::copy(a1.cbegin(), a1.cend(), result.begin());
    std::copy(a2.cbegin(), a2.cend(), result.begin() + N);
    return result;
}

template<typename T, auto N>
auto repeat(const std::array<T, N> &a) {
    return concat(a, a);
}

auto group(const std::vector<long> &dim, const std::vector<size_t> &pattern) {
    std::vector<long> res(pattern.size());
    size_t            dim_offset = 0;
    for(size_t i = 0; i < pattern.size(); i++) {
        long product = 1;
        for(size_t j = 0; j < pattern[i]; j++) product *= dim[dim_offset + j];
        res[i] = product;
        dim_offset += pattern[i];
    }
    return res;
}

template<auto N, auto M>
constexpr auto group(const std::array<long, N> &dim, const std::array<size_t, M> &pattern) {
    /*! Group tensor indices together.
     * */
    std::array<long, M> res{};
    size_t              dim_offset = 0;
    for(size_t i = 0; i < M; i++) {
        long product = 1;
        for(size_t j = 0; j < pattern[i]; j++) product *= dim[dim_offset + j];
        res[i] = product;
        dim_offset += pattern[i];
    }
    return res;
}

template<typename T1, typename T2>
void erase(std::vector<T1> &vec, T2 val) {
    auto it = vec.begin();
    while(it != vec.end()) {
        if(*it == static_cast<T1>(val))
            it = vec.erase(it);
        else
            ++it;
    }
}

Eigen::Tensor<cx64, 2> contract_a(const Eigen::Tensor<cx64, 2> &m, const Eigen::Tensor<cx64, 2> &ud, const std::array<long, 4> &shp_mid4,
                                  const std::array<long, 4> &shp_udn4, const std::array<long, 6> &shf6, const tenx::idxlistpair<1> &idx1,
                                  const tenx::idxlistpair<2> &idx2, const std::array<long, 2> &dim2) {
    auto  res                 = Eigen::Tensor<cx64, 2>(dim2);
    auto &threads             = tenx::threads::get();
    res.device(*threads->dev) = ud.shuffle(tenx::array2{1, 0})
                                    .conjugate()
                                    .reshape(shp_udn4)
                                    .contract(m.reshape(shp_mid4), idx1)
                                    .contract(ud.reshape(shp_udn4), idx2)
                                    .shuffle(shf6)
                                    .reshape(dim2);
    return res;
}

Eigen::Tensor<cx64, 2> contract_b(const Eigen::Tensor<cx64, 2> &m, const Eigen::Tensor<cx64, 2> &ud, const std::array<long, 2> &shp_udn2,
                                  const std::array<long, 4> &shp_udn4, const tenx::idxlistpair<1> &idx1, const tenx::idxlistpair<2> &idx2) {
    auto &threads = tenx::threads::get();
    auto  res     = Eigen::Tensor<cx64, 2>(shp_udn2);
    res.device(*threads->dev) =
        ud.shuffle(tenx::array2{1, 0}).conjugate().reshape(shp_udn4).contract(m, idx1).contract(ud.reshape(shp_udn4), idx2).reshape(shp_udn2);
    return res;
}

Eigen::Tensor<cx64, 2> contract_c(const Eigen::Tensor<cx64, 2> &m, const Eigen::Tensor<cx64, 2> &ud, const std::array<long, 6> &shp_mid6,
                                  const tenx::idxlistpair<1> &idx_dn, const tenx::idxlistpair<1> &idx_up, const std::array<long, 6> &shf6,
                                  const std::array<long, 2> &dim2) {
    auto &threads = tenx::threads::get();
    auto  res     = Eigen::Tensor<cx64, 2>(dim2);
    res.device(*threads->dev) =
        ud.shuffle(tenx::array2{1, 0}).conjugate().contract(m.reshape(shp_mid6), idx_dn).contract(ud, idx_up).shuffle(shf6).reshape(dim2);
    return res;
}

Eigen::Tensor<cx64, 2> contract_d(const Eigen::Tensor<cx64, 2> &m, const Eigen::Tensor<cx64, 2> &ud, const std::array<long, 4> &shp_mid4,
                                  const tenx::idxlistpair<1> &idx_dn, const tenx::idxlistpair<1> &idx_up, const std::array<long, 4> &shf4,
                                  const std::array<long, 2> &dim2) {
    auto &threads = tenx::threads::get();
    auto  res     = Eigen::Tensor<cx64, 2>(dim2);
    res.device(*threads->dev) =
        ud.shuffle(tenx::array2{1, 0}).conjugate().contract(m.reshape(shp_mid4), idx_dn).contract(ud, idx_up).shuffle(shf4).reshape(dim2);
    return res;
}

template<typename scalar_t, typename alpha_t>
Eigen::Tensor<scalar_t, 2> qm::Gate::exp_internal(const Eigen::Tensor<scalar_t, 2> &op_, alpha_t alpha) const {
    /* Note for flbit:
     *  Let h = op(i,i), i.e. h are the diagonal entries in op
     *  Let alpha = -i * delta be purely imaginary (since delta is real). So delta = imag(-alpha)
     *  Now notice that
     *       exp( -i * delta * h )
     *  can become imprecise when delta is large, since delta and h are real and h ~O(1)
     *
     *  Instead, in the flbit case, we can exploit that h is real to compute
     *       exp(-i * mod(delta * real(h), 2*pi))
     *  which is equivalent but with a much smaller exponent.
     *
     *  Remember to do the modulo separately on each diagonal entry h!
     */
    //    static_assert(std::is_same_v<scalar_t, cx128>);
    //    static_assert(std::is_same_v<alpha_t, cx128>);
    //    if(!std::is_same_v<scalar_t, cx128>) { throw except::runtime_error("Expected scalar_t == cx128: got {}", sfinae::type_name<scalar_t>()); }
    //    if(!std::is_same_v<alpha_t, cx128>) { throw except::runtime_error("Expected alpha_t == cx128: got {}", sfinae::type_name<alpha_t>()); }

    auto           op_map                       = tenx::MatrixMap(op_);
    constexpr bool scalar_t_is_float128         = std::is_same_v<scalar_t, __float128>;
    constexpr bool scalar_t_is_complex_float128 = sfinae::is_std_complex_v<scalar_t> and std::is_same_v<typename scalar_t::value_type, __float128>;
    bool           exp_diagonal                 = tenx::isDiagonal(op);

    if(exp_diagonal and op_map.imag().isZero() and std::real(alpha) == 0) {
        using namespace std::complex_literals;
        auto diag = op_map.diagonal()
                        .unaryViewExpr([&alpha](const scalar_t &h) -> scalar_t {
                            // 6.28318530717958623199592693708837032318115234375
                            // Now the same with mpfr
                            //                            scalar_t exp_ialpha_h_mph;
                            //                            if constexpr(std::is_arithmetic_v<scalar_t>) {
                            //                                mpfr::mpreal::set_default_prec(256);
                            //                                mpfr::mpreal::set_default_rnd(MPFR_RNDN);
                            //                                auto alpha_mph        = mpfr::mpreal(-alpha.imag(), 256);
                            //                                auto h_mph            = mpfr::mpreal(h.real(), 256);
                            //                                auto alpha_h_mph      = alpha_mph * h_mph;
                            //                                auto twopi_mph        = mpfr::mpreal("3.14159265358979323846264338327950288419716939937510") * 2;
                            //                                auto fmod_alpha_h_mph = mpfr::mpreal();
                            //                                mpfr_fmod(fmod_alpha_h_mph.mpfr_ptr(), (alpha_mph * h_mph).mpfr_srcptr(), twopi_mph.mpfr_srcptr(),
                            //                                MPFR_RNDN); exp_ialpha_h_mph = static_cast<cx64>(std::exp(-1.0il * fmod_alpha_h_mph.toLDouble()));
                            //                                tools::log->info("fmod mph: alpha {} * h {} = {} | 2pi {} | fmod {} | exp {}",
                            //                                alpha_mph.toString(), h_mph.toString(),
                            //                                                 alpha_h_mph.toString(), twopi_mph.toString(), fmod_alpha_h_mph.toString(),
                            //                                                 exp_ialpha_h_mph);
                            //                            }

                            scalar_t exp_ialpha_t;
#if defined(DMRG_USE_QUADMATH)
                            {
                                fp128 two_pi_128       = acosq(-1.0) * fp128(2.0);
                                fp128 alpha_h_128      = fp128(-alpha.imag()) * fp128(h.real());
                                fp128 fmod_alpha_h_128 = fmodq(alpha_h_128, two_pi_128);
                                exp_ialpha_t            = std::exp(-1.0i * static_cast<fp64>(fmod_alpha_h_128));
                                //                                tools::log->info("fmod: a {0} * h {1} = {2} | 2pi {3} | fmod {4} | exp {5}", -alpha.imag(),
                                //                                h.real(), alpha_h_128, two_pi_128,
                                //                                                 fmod_alpha_h_128, exp_ialpha_t);
                            }
#else
                            {
                                fp128 two_pi_ld       = std::acos(fp128(-1.0)) * fp128(2.0);
                                fp128 alpha_h_ld      = static_cast<fp128>(std::imag(-alpha)) * static_cast<fp128>(std::real(h));
                                fp128 fmod_alpha_h_ld = std::fmod(alpha_h_ld, two_pi_ld);
                                exp_ialpha_t           = std::exp(-1.0i * static_cast<fp64>(fmod_alpha_h_ld));
                                if(std::isnan(fmod_alpha_h_ld)) { throw except::runtime_error("fmod gave nan"); }
                                //                                tools::log->info("fmod: a {0} ({0:a}) * h {1} ({1:a}) = {2} ({2:a}) | 2pi {3} ({3:a}) | fmod
                                //                                {4}({4: a}) | exp{5}({4: 5})",
                                //                                                 -alpha.imag(), h.real(), alpha_h_ld, two_pi_ld, fmod_alpha_h_ld,
                                //                                                 exp_ialpha_t);
                            }
#endif
                            return exp_ialpha_t;
                        })
                        .asDiagonal();
        return tenx::TensorMap(diag.toDenseMatrix());
    } else {
        tools::log->error("The given matrix is not diagonal!");
        if constexpr(scalar_t_is_float128 or scalar_t_is_complex_float128) {
            throw except::runtime_error("Non-diagonal Matrix exponential is undefined for type {}", sfinae::type_name<scalar_t>());
        }
        if constexpr(std::is_arithmetic_v<scalar_t>) {
            // This branch is valid for arithmetic T excluding T == __float128
            return tenx::TensorMap((static_cast<scalar_t>(alpha) * tenx::MatrixMap(op_)).exp().eval());
        } else if constexpr(sfinae::is_std_complex_v<scalar_t>) {
            if constexpr(std::is_arithmetic_v<typename scalar_t::value_type>) {
                // This branch is valid for std::complex<T> excluding T == __float128
                using value_t   = typename scalar_t::value_type;
                auto alpha_cast = std::complex<value_t>(static_cast<value_t>(std::real(alpha)), static_cast<value_t>(std::imag(alpha)));
                return tenx::TensorMap((alpha_cast * tenx::MatrixMap(op_)).exp().eval());
            }
        }
        // We can't do matrix exponential on __float128 or std::complex<__float128>()
        throw except::runtime_error("Matrix exponential is undefined for type {}", sfinae::type_name<scalar_t>());
    }
}

template Eigen::Tensor<cx64, 2>   qm::Gate::exp_internal(const Eigen::Tensor<cx64, 2> &op_, cx64 alpha) const;
template Eigen::Tensor<cx64, 2>   qm::Gate::exp_internal(const Eigen::Tensor<cx64, 2> &op_, cx128 alpha) const;
template Eigen::Tensor<cx128, 2> qm::Gate::exp_internal(const Eigen::Tensor<cx128, 2> &op_, cx64 alpha) const;
template Eigen::Tensor<cx128, 2> qm::Gate::exp_internal(const Eigen::Tensor<cx128, 2> &op_, cx128 alpha) const;

qm::Gate::Gate(const Eigen::Tensor<cx64, 2> &op_, std::vector<size_t> pos_, std::vector<long> dim_, cx64 alpha) : pos(std::move(pos_)), dim(std::move(dim_)) {
    auto dim_prod = std::accumulate(std::begin(dim), std::end(dim), 1, std::multiplies<>());
    if(dim_prod != op_.dimension(0) or dim_prod != op_.dimension(1))
        throw except::logic_error("dim {} not compatible with matrix dimensions {} x {}", dim, op_.dimension(0), op_.dimension(1));
    if(pos.size() != dim.size()) throw except::logic_error("pos.size() {} != dim.size() {}", pos, dim);
    op = exp_internal(op_, alpha);
}

qm::Gate::Gate(const Eigen::Tensor<cx64, 2> &op_, std::vector<size_t> pos_, std::vector<long> dim_, cx128 alpha) : pos(std::move(pos_)), dim(std::move(dim_)) {
    auto dim_prod = std::accumulate(std::begin(dim), std::end(dim), 1, std::multiplies<>());
    if(dim_prod != op_.dimension(0) or dim_prod != op_.dimension(1))
        throw except::logic_error("dim {} not compatible with matrix dimensions {} x {}", dim, op_.dimension(0), op_.dimension(1));
    if(pos.size() != dim.size()) throw except::logic_error("pos.size() {} != dim.size() {}", pos, dim);
    op = exp_internal(op_, alpha);
}

qm::Gate::Gate(const Eigen::Tensor<cx128, 2> &op_, std::vector<size_t> pos_, std::vector<long> dim_, cx64 alpha) : pos(std::move(pos_)), dim(std::move(dim_)) {
    auto dim_prod = std::accumulate(std::begin(dim), std::end(dim), 1, std::multiplies<>());
    if(dim_prod != op_.dimension(0) or dim_prod != op_.dimension(1))
        throw except::logic_error("dim {} not compatible with matrix dimensions {} x {}", dim, op_.dimension(0), op_.dimension(1));
    if(pos.size() != dim.size()) throw except::logic_error("pos.size() {} != dim.size() {}", pos, dim);
    op_t = exp_internal(op_, alpha);

    // We use a unary expression to cast from std::complex<__float128> to std::complex<double>
    op = op_t.unaryExpr([](auto z) { return std::complex<fp64>(static_cast<fp64>(z.real()), static_cast<fp64>(z.imag())); });
}

qm::Gate::Gate(const Eigen::Tensor<cx128, 2> &op_, std::vector<size_t> pos_, std::vector<long> dim_, cx128 alpha)
    : pos(std::move(pos_)), dim(std::move(dim_)) {
    auto dim_prod = std::accumulate(std::begin(dim), std::end(dim), 1, std::multiplies<>());
    if(dim_prod != op_.dimension(0) or dim_prod != op_.dimension(1))
        throw except::logic_error("dim {} not compatible with matrix dimensions {} x {}", dim, op_.dimension(0), op_.dimension(1));
    if(pos.size() != dim.size()) throw except::logic_error("pos.size() {} != dim.size() {}", pos, dim);
    op_t = exp_internal(op_, alpha);
    // We use a unary expression to cast from std::complex<__float128> to std::complex<double>
    op = op_t.unaryExpr([](auto z) { return std::complex<fp64>(static_cast<fp64>(z.real()), static_cast<fp64>(z.imag())); });
}

qm::Gate qm::Gate::exp(cx64 alpha) const {
    if(op_t.size() != 0)
        return Gate(op_t, pos, dim, alpha);
    else
        return Gate(op, pos, dim, alpha);
}
qm::Gate qm::Gate::exp(cx128 alpha) const {
    if(op_t.size() != 0)
        return Gate(op_t, pos, dim, alpha);
    else
        return Gate(op, pos, dim, alpha);
}
bool                   qm::Gate::isUnitary(double prec) const { return tenx::MatrixMap(op).isUnitary(prec); }
Eigen::Tensor<cx64, 2> qm::Gate::adjoint() const {
    return op.conjugate().shuffle(tenx::array2{1, 0});
    //    if(adj) return adj.value();
    //    adj = op.conjugate().shuffle(tenx::array2{1, 0});
    //    return adj.value();
}
Eigen::Tensor<cx64, 2> qm::Gate::conjugate() const {
    return op.conjugate();
    //    if(cnj) return cnj.value();
    //    cnj = op.conjugate();
    //    return cnj.value();
}
Eigen::Tensor<cx64, 2> qm::Gate::transpose() const {
    return op.shuffle(tenx::array2{1, 0});
    //    if(trn) return trn.value();
    //    trn = op.shuffle(tenx::array2{1, 0});
    //    return trn.value();
}
Eigen::Tensor<cx64, 2> qm::Gate::unaryOp(GateOp unop) const {
    switch(unop) {
        case GateOp::NONE: return op;
        case GateOp::CNJ: return conjugate();
        case GateOp::ADJ: return adjoint();
        case GateOp::TRN: return transpose();
        default: throw except::runtime_error("Unhandled switch case: {}", enum2sv(unop));
    }
}

template<auto rank, qm::Gate::Side s>
std::array<long, rank> qm::Gate::shape() const {
    if constexpr(rank == 1) {
        return {op.dimension(0) * op.dimension(1)};
    } else if constexpr(rank == 2) {
        return {op.dimension(0), op.dimension(1)};
    } else if constexpr(rank == 4 and s == Side::R) {
        long dimL = 1;
        long dimR = dim.back();
        for(size_t i = 0; i < dim.size() - 1; i++) dimL *= dim[i];
        return {dimL, dimR, dimL, dimR};
    } else if constexpr(rank == 4 and s == Side::L) {
        long dimL = dim.front();
        long dimR = 1;
        for(size_t i = 1; i < dim.size(); i++) dimR *= dim[i];
        return {dimL, dimR, dimL, dimR};
    } else {
        if(rank == 2 * dim.size()) {
            // Equal split.
            // If pos={0,1,2,3,4} and dim == {2,2,2,2,2} and rank == 10, then
            // - i iterates {0 ... 9},
            // - dims = {2,2,2,2,2 | 2,2,2,2,2}; where | separates above and below the gate.
            std::array<long, rank> dims{};
            for(size_t i = 0; i < rank; i++) dims[i] = dim[i % dim.size()];
            return dims;
        }

        throw except::range_error("Can't compute shape of rank {} for gate with pos {} and dim {}", rank, pos, dim);
    }
}

template std::array<long, 1>  qm::Gate::shape<1, qm::Gate::Side::R>() const;
template std::array<long, 2>  qm::Gate::shape<2, qm::Gate::Side::R>() const;
template std::array<long, 4>  qm::Gate::shape<4, qm::Gate::Side::R>() const;
template std::array<long, 6>  qm::Gate::shape<6, qm::Gate::Side::R>() const;
template std::array<long, 8>  qm::Gate::shape<8, qm::Gate::Side::R>() const;
template std::array<long, 10> qm::Gate::shape<10, qm::Gate::Side::R>() const;
template std::array<long, 12> qm::Gate::shape<12, qm::Gate::Side::R>() const;
template std::array<long, 14> qm::Gate::shape<14, qm::Gate::Side::R>() const;
template std::array<long, 16> qm::Gate::shape<16, qm::Gate::Side::R>() const;
template std::array<long, 18> qm::Gate::shape<18, qm::Gate::Side::R>() const;
template std::array<long, 20> qm::Gate::shape<20, qm::Gate::Side::R>() const;
template std::array<long, 22> qm::Gate::shape<22, qm::Gate::Side::R>() const;
template std::array<long, 24> qm::Gate::shape<24, qm::Gate::Side::R>() const;

template<auto rank>
std::array<long, rank> qm::Gate::shape(const std::array<size_t, rank / 2> &pos_partition) const {
    // std::vector<long> qm::Gate::shape(const std::vector<long> &pos_partition) const {
    /*!
     * Gets us the shape of this gate with positions grouped according to pos_partition
     *
     * Pos split pattern tells how to group the positions.
     *
     *  For example, if the current gate has pos {0,1,2,3,4}, with dims 32x32, we may want to group them as
     *  {{0,1}, {2,3,4}}, in which case pos_partition should be {2, 3},
     *  meaning "first a group of two positions, then another group of three positions"
     *
     *  We use the same partition up and down, so the gate dimensions will be partitioned as {2,3,2,3}.
     *  This will result in shape {2x2, 2x2x2, 2x2, 2x2x2}
     *
     */

    std::array<long, rank / 2> dimpart;
    for(auto &d : dimpart) d = 1l; // Initialize to 1

    size_t offset = 0;
    for(size_t ip = 0; ip < pos_partition.size(); ++ip) {
        auto np = pos_partition[ip]; // Number of positions on this partition
        for(size_t id = 0; id < np; ++id) { dimpart[ip] *= dim[id + offset]; }
        offset += np;
    }
    return concat(dimpart, dimpart);
}
template std::array<long, 2>  qm::Gate::shape(const std::array<size_t, 1> &) const;
template std::array<long, 4>  qm::Gate::shape(const std::array<size_t, 2> &) const;
template std::array<long, 6>  qm::Gate::shape(const std::array<size_t, 3> &) const;
template std::array<long, 8>  qm::Gate::shape(const std::array<size_t, 4> &) const;
template std::array<long, 10> qm::Gate::shape(const std::array<size_t, 5> &) const;
template std::array<long, 12> qm::Gate::shape(const std::array<size_t, 6> &) const;
template std::array<long, 14> qm::Gate::shape(const std::array<size_t, 7> &) const;
template std::array<long, 16> qm::Gate::shape(const std::array<size_t, 8> &) const;
template std::array<long, 18> qm::Gate::shape(const std::array<size_t, 9> &) const;
template std::array<long, 20> qm::Gate::shape(const std::array<size_t, 10> &) const;
template std::array<long, 22> qm::Gate::shape(const std::array<size_t, 11> &) const;
template std::array<long, 24> qm::Gate::shape(const std::array<size_t, 12> &) const;
template std::array<long, 26> qm::Gate::shape(const std::array<size_t, 13> &) const;
template std::array<long, 28> qm::Gate::shape(const std::array<size_t, 14> &) const;
template std::array<long, 30> qm::Gate::shape(const std::array<size_t, 15> &) const;
template std::array<long, 32> qm::Gate::shape(const std::array<size_t, 16> &) const;

std::vector<size_t> qm::Gate::idx() const {
    std::vector<size_t> idx(pos.size() * 2);
    std::iota(idx.begin(), idx.end(), 0);
    return idx;
}

std::vector<size_t> qm::Gate::idx_up() const {
    std::vector<size_t> idx(pos.size());
    std::iota(idx.begin(), idx.end(), 0);
    return idx;
}

std::vector<size_t> qm::Gate::idx_dn() const {
    std::vector<size_t> idx(pos.size());
    std::iota(idx.begin(), idx.end(), pos.size());
    return idx;
}

std::vector<size_t> qm::Gate::idx(const std::vector<size_t> &pos_) const {
    std::vector<size_t> idx;
    for(const auto &p_ : pos_)
        for(const auto &[i, p] : iter::enumerate(pos))
            if(p == p_) {
                idx.emplace_back(i);
                idx.emplace_back(i + pos.size());
            }
    return idx;
}

std::vector<size_t> qm::Gate::idx_up(const std::vector<size_t> &pos_) const {
    std::vector<size_t> idx;
    for(const auto &p_ : pos_)
        for(const auto &[i, p] : iter::enumerate(pos))
            if(p == p_) { idx.emplace_back(i); }
    return idx;
}

std::vector<size_t> qm::Gate::idx_dn(const std::vector<size_t> &pos_) const {
    std::vector<size_t> idx;
    for(const auto &p_ : pos_)
        for(const auto &[i, p] : iter::enumerate(pos))
            if(p == p_) { idx.emplace_back(i + pos.size()); }
    return idx;
}

bool qm::Gate::has_pos(const std::vector<size_t> &pos_) const {
    for(const auto &p_ : pos_) {
        for(const auto &p : pos) {
            if(p_ == p) return true;
        }
    }
    return false;
}

std::vector<size_t> qm::Gate::pos_intersection(const std::vector<size_t> &pos_) const {
    if(pos == std::vector<size_t>{-1ul}) return {};
    // Return indices that pos and pos_ have in common
    std::vector<size_t> pos_isect;
    std::set_intersection(pos.begin(), pos.end(), pos_.begin(), pos_.end(), back_inserter(pos_isect));
    return pos_isect;
}

std::vector<size_t> qm::Gate::pos_difference(const std::vector<size_t> &pos_) const {
    if(pos == std::vector<size_t>{-1ul}) return {};
    // Return indices that pos and pos_ do not have in common
    std::vector<size_t> pos_nsect;
    std::set_difference(pos.begin(), pos.end(), pos_.begin(), pos_.end(), back_inserter(pos_nsect));
    return pos_nsect;
}

void qm::Gate::draw_pos(std::string &layer_str, std::optional<std::string> layer_tag) const {
    if(pos.empty()) return;
    if(pos == std::vector<size_t>{-1ul}) return;

    constexpr size_t ow = 3; // Overlap width of a unitary 2-site gate box
    constexpr size_t tw = 6; // Tag width
    constexpr size_t w2 = 7; // Width of a unitary 2-site gate box. E.g. [0 , 1]

    if(layer_tag) layer_str.replace(0, tw, layer_tag.value());
    std::string gate_str;
    size_t      from = tw + pos.front() * (w2 - ow);
    if(pos.size() == 2) {
        gate_str = fmt::format("[{:<2},{:>2}]", pos.front(), pos.back());
    } else if(pos.size() == 1) {
        gate_str = fmt::format("{}", pos);
    } else {
        gate_str = fmt::format("[{:2}]", fmt::join(pos, ","));
    }
    layer_str.replace(from, gate_str.size(), gate_str);
}

qm::Gate qm::Gate::insert(const Gate &other) const { return qm::insert(*this, other); }
qm::Gate qm::Gate::connect_above(const Gate &other) const { return qm::connect(other, *this); }
qm::Gate qm::Gate::connect_below(const Gate &other) const { return qm::connect(*this, other); }

template<auto N>
qm::Gate qm::Gate::trace(const std::array<Eigen::IndexPair<Eigen::Index>, N> &idxpair) const {
    return qm::trace(*this, idxpair);
}
template qm::Gate qm::Gate::trace(const std::array<Eigen::IndexPair<Eigen::Index>, 1> &idxpairs) const;
template qm::Gate qm::Gate::trace(const std::array<Eigen::IndexPair<Eigen::Index>, 2> &idxpairs) const;

qm::Gate qm::Gate::trace_idx(const std::vector<long> &idx_) const { return qm::trace_idx(*this, idx_); }
qm::Gate qm::Gate::trace_pos(const std::vector<size_t> &pos_) const { return qm::trace_pos(*this, pos_); }
qm::Gate qm::Gate::trace_pos(size_t pos_) const { return qm::trace_pos(*this, pos_); }
cx64     qm::Gate::trace() const { return qm::trace(*this); }

template<typename L>
std::vector<std::vector<size_t>> qm::get_gate_sequence(const L &layer, bool reverse_odd) {
    static_assert(std::is_same_v<L, std::vector<qm::Gate>> or std::is_same_v<L, std::deque<qm::Gate>>);
    std::vector<std::vector<size_t>> gate_sequence;
    size_t                           gate_size = layer.front().pos.size();
    size_t                           pos_max   = layer.back().pos.back();
    for(size_t offset = 0; offset < gate_size; offset++) {
        if(offset + gate_size > pos_max + 1) break;
        auto off_idx = num::range<size_t>(offset, pos_max - gate_size + 2, gate_size);
        if(reverse_odd and num::mod<size_t>(offset, 2) == 1) std::reverse(off_idx.begin(), off_idx.end()); // If odd, reverse the sequence
        gate_sequence.emplace_back(off_idx);
    }
    return gate_sequence;
}
template std::vector<std::vector<size_t>> qm::get_gate_sequence(const std::vector<qm::Gate> &layer, bool reverse_odd);
template std::vector<std::vector<size_t>> qm::get_gate_sequence(const std::deque<qm::Gate> &layer, bool reverse_odd);

template<iter::order o>
std::vector<std::vector<size_t>> qm::get_lightcone(const std::vector<std::vector<qm::Gate>> &layers, size_t pos) {
    std::vector<std::vector<size_t>> cone;
    cone.emplace_back(std::vector<size_t>{pos});
    for(const auto &[idx_layer, layer] : iter::enumerate<o>(layers)) {
        if(layer.empty()) continue;
        auto gate_sequence = get_gate_sequence(layer);
        for(const auto &[idx_sublayer, seq] : iter::enumerate<o>(gate_sequence)) {
            std::set<size_t> match;
            // The cone width is strictly increasing.
            // So we immediately add the sites from the previous layer.
            match.insert(cone.back().begin(), cone.back().end());
            for(const auto &[idx_seq, pos_gate] : iter::enumerate<o>(seq)) {
                // Now we try to find some more matching sites in the new layer.
                auto               &u = layer[pos_gate];
                std::vector<size_t> pos_isect;
                std::set_intersection(cone.back().begin(), cone.back().end(), u.pos.begin(), u.pos.end(), back_inserter(pos_isect));
                if(not pos_isect.empty()) match.insert(u.pos.begin(), u.pos.end()); // Add positions from u if the gate connects to the current cone
            }
            if(match.empty())
                cone.emplace_back(cone.back());
            else
                cone.emplace_back(std::vector<size_t>(match.begin(), match.end()));
        }
    }
    if constexpr(o == iter::order::rev) std::reverse(cone.begin(), cone.end());
    return cone;
}

template std::vector<std::vector<size_t>> qm::get_lightcone<iter::order::def>(const std::vector<std::vector<qm::Gate>> &layers, size_t pos);
template std::vector<std::vector<size_t>> qm::get_lightcone<iter::order::rev>(const std::vector<std::vector<qm::Gate>> &layers, size_t pos);

std::vector<std::vector<size_t>> qm::get_lightcone_intersection(const std::vector<std::vector<qm::Gate>> &unitary_layers, size_t pos_tau, size_t pos_sig) {
    auto tau_cone = qm::get_lightcone<iter::order::def>(unitary_layers, pos_tau);
    auto sig_cone = qm::get_lightcone<iter::order::rev>(unitary_layers, pos_sig); // This cone is upside down!
    if(tau_cone.size() != sig_cone.size()) throw except::runtime_error("tau and sig cones should have equal size!");

    std::vector<std::vector<size_t>> int_cone;

    // Find the intersection between tau and sig cones
    for(size_t idx_sublayer = 0; idx_sublayer < tau_cone.size(); idx_sublayer++) {
        auto               &tau_sublayer = tau_cone[idx_sublayer];
        auto               &sig_sublayer = sig_cone[idx_sublayer];
        std::vector<size_t> pos_isect;
        std::set_intersection(tau_sublayer.begin(), tau_sublayer.end(), sig_sublayer.begin(), sig_sublayer.end(), back_inserter(pos_isect));
        int_cone.emplace_back(pos_isect);
    }
    if constexpr(settings::verbose_gates) {
        bool deb = tools::log->level() == spdlog::level::trace;
        if(deb) {
            fmt::print("Lightcones\n");
            auto tau_pic = get_lightcone_picture(unitary_layers, tau_cone, "tau");
            auto sig_pic = get_lightcone_picture(unitary_layers, sig_cone, "sig");
            auto int_pic = get_lightcone_picture(unitary_layers, int_cone, "int");
            for(const auto &c : iter::reverse(tau_pic)) fmt::print("{}\n", c);
            for(const auto &c : iter::reverse(sig_pic)) fmt::print("{}\n", c);
            for(const auto &c : iter::reverse(int_pic)) fmt::print("{}\n", c);
        }
    }

    return int_cone;
}

std::vector<std::deque<qm::Gate>> qm::get_lightcone_gate_selection(const std::vector<std::vector<qm::Gate>> &unitary_layers,
                                                                   const std::vector<std::vector<size_t>> &lightcone_intersection, bool reverse_odd) {
    size_t                            idx_int = 0;
    std::vector<std::deque<qm::Gate>> unitary_slayers; // Selected gates in the unitary layers
    for(const auto &[idx_layer, layer] : iter::enumerate(unitary_layers)) {
        auto gate_sequence = qm::get_gate_sequence(layer, reverse_odd);
        for(const auto &[idx_sublayer, seq] : iter::enumerate(gate_sequence)) {
            std::deque<qm::Gate> gates;
            for(const auto &[idx_seq, pos_gate] : iter::enumerate(seq)) {
                const auto &u = layer.at(pos_gate);
                if(u.has_pos(lightcone_intersection.at(idx_int))) gates.emplace_back(u);
            }
            idx_int++;
            unitary_slayers.push_back(gates);
        }
    }
    return unitary_slayers;
}
std::vector<std::deque<qm::Gate>> qm::get_lightcone_gate_selection(const std::vector<std::vector<qm::Gate>> &unitary_layers, size_t pos_tau, size_t pos_sig,
                                                                   bool reverse_odd) {
    auto                              intersection = qm::get_lightcone_intersection(unitary_layers, pos_tau, pos_sig);
    size_t                            idx_int      = 0;
    std::vector<std::deque<qm::Gate>> unitary_slayers; // Selected gates in the unitary layers
    for(const auto &[idx_layer, layer] : iter::enumerate(unitary_layers)) {
        auto gate_sequence = qm::get_gate_sequence(layer, reverse_odd);
        for(const auto &[idx_sublayer, seq] : iter::enumerate(gate_sequence)) {
            std::deque<qm::Gate> gates;
            for(const auto &[idx_seq, pos_gate] : iter::enumerate(seq)) {
                const auto &u = layer.at(pos_gate);
                if(u.has_pos(intersection.at(idx_int))) gates.emplace_back(u);
            }
            idx_int++;
            unitary_slayers.push_back(gates);
        }
    }
    return unitary_slayers;
}

std::vector<std::string> qm::get_lightcone_picture(const std::vector<std::vector<qm::Gate>> &layers, const std::vector<std::vector<size_t>> &cone,
                                                   std::string_view tag, size_t pw, std::string_view sep) {
    std::vector<std::string> pic;
    if(not layers.empty() and not cone.empty()) {
        size_t sw = sep.size();
        size_t tw = safe_cast<size_t>(tag.size()) + 5;                                       // Tag width (brackets, number and colon)
        size_t mw = safe_cast<size_t>(layers.front().back().pos.back() + 1) * (pw + sw) + 1; // max cone width
        pic       = std::vector<std::string>(cone.size(), fmt::format("{0:^{1}}", " ", tw + mw));
        for(const auto &[i, c] : iter::enumerate(cone)) {
            pic[i].replace(0, tw, fmt::format("{}[{:^2}]:", tag, i));
            for(const auto &[j, p] : iter::enumerate(c)) pic[i].replace(tw + p * (pw + sw), pw, fmt::format("{0:>{1}}{2}", p, pw, sep));
        }
    }
    return pic;
}

qm::Gate qm::insert(const qm::Gate &middle_gate, const qm::Gate &updown_gate) {
    std::vector<size_t> pos_isect; // locations that intersect: both on middle and updown gates
    std::vector<size_t> pos_nsect; // locations that do not intersect: not in both on middle and updown gates
    std::set_intersection(middle_gate.pos.begin(), middle_gate.pos.end(), updown_gate.pos.begin(), updown_gate.pos.end(), back_inserter(pos_isect));
    std::set_symmetric_difference(middle_gate.pos.begin(), middle_gate.pos.end(), updown_gate.pos.begin(), updown_gate.pos.end(), back_inserter(pos_nsect));
    if(pos_isect.empty()) return middle_gate;
    bool inc = std::includes(middle_gate.pos.begin(), middle_gate.pos.end(), updown_gate.pos.begin(), updown_gate.pos.end());

    auto shp_udn4 = updown_gate.shape<4>();
    auto shp_udn2 = updown_gate.shape<2>();
    if(not pos_isect.empty() and pos_nsect.empty() and inc) {
        // In this case we stack gates vertically that are equally wide.
        // Should be the simplest case
        if constexpr(settings::verbose_gates)
            tools::log->trace("Inserting gate pos {} between gates pos {} | pos_isect {} | pos_nsect {} | inc {}", middle_gate.pos, updown_gate.pos, pos_isect,
                              pos_nsect, inc);
        return qm::Gate{updown_gate.op.conjugate().contract(middle_gate.op, tenx::idx({1}, {0})).contract(updown_gate.op, tenx::idx({1}, {0})), middle_gate.pos,
                        middle_gate.dim};
    }
    if(pos_isect.size() == 1 and pos_nsect.size() == 1 and middle_gate.pos.size() == 1 and updown_gate.pos.size() == 2) {
        // One common location, one uncommon. Then this connects a 1-site gate with 2-site gates up and down
        tenx::idxlistpair<1> idx1;
        tenx::idxlistpair<2> idx2;
        // Decide if this connects on the left or right leg
        if(middle_gate.pos.front() == updown_gate.pos.front()) {
            /*  Left insert
             *            2    3              2    3               2    3
             *            |    |              |    |               |    |
             *           [  up  ]            [  up  ]             [  up  ]
             *            |    |              |    |               |    |
             *            0    1              0   <1>              |    |
             *            1                   3                    |    |
             *            |                   |                    |    |
             *          [mid]       ===>    [mid]        ===>    [mid]  |
             *            |                   |                    |    |
             *           (0)                  |    2               |    |
             *           (2)   3              |    |               |    |
             *            |    |              |    |               |    |
             *           [  dn  ]            [  dn  ]             [  dn  ]
             *            |    |              |    |               |    |
             *            0    1              0    1               0    1
             *
             */

            idx1 = tenx::idx({2}, {0});       // tenx::idx({0}, {0});
            idx2 = tenx::idx({3, 2}, {0, 1}); // tenx::idx({3, 0}, {0, 1});
        } else {
            /*  Right insert
             *            2    3              1    2               2    3
             *            |    |              |    |               |    |
             *           [  up  ]            [  up  ]             [  up  ]
             *            |    |              |    |               |    |
             *            0    1             <0>  (1)              |    |
             *                 1                  (3)              |    |
             *                 |                   |               |    |
             *               [mid]   ===>        [mid]     ===>    |  [mid]
             *                 |                   |               |    |
             *                (0)                  |               |    |
             *            2   (3)            <2>   |               |    |
             *            |    |              |    |               |    |
             *           [  dn  ]            [  dn  ]             [  dn  ]
             *            |    |              |    |               |    |
             *            0    1              0    1               0    1
             *
             */
            idx1 = tenx::idx({3}, {0});       // tenx::idx({1}, {0});
            idx2 = tenx::idx({2, 3}, {0, 1}); // tenx::idx({0, 3}, {0, 1});
        }
        if constexpr(settings::verbose_gates)
            tools::log->trace("Inserting gate pos {} between gates pos {} | pos_isect {} | pos_nsect {} | inc {} | contract_b", middle_gate.pos,
                              updown_gate.pos, pos_isect, pos_nsect, inc);
        auto op = contract_b(middle_gate.op, updown_gate.op, shp_udn2, shp_udn4, idx1, idx2);
        return qm::Gate{op, updown_gate.pos, updown_gate.dim};
    }
    if(pos_isect.size() == 1 and pos_nsect.size() >= 2 and updown_gate.pos.size() == 2) {
        // One common location, so it must be at the edge.
        std::array<long, 4>         shp_mid4{};
        std::array<long, 2>         dim2{};
        std::array<Eigen::Index, 6> shf6{};
        tenx::idxlistpair<1>        idx1;
        tenx::idxlistpair<2>        idx2;
        std::vector<size_t>         pos;
        std::vector<long>           dim;
        size_t                      merged = pos_nsect.size() - pos_isect.size();
        // Decide if this connects on the right or left leg
        if(middle_gate.pos.front() == updown_gate.pos.back()) {
            /*  Right insert (Free legs in mid are merged)
             *
             *    2    3                      2    3                        4    5
             *    |    |                      |    |                        |    |
             *   [  up  ]                    [  up  ]                      [  up  ]
             *    |    |                      |    |                        |    |
             *    0    1                     <0>  (1)                       |    |
             *         2     3                    (4)    5                  |    |     3                                       3   4   5              1
             *         |     |                     |     |                  |    |     |                                       |   |   |              |
             *        [  mid  ]       ===>        [  mid  ]     ===>        |   [  mid  ]       shuffle({0,1,2,4,5,3})  =    [   gate    ]  =   [   gate    ]
             *         |     |                     |     |                  |    |     |                                       |   |   |              |
             *        (0)    1                     |     3                  |    |     2                                       0   1   2              0
             *    2   (3)                    <2>   |                        |    |
             *    |    |                      |    |                        |    |
             *   [  dn  ]                    [  dn  ]                      [  dn  ]
             *    |    |                      |    |                        |    |
             *    0    1                      0    1                        0    1
             *
             */
            shp_mid4 = middle_gate.shape<4>(std::array<size_t, 2>{1, merged});
            idx1     = tenx::idx({3}, {0});       // tenx::idx({1}, {0});
            idx2     = tenx::idx({2, 4}, {0, 1}); // tenx::idx({0, 4}, {0, 1});
            shf6     = std::array<Eigen::Index, 6>{0, 1, 2, 4, 5, 3};
            pos      = concat(updown_gate.pos, subset(middle_gate.pos, 1, merged));
            dim      = concat(updown_gate.dim, subset(middle_gate.dim, 1, merged));
            dim2     = repeat(std::array<long, 1>{shp_mid4[1] * shp_udn2[0]});
        } else {
            /*  Left insert (Free legs in mid are merged)
             *
             *           2    3                      2    3                        4    5
             *           |    |                      |    |                        |    |
             *          [  up  ]                    [  up  ]                      [  up  ]
             *           |    |                      |    |                        |    |
             *           0    1                     (0)  <1>                       |    |
             *     2     3                     4    (5)                      3     |    |                                  3   4   5              0
             *     |     |                     |     |                       |     |    |                                  |   |   |              |
             *    [  mid  ]       ===>        [  mid  ]           ===>      [  mid  ]   |   shuffle({2,0,1,3,4,5})  =    [   gate    ]  =   [   gate    ]
             *     |     |                     |     |                       |     |    |                                  |   |   |              |
             *     0    (1)                    3     |                       2     |    |                                  0   1   2              1
             *          (2)   3                      |   <2>                       |    |
             *           |    |                      |    |                        |    |
             *          [  dn  ]                    [  dn  ]                      [  dn  ]
             *           |    |                      |    |                        |    |
             *           0    1                      0    1                        0    1
             *
             */
            shp_mid4 = middle_gate.shape<4>(std::array<size_t, 2>{merged, 1});
            idx1     = tenx::idx({2}, {1});       // tenx::idx({0}, {1});
            idx2     = tenx::idx({5, 2}, {0, 1}); // tenx::idx({5, 0}, {0, 1});
            shf6     = std::array<Eigen::Index, 6>{2, 0, 1, 3, 4, 5};
            pos      = concat(subset(middle_gate.pos, 0, merged), updown_gate.pos);
            dim      = concat(subset(middle_gate.dim, 0, merged), updown_gate.dim);
            dim2     = repeat(std::array<long, 1>{shp_mid4[0] * shp_udn2[0]});
        }
        if constexpr(settings::verbose_gates)
            tools::log->trace("Inserting gate pos {} between gates pos {} | pos_isect {} | pos_nsect {} | inc {} | contract_a", middle_gate.pos,
                              updown_gate.pos, pos_isect, pos_nsect, inc);
        auto op = contract_a(middle_gate.op, updown_gate.op, shp_mid4, shp_udn4, shf6, idx1, idx2, dim2);
        return qm::Gate{op, pos, dim};
    }
    if(pos_isect.size() == 2 and pos_nsect.size() >= 1 and updown_gate.pos.size() == 2) {
        // All the updown locations get contracted
        size_t                      usize = updown_gate.pos.size();
        size_t                      msize = middle_gate.pos.size();
        std::array<long, 4>         shp_mid4{};
        std::array<long, 6>         shp_mid6{};
        std::array<long, 2>         dim2{};
        std::array<Eigen::Index, 6> shf6{};
        std::array<Eigen::Index, 4> shf4{};
        tenx::idxlistpair<1>        idx_up, idx_dn;
        Eigen::Tensor<cx64, 2>      op;
        auto                        offset =
            static_cast<size_t>(std::distance(middle_gate.pos.begin(), find(middle_gate.pos.begin(), middle_gate.pos.end(), updown_gate.pos.front())));
        size_t offmin = 0;
        size_t offmax = middle_gate.pos.size() - updown_gate.pos.size();
        size_t merged = offmax;
        // Decide if this connects on the left, right or somewhere in the center
        if(offset == offmin or offset == offmax) {
            if(offset == 0) {
                /*  Insert at offset 0
                 *            1                   1                 3
                 *            |                   |                 |
                 *           [up]                [up]              [up]
                 *            |                   |                 |
                 *            0                  (0)                |
                 *            2   3              (2)  3             |   2
                 *            |   |               |   |             |   |
                 *          [  mid  ]   ===>    [  mid  ]         [  mid  ]     ===> shuffle({0,1,3,2})
                 *            |   |               |   |             |   |
                 *           (0)  1               |   1             |   1
                 *           (1)                  |                 |
                 *            |                   |                 |
                 *           [dn]                [dn]              [dn]
                 *            |                   |                 |
                 *            0                   0                 0
                 *
                 */
                shp_mid4 = middle_gate.shape<4>(std::array<size_t, 2>{usize, merged});
                dim2     = middle_gate.shape<2>();
                idx_dn   = tenx::idx({1}, {0}); // tenx::idx({0}, {0});
                idx_up   = tenx::idx({2}, {0}); // tenx::idx({2}, {0});

                shf4 = tenx::array4{0, 1, 3, 2};
                if constexpr(settings::verbose_gates)
                    tools::log->trace("Inserting gate pos {} between gates pos {} | pos_isect {} | pos_nsect {} | inc {} | contract_d", middle_gate.pos,
                                      updown_gate.pos, pos_isect, pos_nsect, inc);
                op = contract_d(middle_gate.op, updown_gate.op, shp_mid4, idx_dn, idx_up, shf4, dim2);
            } else if(offset == offmax) {
                /*  Insert at offmax
                 *            1                   1                 3
                 *            |                   |                 |
                 *           [up]                [up]              [up]
                 *            |                   |                 |
                 *            0                   0                 |
                 *        2   3               2   3             2   |
                 *        |   |               |   |             |   |
                 *      [  mid  ]   ===>    [  mid  ]         [  mid  ]    ===> shuffle({1,0,2,3})
                 *        |   |               |   |             |   |
                 *        0  (1)              1   |             1   |
                 *           (1)                  |                 |
                 *            |                   |                 |
                 *           [dn]                [dn]              [dn]
                 *            |                   |                 |
                 *            0                   0                 0
                 *
                 */

                shp_mid4 = middle_gate.shape<4>(std::array<size_t, 2>{merged, usize});
                dim2     = middle_gate.shape<2>();
                idx_dn   = tenx::idx({1}, {1});
                idx_up   = tenx::idx({3}, {0});
                shf4     = tenx::array4{1, 0, 2, 3};
                if constexpr(settings::verbose_gates)
                    tools::log->trace("Inserting gate pos {} between gates pos {} | pos_isect {} | pos_nsect {} | inc {} | contract_d", middle_gate.pos,
                                      updown_gate.pos, pos_isect, pos_nsect, inc);
                op = contract_d(middle_gate.op, updown_gate.op, shp_mid4, idx_dn, idx_up, shf4, dim2);
            }
        } else {
            /*  Insert at offmax
             *           1                     1                      5
             *           |                     |                      |
             *          [up]                  [up]                   [up]
             *           |                     |                      |
             *           0                    (0)                     |
             *       3   4   5             3  (4)  5              3   |   4
             *       |   |   |             |   |   |              |   |   |
             *      [   mid   ]    ===>   [   mid   ]    ===>    [   mid   ]       ===> shuffle({1,0,2,3,5,4})
             *       |   |   |             |   |   |              |   |   |
             *       0  (1)  2             1   |   2              1   |   2
             *          (1)                    |                      |
             *           |                     |                      |
             *          [dn]                  [dn]                   [dn]
             *           |                     |                      |
             *           0                     0                      0
             *
             */

            shp_mid6 = middle_gate.shape<6>(std::array<size_t, 3>{offset, usize, msize - usize - offset});
            dim2     = middle_gate.shape<2>();
            idx_dn   = tenx::idx({1}, {1});
            idx_up   = tenx::idx({4}, {0});
            shf6     = tenx::array6{1, 0, 2, 3, 5, 4};
            if constexpr(settings::verbose_gates)
                tools::log->trace("Inserting gate pos {} between gates pos {} | pos_isect {} | pos_nsect {} | inc {} | contract_c", middle_gate.pos,
                                  updown_gate.pos, pos_isect, pos_nsect, inc);
            op = contract_c(middle_gate.op, updown_gate.op, shp_mid6, idx_dn, idx_up, shf6, dim2);
        }
        if(op.size() == 0) throw std::logic_error("No op computed!");
        return qm::Gate{op, middle_gate.pos, middle_gate.dim};
    }

    throw except::runtime_error("Insert case not implemented: middle pos {} | updown pos {} | pos_isect {} | pos_nsect {}", middle_gate.pos, updown_gate.pos,
                                pos_isect, pos_nsect);
}

qm::Gate qm::connect(const qm::Gate &dn_gate, const qm::Gate &up_gate) {
    std::vector<size_t> pos_isect; // locations that intersect: both on middle and updown gates
    std::vector<size_t> pos_nsect; // locations that do not intersect: not in both on middle and updown gates
    std::set_intersection(dn_gate.pos.begin(), dn_gate.pos.end(), up_gate.pos.begin(), up_gate.pos.end(), back_inserter(pos_isect));
    std::set_symmetric_difference(dn_gate.pos.begin(), dn_gate.pos.end(), up_gate.pos.begin(), up_gate.pos.end(), back_inserter(pos_nsect));
    bool inc = std::includes(dn_gate.pos.begin(), dn_gate.pos.end(), up_gate.pos.begin(), up_gate.pos.end());
    if(pos_isect.empty()) return dn_gate;

    if constexpr(settings::verbose_gates) tools::log->trace("Connecting dn {} | up {}", dn_gate.pos, up_gate.pos);
    if(not pos_isect.empty() and pos_nsect.empty() and inc) {
        // This case is a vertical stack. Should be the simplest case
        auto op = tenx::gemm(dn_gate.op, up_gate.op); // Calls BLAS GEMM
        return qm::Gate{op, dn_gate.pos, dn_gate.dim};
    }
    if(not pos_isect.empty() and not pos_nsect.empty() and inc) {
        std::array<long, 6> dn_shp6{};
        std::array<long, 4> dn_shp4{};
        std::vector<size_t> pos;
        std::vector<long>   dim;
        auto                dn_size  = dn_gate.pos.size();
        auto                up_size  = up_gate.pos.size();
        size_t              dn_merge = dn_size - pos_isect.size();
        size_t              up_merge = up_size;
        size_t              offset   = up_gate.pos.front() - dn_gate.pos.front();
        // Decide if this connects on the right or right leg
        if(dn_gate.pos.back() == pos_isect.back()) {
            /*  Right connection
             *
             *            |    |
             *            |  [up]
             *            |   |
             *          [  dn  ]
             *           |    |
             *
             */
            dn_shp4   = dn_gate.shape<4>(std::array<size_t, 2>{dn_merge, up_merge});
            auto dim2 = dn_gate.shape<2>();
            return qm::Gate{dn_gate.op.reshape(dn_shp4).contract(up_gate.op, tenx::idx({3}, {0})).reshape(dim2), dn_gate.pos, dn_gate.dim};
        } else if(dn_gate.pos.front() == pos_isect.front()) {
            /*  Left connection
             *
             *            |   |
             *          [up]  |
             *           |    |
             *          [  dn  ]
             *           |    |
             */
            dn_shp4   = dn_gate.shape<4>(std::array<size_t, 2>{up_merge, dn_merge});
            auto dim2 = dn_gate.shape<2>();
            return qm::Gate{dn_gate.op.reshape(dn_shp4).contract(up_gate.op, tenx::idx({2}, {0})).shuffle(tenx::array4{0, 1, 3, 2}).reshape(dim2), dn_gate.pos,
                            dn_gate.dim};
        } else {
            /*  Center connection
             *
             *           |    |   |
             *           |  [up]  |
             *           |   |    |
             *          [    dn    ]
             *           |    |   |
             */
            dn_shp6   = dn_gate.shape<6>(std::array<size_t, 3>{offset, up_size, dn_size - up_size - offset});
            auto dim2 = dn_gate.shape<2>();
            return qm::Gate{dn_gate.op.reshape(dn_shp6).contract(up_gate.op, tenx::idx({4}, {0})).shuffle(tenx::array6{0, 1, 2, 3, 5, 4}).reshape(dim2),
                            dn_gate.pos, dn_gate.dim};
        }
    }
    if(pos_isect.size() == 1 and pos_nsect.size() >= 2) {
        // One common location, and one uncommon. Then this connects two 2-site gates
        std::vector<size_t> pos;
        std::vector<long>   dim;
        auto                dn_size  = dn_gate.pos.size();         // Number of sites
        auto                up_size  = up_gate.pos.size();         // Number of sites
        size_t              dn_merge = dn_size - pos_isect.size(); // Number of sites to merge
        size_t              up_merge = up_size - pos_isect.size(); // Number of sites to merge
        // Decide if this connects on the right or right leg
        if(dn_gate.pos.front() == up_gate.pos.back()) {
            /*  Right connection
             *      4    5     2
             *      |    |     |           3   4   5              1
             *     [  up  ]    |           |   |   |              |
             *      |    |     |   =     [   gate    ]  =   [   gate    ]
             *      |   [  dn  ]           |   |   |              |
             *      |    |     |           0   1   2              0
             *      3    0     1
             */
            auto up_shp4 = up_gate.shape<4>(std::array<size_t, 2>{up_merge, 1});
            auto dn_shp4 = dn_gate.shape<4>(std::array<size_t, 2>{1, dn_merge});
            auto dim2    = repeat(std::array<long, 1>{up_shp4[0] * up_shp4[1] * dn_shp4[1]});
            pos          = concat(up_gate.pos, subset(dn_gate.pos, 1, dn_merge));
            dim          = concat(up_gate.dim, subset(dn_gate.dim, 1, dn_merge));
            return qm::Gate{
                dn_gate.op.reshape(dn_shp4).contract(up_gate.op.reshape(up_shp4), tenx::idx({2}, {1})).shuffle(tenx::array6{3, 0, 1, 4, 5, 2}).reshape(dim2),
                pos, dim};
        } else {
            /*  Left connection
             *     2     4    5
             *     |     |    |             3   4   5              1
             *     |    [  up  ]            |   |   |              |
             *     |     |    |      =    [   gate    ]  =   [   gate    ]
             *    [  dn  ]    |             |   |   |              |
             *     |     |    |             0   1   2              0
             *     0     1    3
             */

            auto dn_shp4 = dn_gate.shape<4>(std::array<size_t, 2>{dn_merge, 1});
            auto up_shp4 = up_gate.shape<4>(std::array<size_t, 2>{1, up_merge});
            auto dim2    = repeat(std::array<long, 1>{dn_shp4[0] * up_shp4[0] * up_shp4[1]});
            pos          = concat(subset(dn_gate.pos, 0, dn_merge), up_gate.pos);
            dim          = concat(subset(dn_gate.dim, 0, dn_merge), up_gate.dim);
            return qm::Gate{
                dn_gate.op.reshape(dn_shp4).contract(up_gate.op.reshape(up_shp4), tenx::idx({3}, {0})).shuffle(tenx::array6{0, 1, 3, 2, 4, 5}).reshape(dim2),
                pos, dim};
        }
    }

    throw except::runtime_error(
        fmt::format("Connect case not implemented: dn pos {} | up pos {} | pos_isect {} | pos_nsect {}", dn_gate.pos, up_gate.pos, pos_isect, pos_nsect));
}

qm::Gate qm::trace_idx(const qm::Gate &gate, const std::vector<long> &idx) {
    if(idx.size() == 2) return qm::trace(gate, tenx::idx({idx[0]}, {idx[1]}));
    // idx is already pair-ordered, so when we call tenx::idx we need to undo that order
    // which explains the weird indexing in the following case
    if(idx.size() == 4) return qm::trace(gate, tenx::idx({idx[0], idx[2]}, {idx[1], idx[3]}));
    throw except::runtime_error("Tracing {} indices is not implemented", idx.size());
}

qm::Gate qm::trace_pos(const qm::Gate &gate, const std::vector<size_t> &pos) {
    if(pos.size() <= 2) {
        auto              idx = gate.idx(pos);
        std::vector<long> idx_long(idx.begin(), idx.end());
        return qm::trace_idx(gate, idx_long);
    }
    qm::Gate tmp = gate;
    for(auto &p : pos) { tmp = tmp.trace_pos(std::vector<size_t>{p}); }
    return tmp;
}

qm::Gate qm::trace_pos(const qm::Gate &gate, size_t pos) { return qm::trace_pos(gate, gate.idx(std::vector<size_t>{pos})); }

cx64 qm::trace(const qm::Gate &gate) {
    qm::Gate t = qm::trace_pos(gate, gate.pos);
    if(not t.pos.empty()) throw except::logic_error("Gate should be empty after tracing all positions. Got pos: {}", t.pos);
    if(t.op.dimension(0) * t.op.dimension(1) != 1)
        throw except::logic_error("Gate should have cx64 op after tracing all positions. Got dims: {}", t.op.dimensions());
    return t.op(0);
}

template<auto N>
qm::Gate qm::trace(const qm::Gate &gate, const std::array<Eigen::IndexPair<Eigen::Index>, N> &idxpairs) {
    // Trace off pairs of indices on a gate
    // This is done one by one recursively until all pairs have been traced
    if constexpr(settings::verbose_gates) {
        if constexpr(N == 1) tools::log->trace("Tracing pos {} | dim {} | index pair [{},{}]", gate.pos, gate.dim, idxpairs[0].first, idxpairs[0].second);
        if constexpr(N == 2)
            tools::log->trace("Tracing gate pos {} | dim {} | index pairs [{},{}][{},{}]", gate.pos, gate.dim, idxpairs[0].first, idxpairs[0].second,
                              idxpairs[1].first, idxpairs[1].second);
    }

    // Compute the remaining indices positions and dimensions
    auto idx = gate.idx(); // Twice as long as pos and dim! It has the top and bottom indices
    auto pos = gate.pos;
    auto dim = gate.dim;
    for(const auto &pair : idxpairs) { // Idx pairs has the indices of the legs we remove
        auto it1 = std::find(idx.begin(), idx.end(), std::max(pair.first, pair.second));
        if(it1 != idx.end()) {
            auto idx_rm1 = std::distance(idx.begin(), it1);
            if(idx_rm1 < safe_cast<long>(pos.size())) pos.erase(pos.begin() + idx_rm1);
            if(idx_rm1 < safe_cast<long>(dim.size())) dim.erase(dim.begin() + idx_rm1);
            idx.erase(it1);
        }
        auto it2 = std::find(idx.begin(), idx.end(), std::min(pair.first, pair.second));
        if(it2 != idx.end()) {
            auto idx_rm2 = std::distance(idx.begin(), it2);
            if(idx_rm2 < safe_cast<long>(pos.size())) pos.erase(pos.begin() + idx_rm2);
            if(idx_rm2 < safe_cast<long>(dim.size())) dim.erase(dim.begin() + idx_rm2);
            idx.erase(it2);
        }
    }
    if(idx.size() != 2 * (gate.pos.size() - N)) throw std::logic_error("Wrong number of indices removed");

    // Assuming the gate is symmetric, we can compute the rank2 dimensions for the storage of the gate op
    long dim_prod = std::accumulate(std::begin(dim), std::end(dim), 1, std::multiplies<>()); // Product of all dimensions of the remaining top legs of the gate
    std::array<long, 2> dim2{dim_prod, dim_prod};

    // Flatten the index pairs to a list
    Eigen::array<Eigen::Index, 2 * N> idx_list;
    for(size_t i = 0; i < idxpairs.size(); i++) {
        idx_list[2 * i]     = idxpairs[i].first;
        idx_list[2 * i + 1] = idxpairs[i].second;
    }

    // Trace
    Eigen::Tensor<cx64, 2> op_traced;
    if(gate.dim.size() == 1) {
        if constexpr(N == 1) { // dim size 1 is special! It can't take 2 index pairs
            op_traced = gate.op.reshape(gate.shape<2>()).trace(idx_list).reshape(dim2);
        } else {
            throw except::runtime_error("Can't trace {} index pairs on gate with pos {}", N, gate.pos);
        }
    }
    /* clang-format off */
    else if(gate.dim.size() == 2 ) op_traced = gate.op.reshape(gate.shape<4>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 3 ) op_traced = gate.op.reshape(gate.shape<6>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 4 ) op_traced = gate.op.reshape(gate.shape<8>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 5 ) op_traced = gate.op.reshape(gate.shape<10>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 6 ) op_traced = gate.op.reshape(gate.shape<12>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 7 ) op_traced = gate.op.reshape(gate.shape<14>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 8 ) op_traced = gate.op.reshape(gate.shape<16>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 9 ) op_traced = gate.op.reshape(gate.shape<18>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 10 ) op_traced = gate.op.reshape(gate.shape<20>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 11 ) op_traced = gate.op.reshape(gate.shape<22>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 12 ) op_traced = gate.op.reshape(gate.shape<24>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 13 ) op_traced = gate.op.reshape(gate.shape<26>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 14 ) op_traced = gate.op.reshape(gate.shape<28>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 15 ) op_traced = gate.op.reshape(gate.shape<30>()).trace(idx_list).reshape(dim2);
    else if(gate.dim.size() == 16 ) op_traced = gate.op.reshape(gate.shape<32>()).trace(idx_list).reshape(dim2);
    /* clang-format on */
    else
        throw except::runtime_error("Trace not implemented: N == {} | dim.size() == {}", N, gate.dim.size());

    return Gate{op_traced, pos, dim};
}

template qm::Gate qm::trace(const qm::Gate &gate, const std::array<Eigen::IndexPair<Eigen::Index>, 1> &idxpairs);
template qm::Gate qm::trace(const qm::Gate &gate, const std::array<Eigen::IndexPair<Eigen::Index>, 2> &idxpairs);
