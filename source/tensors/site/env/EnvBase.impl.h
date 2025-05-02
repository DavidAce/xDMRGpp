#pragma once
#include "EnvBase.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/sfinae.h"
#include "math/hash.h"
#include "math/num.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "math/tenx/threads.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tools/common/log.h"
#include <utility>

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
EnvBase<Scalar>::EnvBase() : block(std::make_unique<Eigen::Tensor<Scalar, 3>>()) {
    assert_block();
} // default ctor
template<typename Scalar>
EnvBase<Scalar>::~EnvBase() = default; // default dtor
template<typename Scalar>
EnvBase<Scalar>::EnvBase(EnvBase &&other) noexcept = default; // default move ctor
template<typename Scalar>
EnvBase<Scalar> &EnvBase<Scalar>::operator=(EnvBase<Scalar> &&other) noexcept = default; // default move assign

template<typename Scalar>
EnvBase<Scalar>::EnvBase(const EnvBase &other)
    : block(std::make_unique<Eigen::Tensor<Scalar, 3>>(*other.block)), sites(other.sites), position(other.position), side(other.side), tag(other.tag),
      unique_id(other.unique_id), unique_id_mps(other.unique_id_mps), unique_id_mpo(other.unique_id_mpo), unique_id_env(other.unique_id_env),
      is_real_cached(other.is_real_cached), has_nan_cached(other.has_nan_cached) {
    assert_block();
}

template<typename Scalar>
EnvBase<Scalar>::EnvBase(size_t position_, std::string side_, std::string tag_)
    : block(std::make_unique<Eigen::Tensor<Scalar, 3>>()), position(position_), side(std::move(side_)), tag(std::move(tag_)) {
    assert_block();
}
template<typename Scalar>
EnvBase<Scalar>::EnvBase(std::string side_, std::string tag_, const MpsSite<Scalar> &MPS, const MpoSite<Scalar> &MPO)
    : block(std::make_unique<Eigen::Tensor<Scalar, 3>>()), side(std::move(side_)), tag(std::move(tag_)) {
    if(MPS.get_position() != MPO.get_position())
        throw except::logic_error("MPS and MPO have different positions: {} != {}", MPS.get_position(), MPO.get_position());
    position = MPS.get_position();
    assert_block();
}

template<typename Scalar>
EnvBase<Scalar> &EnvBase<Scalar>::operator=(const EnvBase<Scalar> &other) {
    if(this != &other) {
        block          = std::make_unique<Eigen::Tensor<Scalar, 3>>(*other.block);
        sites          = other.sites;
        position       = other.position;
        side           = other.side;
        tag            = other.tag;
        unique_id      = other.unique_id;
        unique_id_mps  = other.unique_id_mps;
        unique_id_mpo  = other.unique_id_mpo;
        unique_id_env  = other.unique_id_env;
        is_real_cached = other.is_real_cached;
        has_nan_cached = other.has_nan_cached;
    }
    assert_block();
    return *this;
}

template<typename Scalar>
void EnvBase<Scalar>::assert_block() const {
    if(side != "L" and side != "R") throw except::runtime_error("Expected side [L|R]. Got: [{}]", side);
    if(tag != "ene" and tag != "var") throw except::runtime_error("Expected tag [var|ene]. Got: [{}]", tag);
    if(not block) throw except::runtime_error("env block {} {} at pos {} is nullptr", tag, side, get_position());
}

template<typename Scalar>
void EnvBase<Scalar>::build_block(Eigen::Tensor<Scalar, 3> &otherblock, const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo) {
    /*!< Contracts a site into the block-> */
    // Note that otherblock, mps and mpo should correspond to the same site! I.e. their "get_position()" are all equal.
    // This can't be checked here though, so do that before calling this function.
    if constexpr(settings::debug) tools::log->trace("EnvBase<Scalar>::build_block(otherblock,mps,mpo): side({}), pos({})", side, get_position());
    unique_id      = std::nullopt;
    unique_id_env  = std::nullopt;
    unique_id_mps  = std::nullopt;
    unique_id_mpo  = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
    auto &threads  = tenx::threads::get();
    if(not block) block = std::make_unique<Eigen::Tensor<Scalar, 3>>();
    if(side == "L") {
        /*! # Left environment block contraction
         *   [       ]--0         [       ]--0 0--[LB]--1 1--[  GA    ]--2
         *   [       ]            [       ]                      |
         *   [       ]            [       ]                      0
         *   [       ]            [       ]
         *   [       ]            [       ]                      2
         *   [       ]            [       ]                      |
         *   [ block ]--2     =   [lblock ]--2            0--[   M    ]--1
         *   [       ]            [       ]                      |
         *   [       ]            [       ]                      3
         *   [       ]            [       ]
         *   [       ]            [       ]                      0
         *   [       ]            [       ]                      |
         *   [       ]--1         [       ]--1 0--[LB]--1  1--[GA conj ]--2
         */

        if(mps.dimension(0) != mpo.dimension(2))
            throw except::runtime_error("env{} {} pos {} dimension mismatch: mps dim[{}]:{} != mpo   dim[{}]:{}", side, tag, position.value(), 0,
                                        mps.dimension(0), 2, mpo.dimension(2));
        if(mps.dimension(1) != otherblock.dimension(0))
            throw except::runtime_error("env{} {} pos {} dimension mismatch: mps dim[{}]:{} != left-block dim[{}]:{}", side, tag, position.value(), 1,
                                        mps.dimension(1), 0, otherblock.dimension(0));
        if(mpo.dimension(0) != otherblock.dimension(2))
            throw except::runtime_error("env{} {} pos {} dimension mismatch: mpo dim[{}]:{} != left-block dim[{}]:{}", side, tag, position.value(), 0,
                                        mpo.dimension(0), 2, otherblock.dimension(2));
        // block->resize(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
        // block->device(*threads->dev) = otherblock.contract(mps, tenx::idx({0}, {1}))
        //                                    .contract(mpo, tenx::idx({1, 2}, {0, 2}))
        //                                    .contract(mps.conjugate(), tenx::idx({0, 3}, {1, 0}))
        //                                    .shuffle(tenx::array3{0, 2, 1});
        Eigen::Tensor<Scalar, 4> block_mps_mpo(otherblock.dimension(0), mps.dimension(2), mpo.dimension(1), mpo.dimension(2));
        block_mps_mpo.device(*threads->dev) = otherblock.contract(mps.conjugate(), tenx::idx({1}, {1})).contract(mpo, tenx::idx({1, 2}, {0, 3}));
        block->resize(mps.dimension(2), mps.dimension(2), mpo.dimension(1));
        block->device(*threads->dev) = mps.contract(block_mps_mpo, tenx::idx({0, 1}, {3, 0}));

    } else if(side == "R") {
        /*! # Right environment contraction
         *   0--[       ]          1--[   GB   ]--2  0--[LB]--1  0--[       ]
         *      [       ]                  |                        [       ]
         *      [       ]                  0                        [       ]
         *      [       ]                                           [       ]
         *      [       ]                  2                        [       ]
         *      [       ]                  |                        [       ]
         *   2--[ block ]     =     0--[   M   ]--1              2--[ rblock]
         *      [       ]                  |                        [       ]
         *      [       ]                  3                        [       ]
         *      [       ]                                           [       ]
         *      [       ]                  0                        [       ]
         *      [       ]                  |                        [       ]
         *   1--[       ]           1--[GB conj]--2  0--[LB]--1  1--[       ]
         */

        if(mps.dimension(0) != mpo.dimension(2))
            throw except::runtime_error("env{} {} pos {} dimension mismatch: mps dim[{}]:{} != mpo dim[{}]:{}", side, tag, position.value(), 0,
                                        mps.dimension(0), 2, mpo.dimension(2));
        if(mps.dimension(2) != otherblock.dimension(0))
            throw except::runtime_error("env{} {} pos {} dimension mismatch: mps dim[{}]:{} != right-block dim[{}]:{}", side, tag, position.value(), 2,
                                        mps.dimension(2), 0, otherblock.dimension(0));
        if(mpo.dimension(1) != otherblock.dimension(2))
            throw except::runtime_error("env{} {} pos {} dimension mismatch: mpo dim[{}]:{} != right-block dim[{}]:{}", side, tag, position.value(), 1,
                                        mpo.dimension(1), 2, otherblock.dimension(2));
        // block->resize(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
        // block->device(*threads->dev) = otherblock.contract(mps, tenx::idx({0}, {2}))
        // .contract(mpo, tenx::idx({1, 2}, {1, 2}))
        // .contract(mps.conjugate(), tenx::idx({0, 3}, {2, 0}))
        // .shuffle(tenx::array3{0, 2, 1});

        Eigen::Tensor<Scalar, 4> block_mps_mpo(otherblock.dimension(0), mps.dimension(1), mpo.dimension(0), mpo.dimension(2));
        block_mps_mpo.device(*threads->dev) = otherblock.contract(mps.conjugate(), tenx::idx({1}, {2})).contract(mpo, tenx::idx({1, 2}, {1, 3}));
        block->resize(mps.dimension(1), mps.dimension(1), mpo.dimension(0));
        block->device(*threads->dev) = mps.contract(block_mps_mpo, tenx::idx({0, 2}, {3, 0}));
    }
}

template<typename Scalar>
void EnvBase<Scalar>::enlarge(const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo) {
    /*!< Contracts a site into the current block-> */

    // NOTE:
    // If side == "L", the mps and mpo should correspond to this env position+1
    // If side == "R", the mps and mpo should correspond to this env position-1
    // There is no way to check the positions here, so checks should be done before calling this function

    assert_block();
    Eigen::Tensor<Scalar, 3> thisblock = *block;
    build_block(thisblock, mps, mpo);
    sites++;
    if(position) {
        if(side == "L")
            position = position.value() + 1;
        else if(position.value() > 0)
            position = position.value() - 1;
    }
}

template<typename Scalar>
void EnvBase<Scalar>::clear() {
    assert_block();
    block->resize(0, 0, 0); // = Eigen::Tensor<Scalar,3>();
    tools::log->trace("Ejected env{} pos {}", side, get_position());
    unique_id = std::nullopt;

    // Do not clear the empty edge
    //    if(sites > 0) {
    //        block->resize(0, 0, 0); // = Eigen::Tensor<Scalar,3>();
    //        tools::log->trace("Ejected env{} pos {}", side, get_position());
    //        unique_id = std::nullopt;
    //    }

    unique_id_env  = std::nullopt;
    unique_id_mps  = std::nullopt;
    unique_id_mpo  = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 3> &EnvBase<Scalar>::get_block() const {
    assert_block();
    return *block;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3> &EnvBase<Scalar>::get_block() {
    assert_block();
    return *block;
}

template<typename Scalar>
bool EnvBase<Scalar>::has_block() const {
    return block != nullptr and block->size() != 0;
}

template<typename Scalar>
std::array<long, 3> EnvBase<Scalar>::dimensions() const {
    assert_block();
    return block->dimensions();
}

template<typename Scalar>
std::array<long, 3> EnvBase<Scalar>::get_dims() const {
    assert_block();
    return block->dimensions();
}

template<typename Scalar>
void EnvBase<Scalar>::assert_validity() const {
    assert_block();
    if(tenx::hasNaN(*block)) { throw except::runtime_error("Environment {} side {} at position {} has NAN's", tag, side, get_position()); }
}

template<typename Scalar>
void EnvBase<Scalar>::assert_unique_id(const EnvBase &env, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const {
    std::vector<std::string> msg;
    if(unique_id_env and env.get_unique_id() != unique_id_env.value())
        msg.emplace_back(fmt::format("env({}): new {} | old {}", env.get_position(), env.get_unique_id(), unique_id_env.value()));
    if(unique_id_mps and mps.get_unique_id() != unique_id_mps.value())
        msg.emplace_back(fmt::format("mps({}): new {} | old {}", mps.get_position(), mps.get_unique_id(), unique_id_mps.value()));

    // mpo is special:
    // if this is an energy env we check the normal mpo, but
    // if this is a variance env we need to check the squared mpo
    if(tag == "ene") {
        if(unique_id_mpo and mpo.get_unique_id() != unique_id_mpo.value())
            msg.emplace_back(fmt::format("mpo({}): new {} | old {}", mpo.get_position(), mpo.get_unique_id(), unique_id_mpo.value()));
    } else if(tag == "var") {
        if(unique_id_mpo and mpo.get_unique_id_sq() != unique_id_mpo.value())
            msg.emplace_back(fmt::format("mpo_sq({}): new {} | old {}", mpo.get_position(), mpo.get_unique_id_sq(), unique_id_mpo.value()));
    } else {
        msg.emplace_back(fmt::format("Unrecognized tag: [{}]", tag));
    }
    if(not msg.empty())
        throw except::runtime_error("assert_unique_id: {}{}({}): unique id mismatch:\n{}\n"
                                    "Hint: remember to rebuild edges after operations that may modify them, "
                                    "like 1-site merge, move, normalization, projection, etc\n",
                                    tag, side, get_position(), fmt::join(msg, "\n"));
}

template<typename Scalar>
bool EnvBase<Scalar>::is_real() const {
    assert_block();
    if constexpr(settings::debug) {
        return tenx::isReal(*block);
    } else {
        is_real_cached = is_real_cached.value_or(tenx::isReal(*block));
        return is_real_cached.value();
    }
}

template<typename Scalar>
bool EnvBase<Scalar>::has_nan() const {
    assert_block();
    if constexpr(settings::debug) {
        return tenx::hasNaN(*block);
    } else {
        has_nan_cached = has_nan_cached.value_or(tenx::hasNaN(*block));
        return has_nan_cached.value();
    }
}

template<typename Scalar>
size_t EnvBase<Scalar>::get_position() const {
    if(position)
        return position.value();
    else
        throw except::runtime_error("Position hasn't been set on env side {}", side);
}

template<typename Scalar>
size_t EnvBase<Scalar>::get_sites() const {
    return sites;
}

template<typename Scalar>
void EnvBase<Scalar>::set_edge_dims(const Eigen::Tensor<Scalar, 3> &MPS, const Eigen::Tensor<Scalar, 4> &MPO, const Eigen::Tensor<Scalar, 1> &edge) {
    assert_block();
    if(side == "L") {
        long mpsDim = MPS.dimension(1);
        long mpoDim = MPO.dimension(0);
        block->resize(tenx::array3{mpsDim, mpsDim, mpoDim});
        block->setZero();
        for(long i = 0; i < mpsDim; i++) {
            std::array<long, 1> extent1                     = {mpoDim};
            std::array<long, 3> offset3                     = {i, i, 0};
            std::array<long, 3> extent3                     = {1, 1, mpoDim};
            block->slice(offset3, extent3).reshape(extent1) = edge;
        }
    }
    if(side == "R") {
        long mpsDim = MPS.dimension(2);
        long mpoDim = MPO.dimension(1);
        block->resize(tenx::array3{mpsDim, mpsDim, mpoDim});
        block->setZero();
        for(long i = 0; i < mpsDim; i++) {
            std::array<long, 1> extent1                     = {mpoDim};
            std::array<long, 3> offset3                     = {i, i, 0};
            std::array<long, 3> extent3                     = {1, 1, mpoDim};
            block->slice(offset3, extent3).reshape(extent1) = edge;
        }
    }
    sites          = 0;
    unique_id      = std::nullopt;
    unique_id_env  = std::nullopt;
    unique_id_mps  = std::nullopt;
    unique_id_mpo  = std::nullopt;
    is_real_cached = std::nullopt;
    has_nan_cached = std::nullopt;
}

// void EnvBase<Scalar>::set_mixing_factor(double alpha) {
//     if(std::isnan(alpha)) throw except::logic_error("EnvBase<Scalar>::set_mixing_factor: alpha cannot be nan");
//     mixing_factor_alpha = alpha;
// }
// double EnvBase<Scalar>::get_mixing_factor() const { return mixing_factor_alpha; }

template<typename Scalar>
std::size_t EnvBase<Scalar>::get_unique_id() const {
    if(unique_id) return unique_id.value();
    assert_block();
    unique_id = hash::hash_buffer(block->data(), safe_cast<size_t>(block->size()));
    return unique_id.value();
}

template<typename Scalar>
std::optional<std::size_t> EnvBase<Scalar>::get_unique_id_env() const {
    return unique_id_env;
}
template<typename Scalar>
std::optional<std::size_t> EnvBase<Scalar>::get_unique_id_mps() const {
    return unique_id_mps;
}
template<typename Scalar>
std::optional<std::size_t> EnvBase<Scalar>::get_unique_id_mpo() const {
    return unique_id_mpo;
}

template<typename T>
Eigen::Tensor<T, 3> get_expansion_term_exec(const Eigen::Tensor<T, 3> &mps, const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 3> &env,
                                            std::string_view side) {
    auto               &threads = tenx::threads::get();
    Eigen::Tensor<T, 3> P;
    if(side == "L") {
        long spin = mps.dimension(0);
        long chiL = mps.dimension(1);
        long chiR = mps.dimension(2) * mpo.dimension(1);
        assert(env.dimension(0) == chiL);
        P.resize(spin, chiL, chiR);
        P.device(*threads->dev) = env.contract(tenx::asScalarType<T>(mps), tenx::idx({0}, {1}))
                                      .contract(tenx::asScalarType<T>(mpo), tenx::idx({1, 2}, {0, 2}))
                                      .shuffle(tenx::array4{3, 0, 1, 2})
                                      .reshape(tenx::array3{spin, chiL, chiR});

    } else if(side == "R") {
        long spin = mps.dimension(0);
        long chiL = mps.dimension(1) * mpo.dimension(0);
        long chiR = mps.dimension(2);
        assert(env.dimension(0) == chiR);
        P.resize(spin, chiL, chiR);
        P.device(*threads->dev) = env.contract(tenx::asScalarType<T>(mps), tenx::idx({0}, {2}))
                                      .contract(tenx::asScalarType<T>(mpo), tenx::idx({1, 2}, {1, 2}))
                                      .shuffle(tenx::array4{3, 0, 1, 2})
                                      .reshape(tenx::array3{spin, chiL, chiR});
    }
    return P;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 3> EnvBase<Scalar>::get_expansion_term(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const {
    if constexpr(sfinae::is_std_complex_v<T> and sfinae::is_std_complex_v<Scalar>) {
        if(is_real() and mps.is_real() and mpo.is_real()) {
            using RealT = typename T::value_type;
            return get_expansion_term<RealT>(mps, mpo).template cast<T>();
        }
    }

    assert(tag == "ene" or tag == "var");
    assert(side == "L" or side == "R");
    assert(get_position() == mps.get_position());
    assert(mps.get_position() == mpo.get_position());
    decltype(auto) mpo_tensor = tag == "ene" ? mpo.template MPO_as<T>() : mpo.template MPO2_as<T>();
    decltype(auto) mps_tensor = mps.template get_M_as<T>();
    decltype(auto) blk_tensor = get_block_as<T>();
    return get_expansion_term_exec<T>(mps_tensor, mpo_tensor, blk_tensor, side);
}


