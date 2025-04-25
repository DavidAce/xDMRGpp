#include "EnvVar.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "math/hash.h"
#include "math/num.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tools/common/log.h"
#include <utility>

template class EnvVar<fp32>;
template class EnvVar<fp64>;
template class EnvVar<fp128>;
template class EnvVar<cx32>;
template class EnvVar<cx64>;
template class EnvVar<cx128>;


template<typename Scalar>
EnvVar<Scalar>::EnvVar(std::string side_, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) : EnvBase<Scalar>(std::move(side_), "var", mps, mpo) {
    set_edge_dims(mps, mpo);
}

template<typename Scalar>
EnvVar<Scalar> EnvVar<Scalar>::enlarge(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const {
    tools::log->trace("EnvVar<Scalar>::enlarge(mps,mpo): {}{}[{}]", tag, side, get_position());
    // enlarge() uses "this" block together with mps and mpo to generate a new environment block corresponding to a neighboring site
    if constexpr(settings::debug)
        if(not num::all_equal(get_position(), mps.get_position(), mpo.get_position()))
            throw except::logic_error("EnvVar<Scalar>::enlarge: {}{}[{}]: All positions are not equal: env {} | mps {} | mpo {}", tag, side, get_position(),
                                      get_position(), mps.get_position(), mpo.get_position());

    EnvVar env = *this;

    if(env.sites == 0 and (not block or block->size() == 0)) {
        env.set_edge_dims(mps, mpo);
        env.position = mps.get_position();
        return env;
    }

    env.enlarge(mps.template get_M_bare_as<Scalar>(), mpo.template MPO2_as<Scalar>());
    // Update positions assuming this is a finite chain.
    // This needs to be corrected (on the right side) on infinite chains
    if(env.side == "L")
        env.position = mps.get_position() + 1;
    else if(env.side == "R")
        env.position = mps.get_position() - 1;
    else
        throw except::logic_error("Expected environment side L or R, got: " + side);

    env.tag = "var";
    // Save the hash id's used to create the new block in env
    env.unique_id_env = get_unique_id();
    env.unique_id_mps = mps.get_unique_id();
    env.unique_id_mpo = mpo.get_unique_id_sq();
    if constexpr(settings::debug) {
        tools::log->trace("class_env_{}::enlarge(mps,mpo): side({}), pos({}): unique_id_env: {}", tag, side, get_position(), env.unique_id_env.value());
        tools::log->trace("class_env_{}::enlarge(mps,mpo): side({}), pos({}): unique_id_mps: {}", tag, side, get_position(), env.unique_id_mps.value());
        tools::log->trace("class_env_{}::enlarge(mps,mpo): side({}), pos({}): unique_id_mpo: {}", tag, side, get_position(), env.unique_id_mpo.value());
    }
    return env;
}

template<typename Scalar>
void EnvVar<Scalar>::refresh(const EnvVar &env, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) {
    // If side == L, env,mps and mpo are all corresponding to the neighbor on the left
    // If side == R, env,mps and mpo are all corresponding to the neighbor on the right
    if constexpr(settings::debug)
        if(not num::all_equal(env.get_position(), mps.get_position(), mpo.get_position()))
            throw except::logic_error("class_env_{}::enlarge(): side({}), pos({}),: All positions are not equal: env {} | mps {} | mpo {}", tag, side,
                                      get_position(), get_position(), mps.get_position(), mpo.get_position());

    if(side == "L" and get_position() != mps.get_position() + 1)
        throw except::logic_error(
            fmt::format("EnvVar<Scalar>::refresh(pos == {}): This env{} needs env, mps and mpo at position {}", get_position(), side, get_position() - 1));
    if(side == "R" and get_position() + 1 != mps.get_position())
        throw except::logic_error(
            fmt::format("EnvVar<Scalar>::refresh(pos == {}): This env{} needs env, mps and mpo at position {}", get_position(), side, get_position() + 1));

    // We refresh this block if any of these conditions hold:
    //   not has_block()
    //   unique_id_env != env.unique_id;
    //   unique_id_mps != mps.unique_id;
    //   unique_id_mpo != mpo.unique_id_sq;

    if(not has_block()) {
        if constexpr(settings::debug) tools::log->trace("Refreshing {} env{}({}): missing block", tag, side, get_position());
        *this = env.enlarge(mps, mpo);
        return;
    }
    bool        refresh = false;
    std::string reason;
    if(env.get_unique_id() != unique_id_env) {
        refresh = true;
        if constexpr(settings::debug) {
            reason.append(fmt::format("| env({}) new {} ", env.get_position(), env.get_unique_id()));
            if(unique_id_env) reason.append(fmt::format("!= old {} ", unique_id_env.value()));
        }
    }
    if(mps.get_unique_id() != unique_id_mps) {
        refresh = true;
        if constexpr(settings::debug) {
            reason.append(fmt::format("| mps({}) new {} ", mps.get_position(), mps.get_unique_id()));
            if(unique_id_mps) reason.append(fmt::format("!= old {} ", unique_id_mps.value()));
        }
    }
    auto mpo_unique_id = tag == "ene" ? mpo.get_unique_id() : mpo.get_unique_id_sq();
    if(mpo_unique_id != unique_id_mpo) {
        refresh = true;
        if constexpr(settings::debug) {
            reason.append(fmt::format("| mpo({}) new {} ", mpo.get_position(), mpo_unique_id));
            if(unique_id_mpo) reason.append(fmt::format("!= old {} ", unique_id_mpo.value()));
        }
    }

    if(refresh) {
        [[maybe_unused]] size_t unique_id_bef;
        if constexpr(settings::debug) {
            unique_id_bef = get_unique_id();
            tools::log->trace("Refreshing {} env{}({}): modified {}", tag, side, get_position(), reason);
        }

        build_block(*env.block, mps.template get_M_bare_as<Scalar>(), mpo.template MPO2_as<Scalar>());
        // Store id's to objects used to create this env.
        unique_id_env = env.get_unique_id();
        unique_id_mps = mps.get_unique_id();
        unique_id_mpo = mpo.get_unique_id_sq();

        if constexpr(settings::debug) {
            if(unique_id_bef == get_unique_id()) tools::log->debug("Refreshing {} env{}({}): id did not change: {}", tag, side, get_position(), unique_id_bef);
            //                throw except::logic_error("Refreshing {} env{}({}): failed: id did not change: {}", tag, side, get_position(),
            //                unique_id_bef);
        }
    }
}

template<typename Scalar>
void EnvVar<Scalar>::set_edge_dims(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) {
    Eigen::Tensor<Scalar, 1> edge;
    if(side == "L") edge = mpo.template get_MPO2_edge_left<Scalar>();
    if(side == "R") edge = mpo.template get_MPO2_edge_right<Scalar>();
    std::size_t unique_id_edge = hash::hash_buffer(edge.data(), static_cast<size_t>(edge.size()));
    if(unique_id_env and unique_id_env.value() == unique_id_edge) return;
    if constexpr(settings::debug)
        if(side != "L" and side != "R") throw except::runtime_error("Wrong side: {}", side);

    tools::log->trace("EnvVar<Scalar>::set_edge_dims: {}{}({}): {}", tag, side, get_position(), edge.dimensions());
    set_edge_dims(mps.template get_M_bare_as<Scalar>(), mpo.template MPO2_as<Scalar>(), edge);
    unique_id     = get_unique_id();
    unique_id_env = unique_id_edge;
    unique_id_mps = mps.get_unique_id();
    unique_id_mpo = mpo.get_unique_id_sq();
}
