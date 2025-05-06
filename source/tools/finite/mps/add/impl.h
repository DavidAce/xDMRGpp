#pragma once
#include "../../mps.h"
#include "tensors/state/StateFinite.h"
#include <tensors/site/mps/MpsSite.h>

template<typename Scalar>
StateFinite<Scalar> tools::finite::mps::add_states(const StateFinite<Scalar> &stateA, const StateFinite<Scalar> &stateB) {
    auto algo_type  = stateA.template get_algorithm();
    auto model_size = stateA.template get_length<size_t>();
    auto position   = stateA.template get_position<long>();
    if(model_size != stateB.template get_length<size_t>())
        throw except::logic_error("tools::finite::mps::add_states(): state1 and state2 must be the same length. Got state1 = {}, state2 = {}", model_size,
                                  stateB.template get_length<size_t>());
    if(position != stateB.template get_position<long>())
        throw except::logic_error("tools::finite::mps::add_states(): state1 and state2 must be at the same position. Got state1 @ {}, state2 @ {}", position,
                                  stateB.template get_position<long>());

    auto direct_sum_M = [](const MpsSite<Scalar> &mpsA, const MpsSite<Scalar> &mpsB) {
        const auto &dA = mpsA.dimensions();
        const auto &dB = mpsB.dimensions();
        assert(dA[0] == dB[0]);
        Eigen::Tensor<Scalar, 3> M(dA[0], dA[1] + dB[1], dA[2] + dB[2]);
        M.setZero();
        auto oA         = Eigen::DSizes<long, 3>{0, 0, 0};         // Offset for mpsA
        auto oB         = Eigen::DSizes<long, 3>{0, dA[1], dA[2]}; // Offset for mpsB
        M.slice(oA, dA) = mpsA.get_M_bare();
        M.slice(oB, dB) = mpsB.get_M_bare();
        return M;
    };
    auto direct_sum_L = [](const MpsSite<Scalar> &mpsA, const MpsSite<Scalar> &mpsB) {
        const auto              &dA = mpsA.get_L().dimensions();
        const auto              &dB = mpsB.get_L().dimensions();
        Eigen::Tensor<Scalar, 1> L(dA[0] + dB[0]);
        L.setZero();
        auto oA         = Eigen::DSizes<long, 1>{0};     // Offset for mpsA
        auto oB         = Eigen::DSizes<long, 1>{dA[0]}; // Offset for mpsB
        L.slice(oA, dA) = mpsA.template get_L_as<Scalar>();
        L.slice(oB, dB) = mpsB.template get_L_as<Scalar>();
        return L;
    };
    auto direct_sum_LC = [](const MpsSite<Scalar> &mpsA, const MpsSite<Scalar> &mpsB) {
        const auto &dA = mpsA.get_LC().dimensions();
        const auto &dB = mpsB.get_LC().dimensions();
        assert(mpsA.isCenter());
        assert(mpsB.isCenter());
        Eigen::Tensor<Scalar, 1> LC(dA[0] + dB[0]);
        LC.setZero();
        auto oA          = Eigen::DSizes<long, 1>{0};     // Offset for mpsA
        auto oB          = Eigen::DSizes<long, 1>{dA[0]}; // Offset for mpsB
        LC.slice(oA, dA) = mpsA.get_LC();
        LC.slice(oB, dB) = mpsB.get_LC();
        return LC;
    };
    auto stateR = StateFinite<Scalar>(algo_type, model_size, position);
    stateR.set_name(stateA.get_name());

    for(size_t site = 0; site < model_size; ++site) {
        const auto &mpsA = *stateA.mps_sites[site];
        const auto &mpsB = *stateB.mps_sites[site];
        auto       &mpsR = *stateR.mps_sites[site];
        if(site == 0) {
            Eigen::Tensor<Scalar, 3> M = mpsA.get_M_bare().concatenate(mpsB.get_M_bare(), 2);
            Eigen::Tensor<Scalar, 1> L = mpsA.template get_L_as<Scalar>().concatenate(mpsB.template get_L_as<Scalar>(), 0);
            mpsR.set_M(M);
            mpsR.set_L(L);
            if(mpsR.isCenter()) {
                Eigen::Tensor<Scalar, 1> LC = mpsA.template get_LC_as<Scalar>().concatenate(mpsB.template get_LC_as<Scalar>(), 0);
                mpsR.set_LC(LC);
            }
        } else if(site + 1 == model_size) {
            Eigen::Tensor<Scalar, 3> M = mpsA.template get_M_bare_as<Scalar>().concatenate(mpsB.get_M_bare(), 1);
            Eigen::Tensor<Scalar, 1> L = mpsA.template get_L_as<Scalar>().concatenate(mpsB.template get_L_as<Scalar>(), 0);
            mpsR.set_M(M);
            mpsR.set_L(L);
            if(mpsR.isCenter()) {
                Eigen::Tensor<Scalar, 1> LC = mpsA.template get_LC_as<Scalar>().concatenate(mpsB.template get_LC_as<Scalar>(), 0);
                mpsR.set_LC(LC);
            }
        } else {
            mpsR.set_M(direct_sum_M(mpsA, mpsB));
            mpsR.set_L(direct_sum_L(mpsA, mpsB));
            if(mpsR.isCenter()) mpsR.set_LC(direct_sum_LC(mpsA, mpsB));
        }
    }

    return stateR;
}

