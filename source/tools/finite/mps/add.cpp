#include "../mps.h"
#include "tensors/state/StateFinite.h"
#include <tensors/site/mps/MpsSite.h>
StateFinite tools::finite::mps::add_states(const StateFinite &stateA, const StateFinite &stateB) {
    auto algo_type  = stateA.get_algorithm();
    auto model_size = stateA.get_length<size_t>();
    auto position   = stateA.get_position<long>();
    if(model_size != stateB.get_length<size_t>())
        throw except::logic_error("tools::finite::mps::add_states(): state1 and state2 must be the same length. Got state1 = {}, state2 = {}", model_size,
                                  stateB.get_length<size_t>());
    if(position != stateB.get_position<long>())
        throw except::logic_error("tools::finite::mps::add_states(): state1 and state2 must be at the same position. Got state1 @ {}, state2 @ {}", position,
                                  stateB.get_position<long>());

    auto direct_sum_M = [](const MpsSite &mpsA, const MpsSite &mpsB) {
        const auto &dA = mpsA.dimensions();
        const auto &dB = mpsB.dimensions();
        assert(dA[0] == dB[0]);
        Eigen::Tensor<cx64, 3> M(dA[0], dA[1] + dB[1], dA[2] + dB[2]);
        M.setZero();
        auto oA         = Eigen::DSizes<long, 3>{0, 0, 0};         // Offset for mpsA
        auto oB         = Eigen::DSizes<long, 3>{0, dA[1], dA[2]}; // Offset for mpsB
        M.slice(oA, dA) = mpsA.get_M_bare();
        M.slice(oB, dB) = mpsB.get_M_bare();
        return M;
    };
    auto direct_sum_L = [](const MpsSite &mpsA, const MpsSite &mpsB) {
        const auto &dA = mpsA.get_L().dimensions();
        const auto &dB = mpsB.get_L().dimensions();
        Eigen::Tensor<cx64, 1> L(dA[0] + dB[0]);
        L.setZero();
        auto oA         = Eigen::DSizes<long, 1>{0};     // Offset for mpsA
        auto oB         = Eigen::DSizes<long, 1>{dA[0]}; // Offset for mpsB
        L.slice(oA, dA) = mpsA.get_L();
        L.slice(oB, dB) = mpsB.get_L();
        return L;
    };
    auto direct_sum_LC = [](const MpsSite &mpsA, const MpsSite &mpsB) {
        const auto &dA = mpsA.get_LC().dimensions();
        const auto &dB = mpsB.get_LC().dimensions();
        assert(mpsA.isCenter());
        assert(mpsB.isCenter());
        Eigen::Tensor<cx64, 1> LC(dA[0] + dB[0]);
        LC.setZero();
        auto oA          = Eigen::DSizes<long, 1>{0};     // Offset for mpsA
        auto oB          = Eigen::DSizes<long, 1>{dA[0]}; // Offset for mpsB
        LC.slice(oA, dA) = mpsA.get_LC();
        LC.slice(oB, dB) = mpsB.get_LC();
        return LC;
    };
    auto stateR = StateFinite(algo_type, model_size, position);
    stateR.set_name(stateA.get_name());

    for(size_t site = 0; site < model_size; ++site) {
        const auto &mpsA = *stateA.mps_sites[site];
        const auto &mpsB = *stateB.mps_sites[site];
        auto       &mpsR = *stateR.mps_sites[site];
        if(site == 0) {
            Eigen::Tensor<cx64, 3> M = mpsA.get_M_bare().concatenate(mpsB.get_M_bare(), 2);
            Eigen::Tensor<cx64, 1> L = mpsA.get_L().concatenate(mpsB.get_L(), 0);
            mpsR.set_M(M);
            mpsR.set_L(L);
            if(mpsR.isCenter()) {
                Eigen::Tensor<cx64, 1> LC = mpsA.get_LC().concatenate(mpsB.get_LC(), 0);
                mpsR.set_LC(LC);
            }
        } else if(site + 1 == model_size) {
            Eigen::Tensor<cx64, 3> M = mpsA.get_M_bare().concatenate(mpsB.get_M_bare(), 1);
            Eigen::Tensor<cx64, 1> L = mpsA.get_L().concatenate(mpsB.get_L(), 0);
            mpsR.set_M(M);
            mpsR.set_L(L);
            if(mpsR.isCenter()) {
                Eigen::Tensor<cx64, 1> LC = mpsA.get_LC().concatenate(mpsB.get_LC(), 0);
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
