#pragma once

#include "../lbit.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tid/tid.h"
#include "tools/common/log.h"

namespace settings {
    inline constexpr bool debug_cls = false;
}

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_dn,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_up, bool adj_dn) {
    if(mpos_dn.size() != mpos_up.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_up.size());
    if constexpr(settings::debug_cls) tools::log->debug("Merging mpos dn and up");
    auto t_merge         = tid::tic_scope("merge2");
    auto mpos            = std::vector<Eigen::Tensor<Scalar, 4>>(mpos_dn.size());
    auto cfg             = svd::config();
    cfg.rank_max         = settings::flbit::cls::mpo_circuit_svd_bondlim;
    cfg.truncation_limit = settings::flbit::cls::mpo_circuit_svd_trnclim;
    cfg.switchsize_gesdd = settings::precision::svd_switchsize_bdc;
    cfg.svd_lib          = svd::lib::lapacke;
    cfg.svd_rtn          = svd::rtn::geauto;
    auto  svd            = svd::solver(cfg);
    auto &threads        = tenx::threads::get();

    {
        auto mpo_du = Eigen::Tensor<Scalar, 4>();
        auto SV     = Eigen::Tensor<Scalar, 2>();
        SV.resize(std::array<long, 2>{1, mpos_dn.front().dimension(0) * mpos_up.front().dimension(0)});
        SV.setConstant(1.0);
        for(size_t idx = 0; idx < mpos.size(); ++idx) {
            {
                auto           t_svmpos = tid::tic_scope("svmpos");
                auto           dd       = mpos_dn[idx].dimensions();
                auto           du       = mpos_up[idx].dimensions();
                constexpr auto shf5     = std::array<long, 5>{0, 1, 3, 4, 2};
                auto           rsh_svl3 = std::array<long, 3>{SV.dimension(0), dd[0], du[0]};
                auto           rsh_mpo4 = std::array<long, 4>{SV.dimension(0), dd[1] * du[1], du[2], (adj_dn ? dd[2] : dd[3])};
                auto           idx_contract1 = tenx::idx({1}, {0});
                auto           idx_contract2 = adj_dn ? tenx::idx({1, 4}, {0, 3}) : tenx::idx({1, 3}, {0, 3});
                auto           mpos_dn_idx_  = adj_dn ? Eigen::Tensor<Scalar, 4>(mpos_dn[idx].conjugate()) : mpos_dn[idx];
                mpo_du.resize(rsh_mpo4);
                Eigen::Tensor<Scalar, 5> t1(tenx::array5{SV.dimension(0), du[0], dd[1], dd[2], dd[3]});
                t1.device(*threads->dev) = SV.reshape(rsh_svl3).contract(mpos_dn_idx_, idx_contract1);
                mpo_du.device(*threads->dev) = t1.contract(mpos_up[idx], idx_contract2).shuffle(shf5).reshape(rsh_mpo4);
            }
            if(idx + 1 < mpos.size()) {
                auto t_split            = tid::tic_scope("split");
                std::tie(mpos[idx], SV) = svd.split_mpo_l2r(mpo_du, cfg);
            } else {
                mpos[idx] = mpo_du;
                SV.resize(std::array<long, 2>{mpo_du.dimension(1), 1});
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> {} + SV {} | trunc {:.4e}", idx, mpo_du.dimensions(), mpos[idx].dimensions(), SV.dimensions(),
                                  svd.get_truncation_error());
        }
    }

    {
        auto t_back = tid::tic_scope("back");
        auto mpoUS  = Eigen::Tensor<Scalar, 4>();
        auto US     = Eigen::Tensor<Scalar, 2>();
        US.resize(std::array<long, 2>{mpos.back().dimension(1), 1});
        US.setConstant(1.0);
        for(size_t idx = mpos.size() - 1; idx < mpos.size(); --idx) {
            auto dmpo  = mpos[idx].dimensions();
            auto rshUS = std::array<long, 4>{dmpo[0], US.dimension(1), dmpo[2], dmpo[3]};
            mpoUS.resize(rshUS);
            mpoUS.device(*threads->dev) = mpos[idx].contract(US, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
            if(idx > 0) {
                std::tie(US, mpos[idx]) = svd.split_mpo_r2l(mpoUS, cfg);
            } else {
                mpos[idx] = mpoUS;
                US.resize(std::array<long, 2>{1, mpoUS.dimension(0)});
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> US {} + mpo {} | trunc {:.4e}", idx, dmpo, US.dimensions(), mpos[idx].dimensions(),
                                  svd.get_truncation_error());
        }
    }
    if constexpr(settings::debug_cls)
        for(const auto &[idx, mpo] : iter::enumerate(mpos)) tools::log->debug("mpo {:2}: {}", idx, mpo.dimensions());

    return mpos;
}

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_dn,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_md,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_up) {
    if(mpos_md.empty()) return merge_unitary_mpo_layers(mpos_dn, mpos_up, true);
    if(mpos_dn.size() != mpos_up.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_up.size());
    if(mpos_dn.size() != mpos_md.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_md.size());
    if constexpr(settings::debug_cls) tools::log->debug("Merging mpos dn md up");
    auto t_merge         = tid::tic_scope("merge3");
    auto mpos            = std::vector<Eigen::Tensor<Scalar, 4>>(mpos_dn.size());
    auto cfg             = svd::config();
    cfg.rank_max         = settings::flbit::cls::mpo_circuit_svd_bondlim;
    cfg.truncation_limit = settings::flbit::cls::mpo_circuit_svd_trnclim;
    cfg.switchsize_gesdd = settings::precision::svd_switchsize_bdc;
    cfg.svd_lib          = svd::lib::lapacke;
    cfg.svd_rtn          = svd::rtn::geauto;
    auto  svd            = svd::solver(cfg);
    auto &threads        = tenx::threads::get();
    {
        auto mpo_dmu = Eigen::Tensor<Scalar, 4>();
        auto SV      = Eigen::Tensor<Scalar, 2>();
        SV.resize(std::array<long, 2>{1, mpos_dn.front().dimension(0) * mpos_md.front().dimension(0) * mpos_up.front().dimension(0)});
        SV.setConstant(1.0);
        for(size_t idx = 0; idx < mpos.size(); ++idx) {
            {
                auto           t_svmpos = tid::tic_scope("svmpos");
                auto           dd       = mpos_dn[idx].dimensions();
                auto           dm       = mpos_md[idx].dimensions();
                auto           du       = mpos_up[idx].dimensions();
                constexpr auto shf6     = std::array<long, 6>{0, 1, 3, 4, 5, 2};
                auto           rsh_svl4 = std::array<long, 4>{SV.dimension(0), dd[0], dm[0], du[0]};
                auto           rsh_mpo4 = std::array<long, 4>{SV.dimension(0), dd[1] * dm[1] * du[1], du[2], dd[2]};
                mpo_dmu.resize(rsh_mpo4);
                Eigen::Tensor<Scalar, 6> t1(tenx::array6{SV.dimension(0), dm[0], du[0], dd[1], dd[2], dd[3]});
                t1.device(*threads->dev) = SV.reshape(rsh_svl4).contract(mpos_dn[idx].conjugate(), tenx::idx({1}, {0}));
                Eigen::Tensor<Scalar, 6> t2(tenx::array6{SV.dimension(0), du[0], dd[1], dd[2], dm[1], dm[2]});
                t2.device(*threads->dev) = t1.contract(mpos_md[idx], tenx::idx({1, 5}, {0, 3}));
                mpo_dmu.device(*threads->dev) = t2.contract(mpos_up[idx], tenx::idx({1, 5}, {0, 3})).shuffle(shf6).reshape(rsh_mpo4);
            }
            if(idx + 1 < mpos.size()) {
                auto t_split            = tid::tic_scope("split");
                std::tie(mpos[idx], SV) = svd.split_mpo_l2r(mpo_dmu, cfg);
            } else {
                mpos[idx] = mpo_dmu;
                SV.resize(std::array<long, 2>{mpo_dmu.dimension(1), 1});
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> {} + SV {} | trunc {:.4e}", idx, mpo_dmu.dimensions(), mpos[idx].dimensions(), SV.dimensions(),
                                  svd.get_truncation_error());
        }
    }

    {
        auto t_back = tid::tic_scope("back");
        auto mpoUS  = Eigen::Tensor<Scalar, 4>();
        auto US     = Eigen::Tensor<Scalar, 2>();
        US.resize(std::array<long, 2>{mpos.back().dimension(1), 1});
        US.setConstant(1.0);
        for(size_t idx = mpos.size() - 1; idx < mpos.size(); --idx) {
            auto dmpo  = mpos[idx].dimensions();
            auto rshUS = std::array<long, 4>{dmpo[0], US.dimension(1), dmpo[2], dmpo[3]};
            mpoUS.resize(rshUS);
            mpoUS.device(*threads->dev) = mpos[idx].contract(US, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
            if(idx > 0) {
                std::tie(US, mpos[idx]) = svd.split_mpo_r2l(mpoUS, cfg);
            } else {
                mpos[idx] = mpoUS;
                US.resize(std::array<long, 2>{1, mpoUS.dimension(0)});
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> US {} + mpo {} | trunc {:.4e}", idx, dmpo, US.dimensions(), mpos[idx].dimensions(),
                                  svd.get_truncation_error());
        }
    }
    if constexpr(settings::debug_cls)
        for(const auto &[idx, mpo] : iter::enumerate(mpos)) tools::log->debug("mpo {:2}: {}", idx, mpo.dimensions());

    return mpos;
}
