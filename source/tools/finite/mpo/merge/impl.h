
#include "../../mpo.h"
#include "config/debug.h"
#include "general/iter.h"
#include "math/svd.h"
#include "tid/tid.h"

/*! \brief Merge two MPO layers into a single one using SVD.
 *
 * Step 1:
 * @verbatim
 *
 *
 *
 *                                        2
 *                                        |
 *                                0--[ mpo_up_L ]--1
 *                                        |
 *                                        3
 *            2                           2
 *            |                           |
 *   0--[ mpo_dnmdup_L ]--1   =  0--[ mpo_md_L ]--1
 *            |                           |
 *            3                           3 ---------|
 *                                        2          |
 *                                        |          |
 *                               0--[ mpo_dn_L*]--1  |  (we take the adjoint of mpo_dn)
 *                                        |          |
 *                                        3 ----------

 *
 * @endverbatim
 *
 * Step 2:
 * @verbatim
 *              1                            2                            2
 *              |             SVD            |                            |
 *     0--[ mpo_updn_L ]--3    =      0--[ mpo(i) ]--1    S*VT* 0--[ mpo_updn_R ]--1
 *              |                            |                            |
 *              2                            3                            3
 * @endverbatim
 *
 *
 * @endverbatim
 */
template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_merged_mpos(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_dn,
                                                                          const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_md,
                                                                          const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_up, const svd::config &cfg) {
    if(mpos_dn.size() != mpos_up.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_up.size());
    if(mpos_dn.size() != mpos_md.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_md.size());
    auto t_merge = tid::tic_scope("merge3");
    auto mpos    = std::vector<Eigen::Tensor<Scalar, 4>>(mpos_dn.size());
    // auto cfg             = svd::config();
    // cfg.rank_max         = settings::flbit::cls::mpo_circuit_svd_bondlim;
    // cfg.truncation_limit = settings::flbit::cls::mpo_circuit_svd_trnclim;
    // cfg.switchsize_gesdd = settings::precision::svd_switchsize_bdc;
    // cfg.svd_lib          = svd::lib::lapacke;
    // cfg.svd_rtn          = svd::rtn::geauto;
    auto  svd     = svd::solver(cfg);
    auto &threads = tenx::threads::get();
    {
        // Initialize a dummy SV to start contracting from the left
        auto mpo_dmu = Eigen::Tensor<Scalar, 4>();
        auto SV      = Eigen::Tensor<Scalar, 2>();
        SV.resize(std::array<long, 2>{1, mpos_dn.front().dimension(0) * mpos_md.front().dimension(0) * mpos_up.front().dimension(0)});
        SV.setConstant(1.0);
        for(size_t idx = 0; idx < mpos.size(); ++idx) {
            {
                auto t_svmpos = tid::tic_scope("svmpos");
                auto dd       = mpos_dn[idx].dimensions();
                auto dm       = mpos_md[idx].dimensions();
                auto du       = mpos_up[idx].dimensions();

                constexpr auto shf6     = std::array<long, 6>{0, 1, 3, 4, 5, 2};
                auto           rsh_svl4 = std::array<long, 4>{SV.dimension(0), dd[0], dm[0], du[0]};                 // Dims of SV from the left side
                auto           rsh_mpo4 = std::array<long, 4>{SV.dimension(0), dd[1] * dm[1] * du[1], du[2], dd[2]}; // Dims of the new mpo_dmu to split
                mpo_dmu.resize(rsh_mpo4);
                mpo_dmu.device(*threads->dev) = SV.reshape(rsh_svl4)
                                                    .contract(mpos_dn[idx].conjugate(), tenx::idx({1}, {0}))
                                                    .contract(mpos_md[idx], tenx::idx({1, 5}, {0, 3}))
                                                    .contract(mpos_up[idx], tenx::idx({1, 5}, {0, 3}))
                                                    .shuffle(shf6)
                                                    .reshape(rsh_mpo4);
            }
            if(idx + 1 < mpos.size()) {
                auto t_split            = tid::tic_scope("split");
                std::tie(mpos[idx], SV) = svd.split_mpo_l2r(mpo_dmu, cfg);
            } else {
                mpos[idx] = mpo_dmu;
                SV.resize(std::array<long, 2>{mpo_dmu.dimension(1), 1}); // So that the log message is nice
            }
            // if constexpr(settings::debug)
            tools::log->debug("split svd mpo {}: {} --> {} + SV {} | trunc {:.4e}", idx, mpo_dmu.dimensions(), mpos[idx].dimensions(), SV.dimensions(),
                              svd.get_truncation_error());
        }
    }

    // Now compress once backwards
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
                US.resize(std::array<long, 2>{1, mpoUS.dimension(0)}); // So that the log message is nice
            }
            // if constexpr(settings::debug)
            tools::log->debug("split svd mpo {}: {} --> US {} + mpo {} | trunc {:.4e}", idx, dmpo, US.dimensions(), mpos[idx].dimensions(),
                              svd.get_truncation_error());
        }
    }
    // if constexpr(settings::debug)
    for(const auto &[idx, mpo] : iter::enumerate(mpos)) tools::log->debug("mpo {:2}: {}", idx, mpo.dimensions());

    return mpos;
}