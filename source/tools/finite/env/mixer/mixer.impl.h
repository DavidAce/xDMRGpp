#pragma once
#include "../../env.h"
#include "../BondExpansionConfig.h"
#include "../mixer.h"
#include "config/settings.h"
#include "math/linalg/tensor/to_string.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tools/finite/opt.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"

template<typename EnvType>
requires sfinae::is_specialization_v<EnvType, EnvEne> || sfinae::is_specialization_v<EnvType, EnvVar>
void diagonal_environment_padding(EnvType &env, typename EnvType::Scalar pad_value, Eigen::Index newChi, size_t unique_id_mps) {
    using Scalar                                = typename EnvType::Scalar;
    Eigen::DSizes<Eigen::Index, 3> env_dims_old = env.get_block().dimensions();
    Eigen::DSizes<Eigen::Index, 3> env_dims_new = Eigen::DSizes<Eigen::Index, 3>{newChi, newChi, env_dims_old[2]};
    Eigen::Tensor<Scalar, 3>       new_block(env_dims_new);
    new_block.setZero();
    new_block.slice(Eigen::DSizes<Eigen::Index, 3>{0, 0, 0}, env_dims_old) = env.get_block();
    for(Eigen::Index w = 0; w < env_dims_old[2]; ++w) {
        for(Eigen::Index r = env_dims_old[0]; r < env_dims_new[0]; ++r) { new_block(r, r, w) = pad_value; }
    }
    env.get_block() = new_block;
    env.unique_id.reset();
    env.get_unique_id();
    env.unique_id_mps = unique_id_mps;
}

/*! Typically, the bond dimension of M_P >> bond_lim, and we do not know apriori which columns
 * to keep from the P-part of M_P = [A, P]. Running the eigenvalue solver a few iterations sets
 * mixes A and P and sets appropriate weights on all of M_P. We can then truncate M_P with SVD,
 * without hurting the precision of the pre-expansion MPS.
 */
template<typename Scalar>
void tools::finite::env::internal::run_expansion_term_mixer(TensorsFinite<Scalar> &tensors, long posP, long pos0, [[maybe_unused]] Scalar pad_value_env,
                                                            BondExpansionConfig bcfg) {
    [[maybe_unused]] auto &state = tensors.get_state();
    [[maybe_unused]] auto &model = tensors.get_model();
    [[maybe_unused]] auto &edges = tensors.get_edges();
    tensors.clear_cache();
    tensors.clear_measurements();

    auto &mpsP = state.get_mps_site(posP);
    auto &mps0 = state.get_mps_site(pos0);

    // We have just expanded MP = [A, P] and padded Cpad = [C, 0] and N0 = [B, 0]
    // Now we need to run a few eigensolver iterations on Cpad*N0 to populate the zero-padding.
    auto active_sites_backup = tensors.active_sites;
    tensors.activate_sites(std::vector<size_t>{safe_cast<size_t>(pos0)});
    rebuild_edges(state, model, edges);

    // Run one step of the DMRG optimizer
    auto optm                                 = OptMeta();
    optm.eigs_iter_max                        = 1;
    optm.eigs_lib                             = "EIGSMPO";
    optm.eigs_residual_correction_type        = "CHEAP_OLSEN";
    optm.eigs_preconditioner_type             = "SOLVE";
    optm.eigs_nev                             = 1;
    optm.eigs_abstol                          = settings::precision::eigs_abstol_max;
    optm.eigs_reltol                          = settings::precision::eigs_reltol_max;
    optm.eigs_blk                             = settings::precision::eigs_blk_min;
    optm.eigs_ncv                             = settings::precision::eigs_ncv_min;
    optm.eigs_jcbMaxBlockSize                 = std::min(1l, settings::precision::eigs_jcb_blocksize_min);
    optm.eigs_use_coarse_inner_preconditioner = false;
    optm.optRitz                              = bcfg.optRitz;
    optm.optAlgo                              = bcfg.optAlgo;
    optm.optType                              = bcfg.optType;
    optm.optSolver                            = OptSolver::EIGS;

    // Set up the dmrg block size
    optm.min_sites = 1;
    optm.max_sites = 1;

    // Set up the problem size and select the dmrg sites
    optm.max_problem_size = settings::strategy::dmrg_max_prob_size;
    optm.chosen_sites     = tensors.active_sites;
    optm.problem_dims     = state.active_problem_dims();
    optm.problem_size     = state.active_problem_size();

    auto initial_state = opt::get_opt_initial_mps(tensors, optm);
    auto opt_state     = opt::get_updated_state(tensors, initial_state, optm); // Runs the eigsolver for 1 iteration

    // We now have an mps site tensor with A = mix([A, P]).
    auto &N0_opt = opt_state.get_tensor();

    auto config = svd::config(bcfg.bond_lim, bcfg.trnc_lim);
    auto solver = svd::solver(config);
    if(posP < pos0) {
        // auto                     dL = mpsP.dimensions();
        auto                     dR = mps0.dimensions();
        Eigen::Tensor<Scalar, 2> N2 = N0_opt.shuffle(std::array{1, 0, 2}).reshape(std::array{dR[1], dR[0] * dR[2]});
        auto [U, S, V] = solver.decompose_multisite(N0_opt, 1l, mps0.spin_dim(), mps0.get_chiL(), mps0.get_chiR());

        // tenx::normalize(S);

        mps0.set_M(V);
        mps0.stash_C(S, -1.0, posP);
        mps0.stash_U(U, posP);
        mpsP.take_stash(mps0);

        dR                          = mps0.dimensions();
        auto                     dU = U.dimensions();
        auto                     dV = V.dimensions();
        Eigen::Tensor<Scalar, 2> U2 = U.reshape(std::array{dU[0] * dU[1], dU[2]});
        Eigen::Tensor<Scalar, 2> V2 = V.shuffle(std::array{1, 0, 2}).reshape(std::array{dV[1], dV[0] * dV[2]});
    } else {
        auto [U, S, V] = solver.decompose_multisite(N0_opt, mps0.spin_dim(), 1l, mps0.get_chiL(), mps0.get_chiR());
        mps0.set_M(U);
        mps0.set_LC(S, -1.0);
        mps0.stash_V(V, posP);
        mpsP.take_stash(mps0);
    }

    if constexpr(settings::debug) {
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        static constexpr auto eps     = std::numeric_limits<RealScalar>::epsilon();
        const auto            slack   = settings ::precision::max_norm_slack;
        auto                  normtol = eps * slack;
        mps0.assert_normalized(normtol);
        mpsP.assert_normalized(normtol);
    }
    tensors.activate_sites(active_sites_backup);
    rebuild_edges(state, model, edges);
};