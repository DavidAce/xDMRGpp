#include "../../env.h"
#include "../BondExpansionConfig.h"
#include "../BondExpansionResult.h"
#include "config/debug.h"
#include "config/enums/AlgorithmType.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/linalg/matrix/to_string.h"
#include "math/num.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/contraction/contraction_policy.h"
#include "tools/common/log.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/mps.h"
#include <Eigen/Eigenvalues>

namespace settings {
    inline constexpr bool debug_edges = false;
}

template<typename Scalar>
void tools::finite::env::assert_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("assert_edges_var: fLBIT algorithm should never assert energy edges!");
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites, we shouldn't be asserting edges.
    // For instance, the active sites are cleared after a move of the center site.
    // We could always keep all edges refreshed, but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm to only allow edge assertions with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("assert_edges_ene: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.template get_position<long>();

    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_ene: pos {} | dir {} | "
                          "asserting edges eneL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);

    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &ene = edges.get_env_eneL(pos);
        if(pos == 0 and not ene.has_block()) throw except::runtime_error("ene L at pos {} does not have a block", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &ene_next = edges.get_env_eneL(pos + 1);
        ene_next.assert_unique_id(ene, mps, mpo);
    }
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_ene: pos {} | dir {} | "
                          "asserting edges eneR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);

    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &ene = edges.get_env_eneR(pos);
        if(pos == state.get_length() - 1 and not ene.has_block()) throw except::runtime_error("ene R at pos {} does not have a block", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &ene_prev = edges.get_env_eneR(pos - 1);
        ene_prev.assert_unique_id(ene, mps, mpo);
    }
}

template<typename Scalar>
void tools::finite::env::assert_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("assert_edges_var: fLBIT algorithm should never assert variance edges!");
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites, we shouldn't be asserting edges.
    // For instance, the active sites are cleared after a move of the center site.
    // We could always keep all edges refreshed, but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm to only allow edge assertions with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("assert_edges_var: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.template get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();

    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_var: pos {} | dir {} | "
                          "asserting edges varL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &var = edges.get_env_varL(pos);
        if(pos == 0 and not var.has_block()) throw except::runtime_error("var L at pos {} does not have a block", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &var_next = edges.get_env_varL(pos + 1);
        var_next.assert_unique_id(var, mps, mpo);
    }
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_var: pos {} | dir {} | "
                          "asserting edges varR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &var = edges.get_env_varR(pos);
        if(pos == state.get_length() - 1 and not var.has_block()) throw except::runtime_error("var R at pos {} does not have a block", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &var_prev = edges.get_env_varR(pos - 1);
        var_prev.assert_unique_id(var, mps, mpo);
    }
}

template<typename Scalar>
void tools::finite::env::assert_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) return;
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
}

template<typename Scalar>
void tools::finite::env::rebuild_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_ene: fLBIT algorithm should never rebuild energy edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(), edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto   t_reb   = tid::tic_scope("rebuild_edges_ene", tid::higher);
    size_t L       = state.template get_length<size_t>();
    size_t min_pos = 0;
    size_t max_pos = L - 1;

    // If there are no active sites then we can build up until current position
    /*
     * LOG:
     * - 2021-10-14:
     *      Just had a terribly annoying bug:
     *      Moving the center position clears active_sites, which caused problems when turning back from the right edge.
     *          1)  active_sites [A(L-1), AC(L)] are updated, left edge exist for A(L-1), right edge exists for AC(L)
     *          2)  move dir -1, clear active sites
     *          3)  assert_edges checks up to AC(L-1), but this site has a stale right edge.
     *      Therefore, one would have to rebuild edges between steps 2) and 3) to solve this issue
     *
     *      One solution would be to always rebuild edges up to the current position from both sides, but that would be
     *      wasteful. Instead, we could just accept that some edges are stale after moving the center-point,
     *      as long as we rebuild those when sites get activated again.
     *
     * - 2024-07-23
     *      Just found a way to calculate the optimal mixing factor for subspace expansion.
     *      In forward expansion we need H_eff including one site beyond active_sites.
     *      Therefore, we need to build more environments than we have needed previously.
     *      Examples:
     *          - Forward, direction == 1, active_sites = [5,6,7,8].
     *            Then we expand bond [8,9], and so we need envL[8] and envR[9].
     *          - Forward, direction == -1, active_sites = [5,6,7,8].
     *            Then we expand bond [4,5], and so we need envL[4] and envR[5].
     *      These environments weren't built before.
     *      Therefore, we must now rebuild
     *          - envL 0 to active_sites.back()
     *          - envR L to active_sites.front()
     */

    // If there are no active sites we shouldn't be rebuilding edges.
    // For instance, the active sites are cleared after a move of center site.
    // We could always keep all edges refreshed but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm, to only allow edge rebuilds with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_ene: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this rebuild after\n"
                                    " activating sites.");

    const long current_position = state.template get_position<long>();
    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    const size_t posL_active = edges.active_sites.back();
    const size_t posR_active = edges.active_sites.front();
    assert(posL_active < L && posL_active <= posR_active);
    assert(posR_active < L && posR_active >= posL_active);
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene: pos {} | dir {} | "
                          "inspecting edges eneL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    std::vector<size_t> env_pos_log;
    { // Seed left boundary

        auto &env0 = edges.get_env_eneL(0);
        env0.set_edge_dims(state.get_mps_site(0), model.get_mpo(0));
    }
    const size_t stopL = std::min(posL_active, L - 1);
    for(size_t pos = min_pos; pos < stopL; pos++) {
        const auto &env_here = edges.get_env_eneL(pos);
        auto       &env_rght = edges.get_env_eneL(pos + 1);
        auto        id_here  = env_here.get_unique_id();
        auto        id_rght  = env_rght.get_unique_id();
        env_rght.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }
    if constexpr(settings::debug_edges)
        if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene: rebuilt eneL edges: {}", env_pos_log);

    env_pos_log.clear();
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene: pos {} | dir {} | "
                          "inspecting edges eneR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    { // Seed right boundary (this is where set_edge_dims belongs)
        auto &envN = edges.get_env_eneR(L - 1);
        envN.set_edge_dims(state.get_mps_site(L - 1), model.get_mpo(L - 1));
    }
    const size_t stopR = std::min(posR_active, L - 1); // smallest valid envR index
    for(size_t pos = max_pos; pos > stopR; --pos) {
        const auto &env_here = edges.get_env_eneR(pos);
        auto       &env_left = edges.get_env_eneR(pos - 1);
        auto        id_here  = env_here.get_unique_id();
        auto        id_left  = env_left.get_unique_id();
        env_left.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    if constexpr(settings::debug_edges) {
        std::reverse(env_pos_log.begin(), env_pos_log.end());
        if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene: rebuilt eneR edges: {}", env_pos_log);
    }
    if(not edges.get_env_eneL(posL_active).has_block()) throw except::logic_error("rebuild_edges_ene: active env eneL has undefined block");
    if(not edges.get_env_eneR(posR_active).has_block()) throw except::logic_error("rebuild_edges_ene: active env eneR has undefined block");
}

template<typename Scalar>
void tools::finite::env::rebuild_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_var: fLBIT algorithm should never rebuild variance edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("rebuild_edges_var: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("rebuild_edges_var: All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto         t_reb   = tid::tic_scope("rebuild_edges_var", tid::level::higher);
    const size_t L       = state.template get_length<size_t>();
    const size_t min_pos = 0;
    const size_t max_pos = L - 1;

    // If there are no active sites, we shouldn't be rebuilding edges.
    // For instance, the active sites are cleared after a move of the center site.
    // We could always keep all edges refreshed, but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm to only allow edge rebuilds with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_var: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    const long current_position = state.template get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    const size_t posL_active = edges.active_sites.back();
    const size_t posR_active = edges.active_sites.front();
    assert(posL_active < L && posL_active <= posR_active);
    assert(posR_active < L && posR_active >= posL_active);
    if constexpr(settings::debug_edges) {
        tools::log->trace("rebuild_edges_var: pos {} | dir {} | "
                          "inspecting edges varL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    }

    std::vector<size_t> env_pos_log;
    { // Seed left boundary
        auto &env0 = edges.get_env_varL(0);
        env0.set_edge_dims(state.get_mps_site(0), model.get_mpo(0));
    }
    const size_t stopL = std::min(posL_active, L - 1);
    for(size_t pos = min_pos; pos < stopL; pos++) {
        const auto &env_here = edges.get_env_varL(pos);
        auto       &env_rght = edges.get_env_varL(pos + 1);
        auto        id_here  = env_here.get_unique_id();
        auto        id_rght  = env_rght.get_unique_id();
        env_rght.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }

    if constexpr(settings::debug_edges)
        if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var: rebuilt varL edges: {}", env_pos_log);
    env_pos_log.clear();
    if constexpr(settings::debug_edges) {
        tools::log->trace("rebuild_edges_var: pos {} | dir {} | "
                          "inspecting edges varR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    }
    { // Seed right boundary
        auto &envN = edges.get_env_varR(L - 1);
        envN.set_edge_dims(state.get_mps_site(L - 1), model.get_mpo(L - 1));
    }
    const size_t stopR = std::min(posR_active, L - 1); // smallest valid envR index
    for(size_t pos = max_pos; pos > stopR; --pos) {
        const auto &env_here = edges.get_env_varR(pos);
        auto       &env_left = edges.get_env_varR(pos - 1);
        auto        id_here  = env_here.get_unique_id();
        auto        id_left  = env_left.get_unique_id();
        env_left.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    if constexpr(settings::debug_edges) {
        std::reverse(env_pos_log.begin(), env_pos_log.end());
        if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var: rebuilt varR edges: {}", env_pos_log);
    }
    if(not edges.get_env_varL(posL_active).has_block()) throw except::logic_error("rebuild_edges_var: active env varL has undefined block");
    if(not edges.get_env_varR(posR_active).has_block()) throw except::logic_error("rebuild_edges_var: active env varR has undefined block");
}

template<typename Scalar>
void tools::finite::env::rebuild_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) return;
    [[maybe_unused]] auto assert_equal_block = [](const Eigen::Tensor<Scalar, 3> &blk1, const Eigen::Tensor<Scalar, 3> &blk2) {
        using Real = Eigen::NumTraits<Scalar>::Real;
        if(blk1.data() == nullptr and blk2.data() == nullptr) return;
        auto vec1 = tenx::VectorMap(blk1);
        auto vec2 = tenx::VectorMap(blk2);
        Real err  = (vec1 - vec2).norm() / vec1.norm();
        if(err > Real{1e-12f}) throw except::runtime_error("assert_equal_block: err {:.4e} > 1e-12", fp(err));
    };
    [[maybe_unused]] auto assert_equal_blkx2 = [](const x2::Tensor<Scalar, 3> &A, const x2::Tensor<Scalar, 3> &B) {
        using Real = Eigen::NumTraits<Scalar>::Real;
        if(A.dimensions() != B.dimensions()) throw except::runtime_error("A dims {} != B dims {}", A.dimensions(), B.dimensions());
        const auto n = A.size();

        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> Ahi(A.hi_data(), n);
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> Alo(A.lo_data(), n);
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> Bhi(B.hi_data(), n);
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> Blo(B.lo_data(), n);

        // Force temporaries so we do not keep maps alive across anything that may reallocate.
        const auto Asum = (Ahi + Alo).eval();
        const auto Bsum = (Bhi + Blo).eval();

        const Real denom = Asum.norm();
        const Real err   = (Asum - Bsum).norm() / (denom > Real(0) ? denom : Real(1));
        if(err > Real{1e-12f}) throw except::runtime_error("assert_equal_blkx2: err {:.4e} > 1e-12", fp(err));
    };

    [[maybe_unused]] auto assert_equal = [assert_equal_block, assert_equal_blkx2](const EdgesFinite<Scalar> &e1, const EdgesFinite<Scalar> &e2) {
        if(e1.eneL.size() != e2.eneL.size()) throw std::runtime_error("eneL size mismatch");
        if(e1.varL.size() != e2.varL.size()) throw std::runtime_error("varL size mismatch");
        if(e1.eneR.size() != e2.eneR.size()) throw std::runtime_error("eneR size mismatch");
        if(e1.varR.size() != e2.varR.size()) throw std::runtime_error("varR size mismatch");
        for(size_t i = 0; i < e1.eneL.size(); ++i) {
            assert_equal_blkx2(e1.eneL[i]->get_blkx2(), e2.eneL[i]->get_blkx2());
            assert_equal_block(e1.eneL[i]->get_block(), e2.eneL[i]->get_block());
        }
        for(size_t i = 0; i < e1.eneR.size(); ++i) {
            assert_equal_blkx2(e1.eneR[i]->get_blkx2(), e2.eneR[i]->get_blkx2());
            assert_equal_block(e1.eneR[i]->get_block(), e2.eneR[i]->get_block());
        }
        for(size_t i = 0; i < e1.varL.size(); ++i) {
            assert_equal_blkx2(e1.varL[i]->get_blkx2(), e2.varL[i]->get_blkx2());
            assert_equal_block(e1.varL[i]->get_block(), e2.varL[i]->get_block());
        }
        for(size_t i = 0; i < e1.varR.size(); ++i) {
            assert_equal_blkx2(e1.varR[i]->get_blkx2(), e2.varR[i]->get_blkx2());
            assert_equal_block(e1.varR[i]->get_block(), e2.varR[i]->get_block());
        }
    };
    // auto e1 = edges;
    // auto e2 = edges;
    // auto e3 = edges;
    // auto e4 = edges;
    //
    // {
    //     auto envinfo = SetEnvInfo(ContractionBackend::TBLIS);
    //     rebuild_edges_ene(state, model, e1);
    //     rebuild_edges_var(state, model, e1);
    // }
    // {
    //     auto envinfo = SetEnvInfo(ContractionBackend::X2);
    //     rebuild_edges_ene(state, model, e2);
    //     rebuild_edges_var(state, model, e2);
    // }
    // {
    //     auto envinfo = SetEnvInfo(ContractionBackend::TBLIS);
    //     rebuild_edges_ene_x2(state, model, e3);
    //     rebuild_edges_var_x2(state, model, e3);
    // }
    // {
    //     auto envinfo = SetEnvInfo(ContractionBackend::X2);
    //     rebuild_edges_ene_x2(state, model, e4);
    //     rebuild_edges_var_x2(state, model, e4);
    // }
    // auto envinfo = SetEnvInfo(ContractionBackend::TBLIS);

    rebuild_edges_ene(state, model, edges);
    rebuild_edges_var(state, model, edges);

    // assert_equal(e1, e2);
    // assert_equal(e1, e3);
    // assert_equal(e1, e4);
    // assert_equal(e1, edges);
}
