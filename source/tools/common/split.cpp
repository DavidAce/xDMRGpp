#include "split.h"
#include "config/settings.h"
#include "contraction.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "log.h"
#include "math/linalg/tensor.h"
#include "math/svd.h"
#include "tensors/site/mps/MpsSite.h"
#include "tid/tid.h"
#include <deque>
#include <h5pp/h5pp.h>
#include <optional>

class value;
namespace settings {
    inline constexpr bool debug_split = false;
}

template<typename Scalar, typename T>
std::vector<MpsSite<Scalar>> tools::common::split::split_mps(const Eigen::Tensor<T, 3> &multisite_mps, const std::vector<long> &spin_dims,
                                                             const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg) {
    /*  Here we split multiple sites.
     *  Consider a case with 3 sites:
     *  spin_dims = {2,2,2}
     *  positions = {6,7,8}
     *  center_position = 6
     *
     *  then
     *
     *  chiL ---[mps]--- chiR
     *            |
     *         d^3=2*2*2
     * Becomes
     *
     *  chiL---[A6]---chi  chi---[LC]---chi   chi---[B7]---chi  chi---[L7]---chi   chi---[B8]---chiR
     *          |                                    |                                    |
     *         d=2                                  d=2                                  d=2
     *
     *
     * If instead we had
     *
     * spin_dims = {2,2,2}
     * positions = {3,4,5}
     * center_position = 4
     *
     * then we would get
     * chiL---[A3]---chi   chi---[L]---chi   chi---[A4]---chi  chi---[LC]---chi   chi---[B5]---chiR
     *         |                                    |                                    |
     *        d=2                                  d=2                                  d=2
     *
     * There is a special case where the multisite tensor has a single site. For instance, if moving left-to-right:
     *
     * spin_dims = {2}
     * positions = {3}
     * center_position = 4
     *
     * then we have the following one-site tensor
     * chiL---[A3]---chi    chi---[LC]---chi
     *         |
     *        d=2
     *
     * or, right-to-left:
     *
     * spin_dims = {2}
     * positions = {7}
     * center_position = 6
     *
     * then we have the following one-site tensor
     * chi---[LC]---chi  chiL---[B7]---chi
     *                           |
     *                          d=2
     *
     *
     *
     * The split is done in 2 steps. We start with.
     *
     * (1)chiL ---[mps]--- (2)chiR
     *             |
     *        (0) dL*dR
     *
     * where the index order is in parentheses. This is reinterpreted as
     *
     *
     * (2)chiL---[    mps    ]---(3)chiR
     *            |         |
     *         (0)dL     (1)dR
     *
     * Here dL=d*d*d... as many as there are A-matrices,
     * and dR=d*d*d*... as many as there are B-matrices.
     *
     * Then the mps is reshuffled into
     *
     * (1)chiL---[    mps    ]---(3)chiR
     *            |         |
     *         (0)dL     (2)dR
     *
     * which then is split up into
     *
     * chiL ---[U]----chi chi---[S]---chi chi---[V]---chiR
     *          |                                |
     *         dL                                dR
     *
     * The U and V tensors contain the sites to the left and right of the center bond,
     * and are sent away for further splitting.
     *
     *
     */
    if constexpr(sfinae::is_std_complex_v<T>) {
        auto chiL = multisite_mps.dimension(1);
        auto chiR = multisite_mps.dimension(2);
        if(chiL * chiR > 512 * 512 and tenx::isReal(multisite_mps)) {
            auto t_real = tid::tic_scope("isReal", tid::level::highest);
            tools::log->debug("split_mps: converting to real: chiL({}) * chiR({}) > 512 * 512", chiL, chiR);
            using RealScalar                                = typename Eigen::NumTraits<T>::Real;
            Eigen::Tensor<RealScalar, 3> multisite_mps_real = multisite_mps.real();
            return split_mps<Scalar>(multisite_mps_real, spin_dims, positions, center_position, svd_cfg);
        }
    }

    if(positions.empty()) throw except::runtime_error("Could not split multisite tensor: positions list is empty");
    if(multisite_mps.size() == 0) throw except::runtime_error("Could not split multisite tensor: tensor is empty");
    if(spin_dims.empty()) throw except::runtime_error("Could not split multisite tensor: spin_dims list is empty");
    if(spin_dims.size() != positions.size())
        throw except::runtime_error("Could not split multisite tensor: size mismatch in given lists: spin_dims {} != positions {} -- sizes not equal",
                                    spin_dims, positions);

    auto t_split = tid::tic_scope("split", tid::level::highest);

    // Set up the svd settings if not given explicitly
    svd_cfg                   = svd_cfg.value_or(svd::config());
    svd_cfg->truncation_limit = svd_cfg->truncation_limit.value_or(settings::precision::svd_truncation_min);
    svd_cfg->switchsize_gesdd = svd_cfg->switchsize_gesdd.value_or(settings::precision::svd_switchsize_bdc);
    svd_cfg->svd_save         = svd_cfg->svd_save.value_or(settings::precision::svd_save_fail ? svd::save::FAIL : svd::save::NONE);

    // Define dimensions and positions after the desired split
    long chiL = multisite_mps.dimension(1);
    long chiR = multisite_mps.dimension(2);
    long dL   = 1;
    long dR   = 1;

    // Split the multisite tensor at the given center position.
    std::vector<size_t> positions_left, positions_right;
    std::vector<long>   spin_dims_left, spin_dims_right;

    auto pos_it = positions.begin();
    auto dim_it = spin_dims.begin();
    while(pos_it != positions.end() and dim_it != spin_dims.end()) {
        if(std::cmp_less_equal(*pos_it, center_position)) {
            positions_left.emplace_back(*pos_it);
            spin_dims_left.emplace_back(*dim_it);
            dL *= *dim_it;
        } else {
            positions_right.emplace_back(*pos_it);
            spin_dims_right.emplace_back(*dim_it);
            dR *= *dim_it;
        }
        pos_it++;
        dim_it++;
    }

    if(positions.size() == 1) {
        auto t_short = tid::tic_scope("short", tid::level::highest);
        auto svd     = svd::solver(svd_cfg);
        // We can take a shortcut when given a multisite_mps with just a single site
        auto d0  = multisite_mps.dimension(0);
        auto d1  = multisite_mps.dimension(1);
        auto d2  = multisite_mps.dimension(2);
        auto pos = positions.front();
        if(dL == spin_dims.front()) {
            auto [U3, S1, V3] = svd.schmidt_multisite(multisite_mps, d0, 1, d1, d2, svd_cfg.value());
            if(static_cast<long>(pos) == center_position) {
                auto mps_site = MpsSite<Scalar>(U3, pos, "AC"); // Single site AC = USV -> A = U, C = S, stash V and the truncation error is on LC.
                mps_site.set_LC(S1, svd.get_truncation_error());
                mps_site.stash_V(V3, pos + 1);
                return {mps_site};
            } else {
                auto mps_site = MpsSite<Scalar>(U3, pos, "A"); // Single site
                mps_site.stash_S(S1, svd.get_truncation_error(), pos + 1);
                mps_site.stash_V(V3, pos + 1);
                return {mps_site};
            }

        } else if(dR == spin_dims.front()) {
            auto [U3, S1, V3] = svd.schmidt_multisite(multisite_mps, 1, d0, d1, d2, svd_cfg.value());
            auto mps_site     = MpsSite<Scalar>(V3, positions.front(), "B"); // Single site
            if(static_cast<long>(pos) == center_position + 1) {
                mps_site.stash_C(S1, svd.get_truncation_error(), static_cast<size_t>(center_position)); // The site to the left is a center, S1 belongs to it
            } else {
                mps_site.stash_S(S1, svd.get_truncation_error(), pos - 1); // The site to the left is a B, S1 belongs to it
            }
            mps_site.stash_U(U3, pos - 1);
            return {mps_site};
        }
    }

    // Initialize the new mps_sites
    std::vector<MpsSite<Scalar>> mps_sites_As;
    std::deque<MpsSite<Scalar>>  mps_sites_Bs;
    // Initialize the left/right unitary matrices
    //    Eigen::Tensor<Scalar, 3> U, V;
    //    Eigen::Tensor<Scalar, 1> S;

    // Now there are 5 options
    // 1) All sites will become A-sites
    // 2) All sites will become B-sites
    // 3) One A and one B-site
    // 4) Mixture of A and B-sites, with more (or eq) A's than B's
    // 5) Mixture of A and B-sites, with more B's than A's

    if constexpr(settings::debug_split)
        tools::log->trace("split: positions {} | L {} | R {} | C {} | spins {} | spinL {} | spinR {} ", positions, positions_left, positions_right,
                          center_position, spin_dims, spin_dims_left, spin_dims_right);

    if(positions.size() == positions_left.size()) {
        if constexpr(settings::debug_split) tools::log->trace("split: option 1");
        auto t_o1    = tid::tic_scope("o1", tid::level::highest);
        mps_sites_As = internal::split_mps_into_As<Scalar>(multisite_mps, spin_dims_left, positions_left, center_position, svd_cfg.value());
        // Remnant stash
        auto &mps     = mps_sites_As.back();
        auto &S_stash = mps.get_S_stash();
        auto  pos     = mps.template get_position<long>();
        if(pos == center_position and S_stash) { // Steal LC from stash
            mps.set_LC(S_stash->data, S_stash->error);
            S_stash.reset();
        } else {
            auto &V_stash = mps.get_V_stash();
            if(V_stash) {
                auto vdim = V_stash->data.dimensions();
                if(vdim[0] * vdim[1] > vdim[2]) tools::log->error("V_stash with dimensions {} for pos {} is not a diagonal matrix!", vdim, V_stash->pos_dst);
            }
        }

    } else if(positions.size() == positions_right.size()) {
        if constexpr(settings::debug_split) tools::log->trace("split: option 2 - sites {} become B's", positions);
        auto t_o2    = tid::tic_scope("o2", tid::level::highest);
        mps_sites_Bs = internal::split_mps_into_Bs<Scalar>(multisite_mps, spin_dims_right, positions_right, center_position, svd_cfg.value());
        // Take care of stash
        auto &mps     = mps_sites_Bs.front();
        auto &U_stash = mps.get_U_stash();
        auto &S_stash = mps.get_S_stash();
        auto  pos     = mps.template get_position<long>();
        if constexpr(settings::debug_split) {
            if(S_stash) {
                if(not tenx::isReal(S_stash->data)) throw except::runtime_error("S_stash is not real:\n{}", linalg::tensor::to_string(S_stash->data, 16, 18));
                if(not tenx::isPositive(S_stash->data))
                    throw except::runtime_error("S_stash is not positive:\n{}", linalg::tensor::to_string(S_stash->data, 16, 18));
            }
        }

        if(pos == center_position + 1 and S_stash) { // The stash in S becomes the LC for the site on the left
            mps.stash_C(S_stash->data, S_stash->error, safe_cast<size_t>(center_position));
            S_stash.reset();
        } else if(U_stash) {
            auto udim = U_stash->data.dimensions();
            if(udim[1] < udim[0] * udim[2]) tools::log->error("U_stash with dimensions {} for pos {} is not a diagonal matrix!", udim, U_stash->pos_dst);
        }
    } else if(positions.size() == 2 and positions_left.size() == 1 and positions_right.size() == 1) {
        if constexpr(settings::debug_split) tools::log->trace("split: option 3");
        auto t_o3 = tid::tic_scope("o3", tid::level::highest);
        // Set up the SVD
        svd::solver svd(svd_cfg);
        auto [U, S, V] = svd.schmidt_multisite(multisite_mps, dL, dR, chiL, chiR);
        mps_sites_As.emplace_back(U, positions.front(), "AC");
        mps_sites_As.back().set_LC(S, svd.get_truncation_error());
        mps_sites_Bs.emplace_back(V, positions.back(), "B");

    } else if(positions_left.size() >= positions_right.size()) {
        if constexpr(settings::debug_split) tools::log->trace("split: option 4");
        auto t_o4    = tid::tic_scope("o4", tid::level::highest);
        mps_sites_As = internal::split_mps_into_As<Scalar>(multisite_mps, spin_dims_left, positions_left, center_position, svd_cfg.value());
        // We expect stashed S and V. Merge these and send onward to B's
        auto &V_stash = mps_sites_As.back().get_V_stash();
        auto &S_stash = mps_sites_As.back().get_S_stash();
        if(V_stash and S_stash) {
            auto V = Eigen::Tensor<Scalar, 3>();
            if constexpr(std::is_same_v<Scalar, fp64>)
                V = tools::common::contraction::contract_bnd_mps(S_stash->data, V_stash->data).real();
            else
                V = tools::common::contraction::contract_bnd_mps(S_stash->data, V_stash->data);
            V_stash.reset();
            S_stash.reset();
            mps_sites_Bs = internal::split_mps_into_Bs<Scalar>(V, spin_dims_right, positions_right, center_position, svd_cfg.value());
            // Remnant S is now an LC for the last A
            auto &LC_stash = mps_sites_Bs.front().get_S_stash();
            auto &mpsA     = mps_sites_As.back();
            mpsA.set_LC(LC_stash->data, LC_stash->error);
            LC_stash.reset();
            // Remnant U is merged back into A
            mpsA.take_stash(mps_sites_Bs.front());
        }
    } else if(positions_left.size() < positions_right.size()) {
        if constexpr(settings::debug_split) tools::log->trace("split: option 5");
        auto t_o5    = tid::tic_scope("o5", tid::level::highest);
        mps_sites_Bs = internal::split_mps_into_Bs<Scalar>(multisite_mps, spin_dims_right, positions_right, center_position, svd_cfg.value());
        // We expect stashed U and S. Merge these and send onward to A's
        auto &U_stash = mps_sites_Bs.front().get_U_stash();
        auto &S_stash = mps_sites_Bs.front().get_S_stash();
        if(U_stash and S_stash) {
            auto U = Eigen::Tensor<Scalar, 3>();
            if constexpr(std::is_same_v<Scalar, fp64>)
                U = tools::common::contraction::contract_mps_bnd(U_stash->data, S_stash->data).real();
            else
                U = tools::common::contraction::contract_mps_bnd(U_stash->data, S_stash->data);

            U_stash.reset();
            S_stash.reset();
            mps_sites_As = internal::split_mps_into_As<Scalar>(U, spin_dims_left, positions_left, center_position, svd_cfg.value());
            // Remnant S is now an LC for the last A
            auto &LC_stash = mps_sites_As.back().get_S_stash();
            auto &mpsA     = mps_sites_As.back();
            mpsA.set_LC(LC_stash->data, LC_stash->error);
            LC_stash.reset();
            mps_sites_Bs.front().take_stash(mpsA);
        }
    }

    if(mps_sites_As.empty() and mps_sites_Bs.empty()) throw except::runtime_error("Got empty mps_sites from both left and right");

    // Sanity checks
    if(mps_sites_As.size() != spin_dims_left.size())
        throw except::runtime_error(
            fmt::format("Could not split multisite tensor: Got mps_sites_As.size() {} != spins_dims_left {}", mps_sites_As.size(), spin_dims_left.size()));
    if(mps_sites_Bs.size() != spin_dims_right.size())
        throw except::runtime_error(
            fmt::format("Could not split multisite tensor: Got mps_sites_Bs.size() {} != spins_dims_right {}", mps_sites_Bs.size(), spin_dims_right.size()));

    // Move the right side onto the left side (This is equivalent to std::list::splice)
    mps_sites_As.insert(mps_sites_As.end(), std::make_move_iterator(mps_sites_Bs.begin()), std::make_move_iterator(mps_sites_Bs.end()));

    for(const auto &&[idx, mps] : iter::enumerate(mps_sites_As)) {
        auto pos = positions[idx];
        if(not mps.is_at_position(pos)) {
            throw except::runtime_error("Could not split multisite tensor: Position mismatch: expected site {} != mps pos {} | positions to merge {}", pos,
                                        mps.get_position(), positions);
        }
        if(mps.is_at_position(center_position) and not mps.isCenter()) {
            throw except::runtime_error("Could not split multisite tensor: Specified center position {} did not become a center MPS", mps.get_position());
        }
    }

    // Return the all the positions including the bond matrix
    return mps_sites_As;
}
/* clang-format off */
template std::vector<MpsSite<cx64>>  tools::common::split::split_mps(const Eigen::Tensor<fp32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx64>>  tools::common::split::split_mps(const Eigen::Tensor<fp64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx64>>  tools::common::split::split_mps(const Eigen::Tensor<fp128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx64>>  tools::common::split::split_mps(const Eigen::Tensor<cx32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx64>>  tools::common::split::split_mps(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx64>>  tools::common::split::split_mps(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);

template std::vector<MpsSite<cx128>>  tools::common::split::split_mps(const Eigen::Tensor<fp32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx128>>  tools::common::split::split_mps(const Eigen::Tensor<fp64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx128>>  tools::common::split::split_mps(const Eigen::Tensor<fp128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx128>>  tools::common::split::split_mps(const Eigen::Tensor<cx32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx128>>  tools::common::split::split_mps(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
template std::vector<MpsSite<cx128>>  tools::common::split::split_mps(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);
/* clang-format on */

template<typename Scalar, typename T>
std::vector<MpsSite<Scalar>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<T, 3> &multisite_mps, const std::vector<long> &spin_dims,
                                                                               const std::vector<size_t> &positions, long center_position,
                                                                               svd::config &svd_cfg) {
    /*  Here we split multiple sites from the left.
     *  Consider a case with 3 sites and
     *  spin_dims = {2,2,2}
     *  sites = {6,7,8}
     *
     *  then
     *
     *  chiL ---[mps]--- chiR
     *            |
     *         d^3=2*2*2
     * Becomes
     *
     *  chiL---[A6]---chi  chi---[L7]---chi   chi---[A7]---chi  chi---[L8]---chi   chi---[A8]---chiR
     *          |                                    |                                    |
     *         d=2                                  d=2                                  d=2
     *
     *
     * By convention, a multisite_tensor is created by merging sites from left to right, that is
     *
     * 1-2-3-4-5
     * 12-3-4-5
     * 123-4-5
     * 1234-5
     * 12345
     *
     * Here we do the process in reverse order, i.e.
     *
     * 12345
     * 1234-5
     * 123-4-5
     * 12-3-4-5
     * 1-2-3-4-5
     *
     * Note that the spin dimensions are given in left-to-right order, i.e. 12345
     */

    //    if(spin_dims.empty()) throw except::runtime_error("Could not split multisite tensor from the left: spin_dims list is empty");
    if(spin_dims.size() != positions.size())
        throw except::runtime_error("Could not split multisite tensor from the left: size mismatch in given lists: spin_dims {} != sites {} -- sizes not equal",
                                    spin_dims, positions);

    // A special case is when we do one-site tensors. Then we expect
    // this function to receive a "U" without sites in it ( U*S will become a center bond).
    if(positions.empty()) return {};
    auto t_to_a = tid::tic_scope("to_a", tid::level::highest);
    // Initialize the resulting container of split sites
    std::vector<MpsSite<Scalar>> mps_sites;

    // Now we have multiple spin dimensions

    // Set up the SVD
    svd::solver svd(svd_cfg);

    // Declare the tensors that will catch the schmidt (SVD) decompositions
    Eigen::Tensor<T, 3>                U;                            // This will become the first site to be extracted
    Eigen::Tensor<T, 1>                S;                            // The singular values
    Eigen::Tensor<T, 3>                V            = multisite_mps; // This side contains all the remaining sites
    std::optional<Eigen::Tensor<T, 1>> S_prev       = std::nullopt;  // Starts out empty, carries the schmidt values from the previous iteration
    double                             S_prev_error = -1.0;          // Truncation error from the previous iteration
    for(const auto &[idx, spin_dim] : iter::enumerate(spin_dims)) {
        /* The schmidt decomposition gives us the 3 matrices to the right of the line |:
         *                       |
         *                       |
         *  chi---[S_prev]---chi | chiL ---[U]----chi chi---[S]---chi chi---[V]---chiR
         *                       |          |                                |
         *                       |          d                             d^(sites-1)
         *                       |
         * Here U is an "M" matrix of type A = Lambda * Gamma
         * See the notes on svd.schmidt at its definition
         *
         */

        if(S_prev) {
            auto t_sv = tid::tic_token("contract_sv", tid::level::highest);
            // Let V absorb S from the previous SVD (see note below)
            V = tools::common::contraction::contract_bnd_mps(S_prev.value(), V);
        }
        if(&spin_dim == &spin_dims.back()                          // Reached last site in spin_dims
           and spin_dim == V.dimension(0)                          // V has one site left
           and safe_cast<size_t>(center_position) > positions[idx] // Only needed when the site to the right is another A
        ) {
            // We reached the last site, and now V == L*GL, i.e. a theta with dim(0) == spin_dims.back().
            // Make sure we don't over-truncate during this SVD, so that we keep the bond dimension compatible with the adjacent site to the right.
            // In other words, we would like to retain the right bond dim of multisite_mps.
            svd.rank_min = V.dimension(2);
        }
        std::tie(U, S, V) = svd.schmidt_into_left_normalized(V, spin_dim);
        if(S.size() == 0) throw except::runtime_error("Could not split multisite tensor: Got 0 singular values from left svd");

        // Note: Now we have the three components
        // U:   A left-unitary "A" matrix which is a "Lambda * Gamma" in Vidal's notation
        // S:   A set of singular values, "L" matrix, belonging to the site on the right of this one (i.e. the next A to pop out of V)
        // V:   Contains one site less than it did before. That site is now in U.
        //      Before using V as the next multisite mps, it absorbs S so that the next U to pop out is a left-unitary A-type matrix.

        mps_sites.emplace_back(U, S_prev, positions[idx], S_prev_error, "A"); // S_prev is empty in the first iteration.

        // Store the singular values for the next iteration
        S_prev       = std::move(S);
        S_prev_error = svd.get_truncation_error();
    }

    // Now we have a series of A-A-A-A matrices and their corresponding L's
    // At the last step we have residual_norm S and V left over. Stash them!
    auto &mps = mps_sites.back();
    auto  pos = mps.get_position();
    if(S_prev) mps.stash_S(S_prev.value(), S_prev_error, pos + 1);
    mps.stash_V(V, pos + 1);
    return mps_sites;
}
/* clang-format off */
template std::vector<MpsSite<cx64>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<fp32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx64>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<fp64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx64>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<fp128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx64>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<cx32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx64>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx64>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);

template std::vector<MpsSite<cx128>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<fp32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx128>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<fp64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx128>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<fp128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx128>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<cx32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx128>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::vector<MpsSite<cx128>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);

/* clang-format on */

template<typename Scalar, typename T>
std::deque<MpsSite<Scalar>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<T, 3> &multisite_mps, const std::vector<long> &spin_dims,
                                                                              const std::vector<size_t> &positions, long center_position,
                                                                              svd::config &svd_cfg) {
    /*  Here we split multiple sites from the right
     *  Consider a case with 3 sites and
     *  spin_dims = {2,2,2}
     *  sites = {6,7,8}
     *
     *  then
     *
     *  chiL ---[mps]--- chiR
     *            |
     *         d^3=2*2*2
     * Becomes
     *
     *  chiL---[B6]---chi  chi---[L6]---chi   chi---[B7]---chi  chi---[L7]---chi   chi---[B8]---chiR
     *          |                                    |                                    |
     *         d=2                                  d=2                                  d=2
     *
     *
     * By convention, a multisite_tensor is created by merging sites from left to right, that is
     *
     * 1-2-3-4-5
     * 12-3-4-5
     * 123-4-5
     * 1234-5
     * 12345
     *
     * Here we do the process in reverse order, i.e.
     *
     * 12345
     * 1234-5
     * 123-4-5
     * 12-3-4-5
     * 1-2-3-4-5
     *
     * Note that the spin dimensions are given in left-to-right order, i.e. 12345
     */
    //    if(spin_dims.empty()) throw except::runtime_error("Could not split multisite tensor from the right: spin_dims list is empty");
    if(spin_dims.size() != positions.size())
        throw except::runtime_error(
            fmt::format("Could not split multisite tensor from the right: size mismatch in given lists: spin_dims {} != sites {} -- sizes not equal", spin_dims,
                        positions));

    // A special case is when we do one-site tensors. Then we expect
    // this function to receive a "V^dagger" without sites in it ( S*V^dagger will become a center bond).
    if(positions.empty()) return {};
    auto t_to_b = tid::tic_scope("to_b", tid::level::highest);

    // Initialize the resulting container of split sites
    std::deque<MpsSite<Scalar>> mps_sites;
    // Now we have multiple spin dimensions

    // Set up the SVD
    svd::solver svd(svd_cfg);

    // Declare the tensors that will catch the schmidt (SVD) decompositions
    Eigen::Tensor<T, 3>                U = multisite_mps;                   // This side contains all the sites
    Eigen::Tensor<T, 1>                S;                                   // The singular values
    Eigen::Tensor<T, 3>                V;                                   // This will become the first site extracted
    std::optional<Eigen::Tensor<T, 1>> S_prev       = std::nullopt;         // Starts out empty, needs to be checked outside this split
    double                             S_prev_error = -1.0;                 // Truncation error from the previous iteration
    for(const auto &[idx, spin_dim] : iter::enumerate_reverse(spin_dims)) { // Iterate in reverse order
        /* The schmidt decomposition gives us the 3 matrices to the left of the line |:
         *                                                      |
         * chiL ---[U]----chi chi---[S]---chi chi---[V]---chiR  |  chi---[S_prev]---chi
         *          |                                |          |
         *      d^(sites-1)                          d          |
         *
         * Here V is an "M" matrix of type B = Gamma * Lambda
         * See the notes on svd.schmidt at its definition
         */

        if(S_prev) {
            auto t_sv = tid::tic_token("contract_us", tid::level::highest);
            // Let U absorb S from the previous SVD (see note below)
            U = tools::common::contraction::contract_mps_bnd(U, S_prev.value());
        }
        if(&spin_dim == &spin_dims.front()                             // Reached the first site in spin_dims
           and spin_dim == U.dimension(0)                              // U has one site left
           and safe_cast<size_t>(center_position + 1) < positions[idx] // Only needed if the site to the left would be another B.
        ) {
            // We reached the first site, and now U == LG*L, i.e. a theta with dim(0) == spin_dims.front().
            // Make sure we don't over-truncate during this SVD, so that we keep the bond dimension compatible with the adjacent site to the left.
            // In other words, we would like to retain the left bond dim of multisite_mps.
            svd.rank_min = U.dimension(1);
        }
        std::tie(U, S, V) = svd.schmidt_into_right_normalized(U, spin_dim);
        if(S.size() == 0) throw except::runtime_error("Could not split multisite tensor: Got 0 singular values from right svd");

        // Note: Now we have the three components
        // U:   Contains one site less than the previous iteration.
        //      Before using U as the next multisite mps, it absorbs the S so that the next V to pop out is a right-unitary B-type matrix.
        // S:   A set of singular values, "L" matrix, belonging to the left site  (i.e. the next B to pop out of U)
        // V:   A right-unitary "B" matrix which is a "Gamma*Lambda" in Vidal's notation

        mps_sites.emplace_front(V, S_prev, positions[idx], S_prev_error, "B"); // S_prev is empty the first iteration.

        // Store the singular values for the next iteration
        S_prev       = std::move(S);
        S_prev_error = svd.get_truncation_error();
    }

    // Now we have a series of B-B-B-B matrices and their corresponding L's
    // At the last step we have residual_norm U and S left over. Stash them!
    auto &mps = mps_sites.front();
    auto  pos = mps.get_position();
    if(pos > 0) { // More left-normalized sites left in U
        if(S_prev) mps.stash_S(S_prev.value(), S_prev_error, pos - 1);
        mps.stash_U(U, pos - 1);
    }
    return mps_sites;
}
/* clang-format off */
template std::deque<MpsSite<cx64>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<fp32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx64>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<fp64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx64>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<fp128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx64>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<cx32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx64>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx64>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);

template std::deque<MpsSite<cx128>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<fp32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx128>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<fp64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx128>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<fp128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx128>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<cx32, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx128>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
template std::deque<MpsSite<cx128>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);

/* clang-format on */
