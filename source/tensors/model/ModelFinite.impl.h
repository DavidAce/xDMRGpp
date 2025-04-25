#pragma once
#include "config/settings.h"
#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include <array>
#include <complex>
#include <vector>

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL,
                                                           bool with_edgeR) const {
    // Observe that nbody empty/nullopt have very different meanings
    //      - empty means that no interactions should be taken into account, effectively setting all J(i,j...) = 0
    //      - nullopt means that we want the default mpo with (everything on)
    //      - otherwise nbody with values like {1,2} would imply we want 1 and 2-body interactions turned on
    //      - if nbody has a 0 value in it, it means we want to make an attempt to account for double-counting in multisite mpos.

    if(sites.empty()) throw std::runtime_error("No active sites on which to build a multisite mpo tensor");
    auto &cache = get_cache<T>();
    if(sites == active_sites and cache.multisite_mpo and not nbody) return cache.multisite_mpo.value();

    auto                nbody_str    = fmt::format("{}", nbody.has_value() ? nbody.value() : std::vector<size_t>{});
    auto                t_mpo        = tid::tic_scope("get_multisite_mpo", tid::level::highest);
    constexpr auto      shuffle_idx  = tenx::array6{0, 3, 1, 4, 2, 5};
    constexpr auto      contract_idx = tenx::idx({1}, {0});
    auto                positions    = num::range<size_t>(sites.front(), sites.back() + 1);
    auto                skip         = std::vector<size_t>{};
    auto                keep_log     = std::vector<size_t>();
    auto                skip_log     = std::vector<size_t>();
    bool                do_cache     = !with_edgeL and !with_edgeR and nbody.has_value() and nbody->back() > 1; // Caching doesn't make sense for nbody == 1
    auto               &threads      = tenx::threads::get();
    Eigen::Tensor<T, 4> multisite_mpo, mpoL, mpoR;
    Eigen::Tensor<T, 2> mpoR_traced;
    // The hamiltonian is the lower left corner he full system mpo chain, which we can extract using edgeL and edgeR
    Eigen::Tensor<T, 1> edgeL = get_mpo(sites.front()).template get_MPO_edge_left<T>();
    Eigen::Tensor<T, 1> edgeR = get_mpo(sites.back()).template get_MPO_edge_right<T>();

    tools::log->trace("Contracting multisite mpo tensor with sites {} | nbody {} ", sites, nbody_str);

    if(sites != positions) {
        for(const auto &pos : positions) {
            if(std::find(sites.begin(), sites.end(), pos) == sites.end()) skip.emplace_back(pos);
        }
    }

    for(const auto &pos : positions) {
        if constexpr(verbose_nbody) tools::log->trace("contracting position {}", pos);
        // sites needs to be sorted, but may skip sites.
        // For instance, sites == {3,9} is valid. Then sites 4,5,6,7,8 are skipped.
        // When a site is skipped, we set the contribution from its interaction terms to zero and trace over it so that
        // the physical dimension doesn't grow.
        bool do_trace = std::find(skip.begin(), skip.end(), pos) != skip.end();
        if(pos == positions.front()) {
            auto t_pre = tid::tic_scope("prepending", tid::level::highest);
            if(nbody or not skip.empty()) {
                multisite_mpo = get_mpo(pos).template MPO_nbody_view_as<T>(nbody, skip);
            } else {
                multisite_mpo = get_mpo(pos).template MPO_as<T>();
            }
            if(do_cache) {
                if(do_trace) {
                    skip_log.emplace_back(pos);
                } else {
                    keep_log.emplace_back(pos);
                }
            }
            if(with_edgeL and pos == positions.front()) {
                /* We can prepend edgeL to the first mpo to reduce the size of subsequent operations.
                 * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
                 *                        2               2
                 *                        |               |
                 *    0---[L]---(1)(0)---[M]---1 =  0---[LM]---1
                 *                        |               |
                 *                        3               3
                 */
                mpoL          = edgeL.reshape(tenx::array2{1, edgeL.size()}).contract(multisite_mpo, tenx::idx({1}, {0}));
                multisite_mpo = mpoL;
            }
            if(with_edgeR and pos == positions.back()) {
                /* This only happens when positions.size() == 1
                 * We can append edgeR to the last mpo to reduce the size of subsequent operations.
                 * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
                 *         2                              1                       2
                 *         |                              |                       |
                 *    0---[M]---1  0---[R]---1   =  0---[MR]---3  [shuffle]  0---[MR]---1
                 *         |                              |                       |
                 *         3                              2                       3
                 */
                auto mpoR_edgeR = Eigen::Tensor<T, 4>(multisite_mpo.contract(edgeR.reshape(tenx::array2{edgeR.size(), 1}), tenx::idx({1}, {0})));
                multisite_mpo   = mpoR_edgeR.shuffle(tenx::array4{0, 3, 1, 2});
            }
            continue;
        }

        mpoL = multisite_mpo;
        mpoR = nbody or not skip.empty() ? get_mpo(pos).template MPO_nbody_view_as<T>(nbody, skip) : get_mpo(pos).template MPO_as<T>();

        if(with_edgeR and pos == positions.back()) {
            /* We can append edgeL to the first mpo to reduce the size of subsequent operations.
             * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
             *         2                              1                       2
             *         |                              |                       |
             *    0---[M]---1  0---[R]---1   =  0---[MR]---3  [shuffle]  0---[MR]---1
             *         |                              |                       |
             *         3                              2                       3
             */
            auto mpoR_edgeR = Eigen::Tensor<T, 4>(mpoR.contract(edgeR.reshape(std::array<long, 2>{edgeR.size(), 1}), tenx::idx({1}, {0})));
            mpoR            = mpoR_edgeR.shuffle(tenx::array4{0, 3, 1, 2});
        }
        if constexpr(verbose_nbody) tools::log->trace("contracting position {} | mpoL {} | mpoR {}", pos, mpoL.dimensions(), mpoR.dimensions());

        // Determine if this position adds to the physical dimension or if it will get traced over
        long dim0     = mpoL.dimension(0);
        long dim1     = mpoR.dimension(1);
        long dim2     = mpoL.dimension(2) * (do_trace ? 1l : mpoR.dimension(2));
        long dim3     = mpoL.dimension(3) * (do_trace ? 1l : mpoR.dimension(3));
        auto new_dims = std::array<long, 4>{dim0, dim1, dim2, dim3};
        multisite_mpo.resize(new_dims);
        // Generate a unique cache string for the mpo that will be generated.
        // If there is a match for the string in cache, use the corresponding mpo, otherwise we make it.
        if(do_cache) {
            if(do_trace) {
                skip_log.emplace_back(pos);
            } else {
                keep_log.emplace_back(pos);
            }
        }
        auto new_cache_string = fmt::format("keep{}|skip{}|nbody{}|dims{}", keep_log, skip_log, nbody_str, new_dims);
        if(do_cache and cache.multisite_mpo_temps.find(new_cache_string) != cache.multisite_mpo_temps.end()) {
            if constexpr(debug_cache or verbose_nbody) tools::log->trace("cache hit: {}", new_cache_string);
            multisite_mpo = cache.multisite_mpo_temps.at(new_cache_string);
        } else {
            if constexpr(debug_cache or verbose_nbody) tools::log->trace("cache new: {}", new_cache_string);
            if(do_trace) {
                auto t_skip = tid::tic_scope("skipping", tid::level::highest);
                // Trace the physical indices of this skipped mpo (this should trace an identity)
                mpoR_traced = mpoR.trace(tenx::array2{2, 3});
                mpoR_traced *= mpoR_traced.constant(static_cast<T>(0.5)); // divide by 2 (after tracing identity)
                // Append it to the multisite mpo
                multisite_mpo.device(*threads->dev) = mpoL.contract(mpoR_traced, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2}).reshape(new_dims);
            } else {
                auto t_app                          = tid::tic_scope("appending", tid::level::highest);
                multisite_mpo.device(*threads->dev) = mpoL.contract(mpoR, contract_idx).shuffle(shuffle_idx).reshape(new_dims);
            }
            // This intermediate multisite_mpo_t could be the result we are looking for at a later time, so cache it!
            if(do_cache) cache.multisite_mpo_temps[new_cache_string] = multisite_mpo;
        }
    }
    if(with_edgeL) assert(multisite_mpo.dimension(0) == 1);
    if(with_edgeR) assert(multisite_mpo.dimension(1) == 1);
    return multisite_mpo;
}

template<typename Scalar>
template<typename T>
const Eigen::Tensor<T, 4> &ModelFinite<Scalar>::get_multisite_mpo() const {
    auto &cache = get_cache<T>();
    if(cache.multisite_mpo and cache.multisite_mpo_ids and not active_sites.empty()) {
        if constexpr(debug_cache) tools::log->trace("get_multisite_mpo: cache hit");
        // Check that the ids match
        auto active_ids = get_active_ids();
        if(active_ids != cache.multisite_mpo_ids)
            throw except::runtime_error("get_multisite_mpo: cache has mismatching ids: active ids: {} | cache ids {}", active_ids,
                                        cache.multisite_mpo_ids.value());
        return cache.multisite_mpo.value();
    }
    cache.multisite_mpo     = get_multisite_mpo<T>(active_sites);
    cache.multisite_mpo_ids = get_active_ids();
    return cache.multisite_mpo.value();
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const {
    auto                t_mpo = tid::tic_scope("mpo_shifted_view");
    Eigen::Tensor<T, 4> multisite_mpo, temp;
    constexpr auto      shuffle_idx  = tenx::array6{0, 3, 1, 4, 2, 5};
    constexpr auto      contract_idx = tenx::idx({1}, {0});
    auto               &threads      = tenx::threads::get();
    for(const auto &site : active_sites) {
        if(multisite_mpo.size() == 0) {
            multisite_mpo = get_mpo(site).template MPO_energy_shifted_view_as<T>(energy_per_site);
            continue;
        }
        const auto         &mpo      = get_mpo(site);
        long                dim0     = multisite_mpo.dimension(0);
        long                dim1     = mpo.MPO().dimension(1);
        long                dim2     = multisite_mpo.dimension(2) * mpo.MPO().dimension(2);
        long                dim3     = multisite_mpo.dimension(3) * mpo.MPO().dimension(3);
        std::array<long, 4> new_dims = {dim0, dim1, dim2, dim3};
        temp.resize(new_dims);
        temp.device(*threads->dev) =
            multisite_mpo.contract(mpo.template MPO_energy_shifted_view_as<T>(energy_per_site), contract_idx).shuffle(shuffle_idx).reshape(new_dims);
        multisite_mpo = std::move(temp);
    }
    return multisite_mpo;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const {
    if(sites.empty()) throw std::runtime_error("No active sites on which to build a multisite mpo squared tensor");
    auto &cache = get_cache<T>();
    if(sites == active_sites and cache.multisite_mpo_squared and not nbody) return cache.multisite_mpo_squared.value();
    tools::log->trace("Contracting multisite mpo² tensor with sites {}", sites);
    auto                t_mpo     = tid::tic_scope("get_multisite_mpo_squared", tid::level::highest);
    auto                positions = num::range<size_t>(sites.front(), sites.back() + 1);
    Eigen::Tensor<T, 4> multisite_mpo_squared;
    constexpr auto      shuffle_idx  = tenx::array6{0, 3, 1, 4, 2, 5};
    constexpr auto      contract_idx = tenx::idx({1}, {0});
    tenx::array4        new_dims;
    Eigen::Tensor<T, 4> temp;
    bool                first   = true;
    auto               &threads = tenx::threads::get();
    for(const auto &pos : positions) {
        // sites needs to be sorted, but may skip sites.
        // For instance, sites == {3,9} is valid. Then sites 4,5,6,7,8 are skipped.
        // When a site is skipped, we set the contribution from its interaction terms to zero and trace over it so that
        // the physical dimension doesn't grow.

        auto nbody_local = nbody;
        bool skip        = std::find(sites.begin(), sites.end(), pos) == sites.end();
        if(skip) nbody_local = std::vector<size_t>{};
        const auto &mpo = get_mpo(pos);
        if(first) {
            if(nbody_local)
                multisite_mpo_squared = mpo.template MPO2_nbody_view_as<T>(nbody_local);
            else
                multisite_mpo_squared = mpo.template MPO2_as<T>();
            first = false;
            continue;
        }

        decltype(auto) MPO  = mpo.template MPO_as<T>();
        decltype(auto) MPO2 = mpo.template MPO2_as<T>();
        long           dim0 = multisite_mpo_squared.dimension(0);
        long           dim1 = MPO2.dimension(1);
        long           dim2 = multisite_mpo_squared.dimension(2) * MPO2.dimension(2);
        long           dim3 = multisite_mpo_squared.dimension(3) * MPO2.dimension(3);
        new_dims            = {dim0, dim1, dim2, dim3};
        temp.resize(new_dims);
        if(nbody_local) // Avoids creating a temporary
            temp.device(*threads->dev) =
                multisite_mpo_squared.contract(mpo.template MPO2_nbody_view_as<T>(nbody_local), contract_idx).shuffle(shuffle_idx).reshape(new_dims);
        else
            temp.device(*threads->dev) = multisite_mpo_squared.contract(MPO2, contract_idx).shuffle(shuffle_idx).reshape(new_dims);

        if(skip) {
            /*! We just got handed a multisite-mpo created as
             *
             *       2   3                      2
             *       |   |                      |
             *  0 --[ mpo ]-- 1  --->    0 --[ mpo ]-- 1
             *       |   |                      |
             *       4   5                      3
             *
             *
             * In this step, a reshape brings back the 6 indices, and index 3 and 5 should be traced over.
             *
             */
            long                d0    = dim0;
            long                d1    = dim1;
            long                d2    = multisite_mpo_squared.dimension(2);
            long                d3    = MPO.dimension(2);
            long                d4    = multisite_mpo_squared.dimension(3);
            long                d5    = MPO.dimension(3);
            Eigen::Tensor<T, 4> temp2 = temp.reshape(tenx::array6{d0, d1, d2, d3, d4, d5}).trace(tenx::array2{3, 5});
            multisite_mpo_squared     = temp2 * temp2.constant(static_cast<T>(0.5));
        } else {
            multisite_mpo_squared = std::move(temp);
        }
    }
    return multisite_mpo_squared;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const {
    auto                multisite_mpo_shifted = get_multisite_mpo_shifted_view<T>(energy_per_site);
    long                dim0                  = multisite_mpo_shifted.dimension(0) * multisite_mpo_shifted.dimension(0);
    long                dim1                  = multisite_mpo_shifted.dimension(1) * multisite_mpo_shifted.dimension(1);
    long                dim2                  = multisite_mpo_shifted.dimension(2);
    long                dim3                  = multisite_mpo_shifted.dimension(3);
    std::array<long, 4> mpo_squared_dims      = {dim0, dim1, dim2, dim3};
    Eigen::Tensor<T, 4> multisite_mpo_squared_shifted(mpo_squared_dims);
    auto               &threads = tenx::threads::get();
    multisite_mpo_squared_shifted.device(*threads->dev) =
        multisite_mpo_shifted.contract(multisite_mpo_shifted, tenx::idx({3}, {2})).shuffle(tenx::array6{0, 3, 1, 4, 2, 5}).reshape(mpo_squared_dims);
    return multisite_mpo_squared_shifted;
}

template<typename Scalar>
template<typename T>
const Eigen::Tensor<T, 2> &ModelFinite<Scalar>::get_multisite_ham() const {
    auto &cache = get_cache<T>();
    if(cache.multisite_ham and not active_sites.empty()) return cache.multisite_ham.value();
    cache.multisite_ham = get_multisite_ham<T>(active_sites);
    return cache.multisite_ham.value();
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 2> ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const {
    /*! We use edges without particle content to get a hamiltonian for a subsystem (as opposed to an effective hamiltonian for a subsystem).
     *  To get an effective hamiltonian with particle-full edges, use TensorsFinite::get_effective_hamiltonian.
     */
    if(sites.empty()) throw std::runtime_error("No active sites on which to build a multisite hamiltonian tensor");
    auto &cache = get_cache<T>();
    if(sites == active_sites and cache.multisite_ham and not nbody) {
        if constexpr(debug_cache) tools::log->info("cache hit: sites{}|nbody{}", sites, nbody.has_value() ? nbody.value() : std::vector<size_t>{});
        return cache.multisite_ham.value();
    }
    tools::log->trace("get_multisite_ham(): Contracting effective Hamiltonian");
    long spin_dim = 1;
    for(const auto &pos : sites) { spin_dim *= get_mpo(pos).get_spin_dimension(); }
    auto dim2 = tenx::array2{spin_dim, spin_dim};
    if(sites.size() <= 4) {
        return get_multisite_mpo<T>(sites, nbody, true, true).reshape(dim2);
    } else {
        // When there are many sites, it is better to split sites into two equal chunks and then merge them (because edgeL/edgeR makes them small)
        auto half   = static_cast<long>((sites.size() + 1) / 2); // Let the left side take one more site in odd cases, because we contract from the left
        auto sitesL = std::vector<size_t>(sites.begin(), sites.begin() + half);
        auto sitesR = std::vector<size_t>(sites.begin() + half, sites.end());
        auto mpoL   = get_multisite_mpo<T>(sitesL, nbody, true, false); // Shuffle so we can use GEMM
        auto mpoR   = get_multisite_mpo<T>(sitesR, nbody, false, true);
        auto mpoLR  = tenx::gemm_mpo(mpoL, mpoR);
        return mpoLR.reshape(tenx::array2{spin_dim, spin_dim});
    }
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 2> ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const {
    /*! We use edges without particle content to get a hamiltonian for a subsystem.
     *  To get an effective hamiltonian with particle-full edges, use TensorsFinite::get_effective_hamiltonian.
     */
    if(sites.empty()) throw std::runtime_error("No active sites on which to build a multisite hamiltonian squared tensor");
    auto &cache = get_cache<T>();
    if(sites == active_sites and cache.multisite_ham_squared and not nbody) {
        tools::log->info("cache hit: sites{}|nbody{}", sites, nbody.has_value() ? nbody.value() : std::vector<size_t>{});
        return cache.multisite_ham_squared.value();
    }
    tools::log->trace("get_multisite_ham_squared(): Contracting effective Hamiltonian squared");
    long spin_dim = 1;
    for(const auto &pos : sites) { spin_dim *= get_mpo(pos).get_spin_dimension(); }
    auto dim2 = tenx::array2{spin_dim, spin_dim};
    if(sites.size() <= 4) {
        return get_multisite_mpo_squared<T>(sites, nbody).reshape(dim2);
    } else {
        // When there are many sites, it is better to split sites into two equal chunks and then merge them, especially if when taking all mpos.
        auto half   = static_cast<long>((sites.size() + 1) / 2); // Let the left side take one more site in odd cases, because we contract from the left
        auto sitesL = std::vector<size_t>(sites.begin(), sites.begin() + half);
        auto sitesR = std::vector<size_t>(sites.begin() + half, sites.end());
        auto mpo2L  = get_multisite_mpo_squared<T>(sitesL, nbody);
        auto mpo2R  = get_multisite_mpo_squared<T>(sitesR, nbody);
        auto mpo2LR = tenx::gemm_mpo(mpo2L, mpo2R);
        return mpo2LR.reshape(tenx::array2{spin_dim, spin_dim});
    }
}

template<typename Scalar>
template<typename T>
const Eigen::Tensor<T, 4> &ModelFinite<Scalar>::get_multisite_mpo_squared() const {
    auto &cache = get_cache<T>();
    if(cache.multisite_mpo_squared and cache.multisite_mpo_squared_ids and not active_sites.empty()) {
        if constexpr(debug_cache) tools::log->trace("get_multisite_mpo_squared: cache hit");
        // Check that the ids match
        auto active_ids_sq = get_active_ids_sq();
        if(active_ids_sq != cache.multisite_mpo_squared_ids)
            throw except::runtime_error("get_multisite_mpo_squared: cache has mismatching ids: active ids: {} | cache ids {}", active_ids_sq,
                                        cache.multisite_mpo_squared_ids.value());
        return cache.multisite_mpo_squared.value();
    }
    cache.multisite_mpo_squared     = get_multisite_mpo_squared<T>(active_sites);
    cache.multisite_mpo_squared_ids = get_active_ids_sq();
    return cache.multisite_mpo_squared.value();
}

template<typename Scalar>
template<typename T>
const Eigen::Tensor<T, 2> &ModelFinite<Scalar>::get_multisite_ham_squared() const {
    auto &cache = get_cache<T>();
    if(cache.multisite_ham_squared and not active_sites.empty()) return cache.multisite_ham_squared.value();
    cache.multisite_ham_squared = get_multisite_ham_squared<T>(active_sites);
    return cache.multisite_ham_squared.value();
}
