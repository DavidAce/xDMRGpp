#pragma once
#include "config/settings.h"
#include "debug/info.h"
#include "general/iter.h"
#include "math/cast.h"
#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "StateFinite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"

template<typename Scalar>
template<typename T>
StateFinite<Scalar>::StateFinite(const StateFinite<T> &other) noexcept {
    direction            = other.direction;
    tag_normalized_sites = other.tag_normalized_sites;
    name                 = other.name;
    algo                 = other.algo;
    convrates            = other.convrates;
    mps_sites.clear();
    mps_sites.reserve(other.mps_sites.size());
    for(const auto &mps_other : other.mps_sites) { mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(*mps_other)); }
    active_sites = other.active_sites;
    popcount     = other.popcount;

    if constexpr(std::is_same_v<Scalar, T>) {
        cache_fp32   = other.cache_fp32;
        cache_fp64   = other.cache_fp64;
        cache_fp128  = other.cache_fp128;
        cache_cx32   = other.cache_cx32;
        cache_cx64   = other.cache_cx64;
        cache_cx128  = other.cache_cx128;
        measurements = other.measurements;
    }
}

template<typename Scalar>
template<typename T>
StateFinite<Scalar> &StateFinite<Scalar>::operator=(const StateFinite<T> &other) noexcept {
    if constexpr(std::is_same_v<Scalar, T>) {
        if(this == &other) return *this; // check for self-assignment
    }
    direction            = other.direction;
    tag_normalized_sites = other.tag_normalized_sites;
    name                 = other.name;
    algo                 = other.algo;
    convrates            = other.convrates;
    mps_sites.clear();
    mps_sites.reserve(other.mps_sites.size());
    for(const auto &mps_other : other.mps_sites) { mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(*mps_other)); }
    active_sites = other.active_sites;
    popcount     = other.popcount;

    if constexpr(std::is_same_v<Scalar, T>) {
        cache_fp32   = other.cache_fp32;
        cache_fp64   = other.cache_fp64;
        cache_fp128  = other.cache_fp128;
        cache_cx32   = other.cache_cx32;
        cache_cx64   = other.cache_cx64;
        cache_cx128  = other.cache_cx128;
        measurements = other.measurements;
    }
    return *this;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 3> StateFinite<Scalar>::get_multisite_mps(const std::vector<size_t> &sites, bool use_cache) const {
    if(sites.empty()) throw except::runtime_error("No active sites on which to build a multisite mps tensor");
    auto                t_mps  = tid::tic_scope("gen_mps", tid::level::highest);
    auto                length = get_length<size_t>();
    Eigen::Tensor<T, 3> multisite_mps;
    for(auto &site : sites) {
        const auto mps_key = use_cache ? generate_cache_key(sites, site, "l2r") : "";
        const auto mps_cch = use_cache ? get_cached_mps<T>(mps_key) : std::nullopt;
        if(mps_cch.has_value()) {
            if constexpr(debug_cache) tools::log->trace("multisite_mps: cache_hit: {}", mps_key);
            multisite_mps = mps_cch->get();
        } else {
            const auto &mps       = get_mps_site(site);
            auto        M         = mps.template get_M_as<T>();
            bool        prepend_L = mps.get_label() == "B" and site > 0 and site == sites.front();
            bool        append_L  = mps.get_label() == "A" and site + 1 < length and site == sites.back();
            if(prepend_L) {
                // In this case all sites are "B" and we need to prepend the "L" from the site on the left to make a normalized multisite mps
                if constexpr(debug_state) tools::log->trace("Prepending L to B site {}", site);
                auto        t_prepend = tid::tic_scope("prepend", tid::level::higher);
                const auto &mps_left  = get_mps_site(site - 1); // mps_left is either AC or B
                const auto  L         = mps_left.isCenter() ? mps_left.template get_LC_as<T>() : mps_left.template get_L_as<T>();
                if(L.dimension(0) != M.dimension(1))
                    throw except::logic_error("get_multisite_mps<{}>: mismatching dimensions ({},{}): L (left) {} | M {}", mps_left.get_tag(), mps.get_tag(),
                                              sfinae::type_name<T>(), L.dimensions(), M.dimensions());
                M = tools::common::contraction::contract_bnd_mps(L, M);
            }
            if(append_L) {
                // In this case all sites are "A" and we need to append the "L" from the site on the right to make a normalized multisite mps
                if constexpr(debug_state) tools::log->trace("Appending L to A site {}", site);
                auto        t_append  = tid::tic_scope("append", tid::level::higher);
                const auto &mps_right = get_mps_site(site + 1);
                const auto  L         = mps_right.template get_L_as<T>();
                if(L.dimension(0) != M.dimension(2))
                    throw except::logic_error("get_multisite_mps<{}>: mismatching dimensions: M {} | L (right) {}", sfinae::type_name<T>(), M.dimensions(),
                                              L.dimensions());
                M = tools::common::contraction::contract_mps_bnd(M, L);
            }

            if(&site == &sites.front()) { // First site
                multisite_mps = std::move(M);
            } else { // Next sites
                multisite_mps = tools::common::contraction::contract_mps_mps(multisite_mps, M);
            }

            if(use_cache and not append_L) {
                // If it is the last site, we may have closed off by appending L
                get_cache<T>().mps.emplace_back(std::make_pair(mps_key, multisite_mps)); // We know it is not present already
                shrink_cache();
            }
        }
    }
    if constexpr(settings::debug) {
        // Check the norm of the tensor on debug builds
        auto t_dbg = tid::tic_scope("debug");
        auto norm  = std::abs(tools::common::contraction::contract_mps_norm(multisite_mps));
        if constexpr(debug_state) tools::log->trace("get_multisite_mps<{}>({}): norm ⟨ψ|ψ⟩ = {:.16f}", sfinae::type_name<T>(), sites, fp(norm));
        if(static_cast<fp64>(std::abs(norm - 1)) > settings::precision::max_norm_error) {
            tools::log->warn("get_multisite_mps<{}>({}): norm error |1-⟨ψ|ψ⟩| = {:.2e} > max_norm_error {:.2e}", sfinae::type_name<T>(), sites,
                             fp(std::abs(norm - 1)), settings::precision::max_norm_error);
            //                throw except::runtime_error("get_multisite_mps<fp64>({}): norm error |1-⟨ψ|ψ⟩| = {:.2e} > max_norm_error {:.2e}", sites,
            //                std::abs(norm - 1),
            //                                            settings::precision::max_norm_error);
        }
    }
    return multisite_mps;
    // }
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 2> StateFinite<Scalar>::get_reduced_density_matrix(const std::vector<size_t> &sites) const {
    auto t_rho        = tid::tic_scope("rho");
    auto cites        = num::range<size_t>(sites.front(), sites.back() + 1); // Contiguous list of all sites E.g. [012|6789] -> [012|345|6789]
    auto costs        = get_reduced_density_matrix_cost<T>(sites);
    auto min_cost_idx = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
    if constexpr(debug_density_matrix) tools::log->trace("get_reduced_density_matrix: cost_t2b {} | cost_l2r {} | cost_r2l {}", costs[0], costs[1], costs[2]);
    // if(debug::mem_hwm_in_mb() > 10000) throw except::runtime_error("Exceeded 5G high water mark after min_cost_idx");
    // min_cost_idx = 0; // Disable side to side contractions for a while
    if(min_cost_idx == 0 /* top to bottom */) {
        // We have a contiguous set
        auto mps = get_multisite_mps<T>(sites, true);
        // if(debug::mem_hwm_in_mb() > 10000) throw except::runtime_error("Exceeded 5G high water mark after multisite mps");
        return tools::common::contraction::contract_mps_partial<std::array{1l, 2l}>(mps);
    } else {
        // We probably have a non-contiguous set like [0123]4567[89]
        // Note that for non-contiguous sets, we expect the front and back sites to be at the edges!
        // Therefore, it matters from which side we do the contraction: we want the smaller of the two non-contiguous parts to accumulate
        // the middle part of the system. For instance, on [0123]4567[89] we want to contract from the right, so that [89] contracts 4567 and then [0123].
        // The reason is that
        // * [0123] with 8 free spin indices with dim (2**8)=256 when contracting the middle sites
        // * [89] with 4 free spin indices with dim(2**4)=16
        // So if subsystem A has N more spins than B, then A costs 2**(2*N) times more than B to compute.
        auto &threads   = tenx::threads::get();
        auto  rho_temp  = Eigen::Tensor<T, 4>(); // Will accumulate the sites
        auto  rho_temp2 = Eigen::Tensor<T, 4>();
        auto  M         = Eigen::Tensor<T, 3>();

        // Decide to go from the left or from the right
        // auto site_mean = std::accumulate(sites.begin(), sites.end(), 0.5) / static_cast<double>(sites.size());
        // bool from_left = site_mean >= get_length<double>() / 2.0;
        if(min_cost_idx == 1 /* left to right */) {
            // tools::log->info("from left");
            for(const auto &i : cites) {
                // tools::log->info("contracting site {}", i);
                const auto &mps = get_mps_site(i);
                if(i == sites.front()) {
                    // Could be an A, AC or B. Either way we need the first site to include the left schmidt values
                    // If it is the only site, we also need it to include the right schmidt values.
                    bool use_multisite = mps.get_label() == "B" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    rho_temp.resize(std::array{dim[0], dim[0], dim[2], dim[2]});
                    rho_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({1}, {1})).shuffle(std::array{0, 2, 1, 3});
                } else {
                    // This site could be A, AC or B. Only A lacks schmidt values on the right, so we use multisite when the last site is A.
                    bool use_multisite = i == sites.back() and mps.get_label() == "A";
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto rho_dim       = rho_temp.dimensions();
                    bool do_trace      = std::find(sites.begin(), sites.end(), i) == sites.end();
                    if(do_trace) {
                        auto new_dim = std::array{rho_dim[0], rho_dim[1], mps_dim[2], mps_dim[2]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {1})).contract(M, tenx::idx({2, 3}, {1, 0}));
                    } else {
                        auto new_dim = std::array{rho_dim[0] * mps_dim[0], rho_dim[1] * mps_dim[0], mps_dim[2], mps_dim[2]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {1}))
                                                              .contract(M, tenx::idx({2}, {1}))
                                                              .shuffle(std::array{0, 2, 1, 4, 3, 5})
                                                              .reshape(new_dim);
                    }
                    rho_temp = std::move(rho_temp2);
                    if constexpr(debug_cache) {
                        if(debug::mem_hwm_in_mb() > 10000) {
                            for(const auto &elem : get_cache<T>().mps) tools::log->info("from left: cache memory > 10000 MB: {}", elem.first);
                            // throw except::runtime_error("Exceeded 5G high water mark after rho l2r site {} | sites", i, cites);
                        }
                    }
                }
            }
        } else if(min_cost_idx == 2 /* left to right */) {
            // tools::log->info("from right");
            for(const auto &i : iter::reverse(cites)) {
                const auto &mps = get_mps_site(i);
                // tools::log->info("contracting site {}", i);
                if(i == sites.back()) {
                    // Could be an A, AC or B. Either way we need the last site to include the right schmidt values
                    // If it is the only site, we also need it to include the left schmidt values.
                    bool use_multisite = mps.get_label() == "A" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    rho_temp.resize(std::array{dim[0], dim[0], dim[1], dim[1]});
                    rho_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({2}, {2})).shuffle(std::array{0, 2, 1, 3});
                } else {
                    // This site could be A, AC or B. Only B lacks schmidt values on the left, so we use multisite when the first site is B.
                    bool use_multisite = i == sites.front() and mps.get_label() == "B";
                    M                  = use_multisite ? get_multisite_mps<T>({i}, true) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto rho_dim       = rho_temp.dimensions();
                    bool do_trace      = std::find(sites.begin(), sites.end(), i) == sites.end();
                    if(do_trace) {
                        auto new_dim = std::array{rho_dim[0], rho_dim[1], mps_dim[1], mps_dim[1]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {2})).contract(M, tenx::idx({2, 3}, {2, 0}));
                    } else {
                        auto new_dim = std::array{rho_dim[0] * mps_dim[0], rho_dim[1] * mps_dim[0], mps_dim[1], mps_dim[1]};
                        rho_temp2.resize(new_dim);
                        rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {2}))
                                                              .contract(M, tenx::idx({2}, {2}))
                                                              .shuffle(std::array{2, 0, 4, 1, 3, 5})
                                                              .reshape(new_dim);
                    }
                    rho_temp = std::move(rho_temp2);
                    if constexpr(debug_cache) {
                        if(debug::mem_hwm_in_mb() > 10000) {
                            for(const auto &elem : get_cache<T>().mps) tools::log->info("from right: cache memory > 10000 MB: {}", elem.first);
                            // throw except::runtime_error("Exceeded 5G high water mark after rho r2l site {} | sites", i, cites);
                        }
                    }
                }
            }
        }
        return rho_temp.trace(std::array{2, 3});
    }
}

template<typename Scalar>
template<typename T>
std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost(const std::vector<size_t> &sites) const {
    auto t_rho         = tid::tic_scope("rho");
    auto cites         = num::range<size_t>(sites.front(), sites.back() + 1); // Contiguous list of all sites E.g. [012|6789] -> [012|345|6789]
    bool is_contiguous = sites == cites;
    // We have a contiguous set
    // Calculate the numerical cost for contracting top to bottom or side to side. Account for both the number of operations and the memory size
    auto ops_t2b = 0.0, mem_t2b = 0.0;
    auto ops_l2r = 0.0, mem_l2r = 0.0;
    auto ops_r2l = 0.0, mem_r2l = 0.0;
    if(is_contiguous) {
        auto bonds   = get_bond_dims(sites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
        auto chiL    = get_mps_site(sites.front()).get_chiL();
        auto chiR    = get_mps_site(sites.back()).get_chiR();
        auto max_chi = std::max(chiR, chiL);
        auto min_chi = std::min(chiR, chiL);
        auto spindim = std::pow(2.0, sites.size());

        mem_t2b = spindim * spindim; // The number of elements in the largest object that will be held in memory
        ops_t2b = spindim * spindim * static_cast<double>(max_chi * min_chi * min_chi + min_chi); // This is the last step, but we will add earlier steps below.
        for(const auto &[i, pos] : iter::enumerate(sites)) {
            auto key = generate_cache_key(sites, pos, "l2r");
            if(has_cached_mps<T>(key)) continue;
            // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
            if(i == 0) { // It will append either chiL or chiR, but we take the worst case scenario here
                if(sites.size() == 1) { ops_t2b += static_cast<double>(2l * chiL * chiR * std::max(chiL, chiR)); }
            } else {
                auto bondR = pos == sites.back() ? chiR : bonds[i];
                ops_t2b += std::pow(2.0, i + 1) * static_cast<double>(chiL * bonds[i - 1] * bondR);
                mem_t2b = std::max(mem_t2b, std::pow(2.0, i + 1) * static_cast<double>(chiL * bondR));
            }
        }
    }

    if(!is_contiguous or static_cast<double>(sizeof(T)) * mem_t2b / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem_t2b = std::numeric_limits<double>::infinity();
        ops_t2b = std::numeric_limits<double>::infinity();
    }

    auto bonds   = get_bond_dims(cites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
    auto chiL    = get_mps_site(cites.front()).get_chiL();
    auto chiR    = get_mps_site(cites.back()).get_chiR();
    auto spindim = 1.0;
    for(const auto &[i, pos] : iter::enumerate(cites)) {
        // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
        auto bondR = pos == sites.back() ? chiR : bonds.at(i);
        if(pos == sites.front()) {
            spindim *= 2;
            ops_l2r += spindim * spindim * static_cast<double>(chiL * bondR * bondR);
            mem_l2r = std::max(mem_l2r, spindim * spindim * static_cast<double>(bondR * bondR));
        } else {
            bool do_trace = std::find(sites.begin(), sites.end(), pos) == sites.end();
            if(do_trace) {
                ops_l2r += spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bonds[i - 1] * bondR); // Upper
                ops_l2r += spindim * spindim * static_cast<double>(4l * bonds[i - 1] * bondR * bondR);        // Lower part 1 of 2
                ops_l2r += spindim * spindim * static_cast<double>(2l * bondR * bondR);                       // Lower part 2 of 2
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bondR);
                auto tmp2 = spindim * spindim * static_cast<double>(bondR * bondR);
                mem_l2r   = std::max(mem_l2r, tmp1 + tmp2);

            } else {
                ops_l2r += spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bonds[i - 1] * bondR); // Upper
                ops_l2r += spindim * spindim * static_cast<double>(4l * bonds[i - 1] * bondR * bondR);        // Lower part
                mem_l2r   = std::max(mem_l2r, spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bondR));
                mem_l2r   = std::max(mem_l2r, spindim * spindim * static_cast<double>(4l * bondR * bondR));
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bonds[i - 1] * bondR);
                auto tmp2 = spindim * spindim * static_cast<double>(4l * bondR * bondR);
                mem_l2r   = std::max(mem_l2r, tmp1 + tmp2);
                spindim *= 2;
            }
        }
    }
    ops_l2r += spindim * spindim * static_cast<double>(chiR); // add the last contraction that closes the density matrix
    mem_l2r = std::max(mem_l2r, spindim * spindim);
    if(static_cast<double>(sizeof(T)) * mem_l2r / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem_l2r = std::numeric_limits<double>::infinity();
        ops_l2r = std::numeric_limits<double>::infinity();
    }

    spindim = 1.0;
    for(const auto &[i, pos] : iter::enumerate_reverse(cites)) {
        // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
        auto bondL = pos == sites.front() ? chiL : bonds.at(i - 1);
        if(pos == sites.back()) {
            spindim *= 2;
            ops_r2l += spindim * spindim * static_cast<double>(chiR * bondL * bondL);
            mem_r2l = std::max(mem_r2l, spindim * spindim * static_cast<double>(bondL * bondL));
        } else {
            bool do_trace = std::find(sites.begin(), sites.end(), pos) == sites.end();
            if(do_trace) {
                ops_r2l += spindim * spindim * static_cast<double>(2l * bondL * bonds[i] * bonds[i]); // Upper
                ops_r2l += spindim * spindim * static_cast<double>(4l * bondL * bondL * bonds[i]);    // Lower part 1 of 2
                ops_r2l += spindim * spindim * static_cast<double>(2l * bondL * bondL);               // Lower part 2 of 2
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bondL * bonds[i]);
                auto tmp2 = spindim * spindim * static_cast<double>(bondL * bondL);
                mem_r2l   = std::max(mem_r2l, tmp1 + tmp2);
            } else {
                ops_r2l += spindim * spindim * static_cast<double>(2l * bondL * bonds[i] * bonds[i]); // Upper
                ops_r2l += spindim * spindim * static_cast<double>(4l * bondL * bondL * bonds[i]);    // Lower part 1 of 2
                auto tmp1 = spindim * spindim * static_cast<double>(2l * bondL * bonds[i]);
                auto tmp2 = spindim * spindim * static_cast<double>(4l * bondL * bondL);
                mem_r2l   = std::max(mem_r2l, tmp1 + tmp2);
                spindim *= 2;
            }
        }
    }
    ops_r2l += spindim * spindim * static_cast<double>(chiL); // add the last contraction that closes the density matrix
    mem_r2l = std::max(mem_r2l, spindim * spindim);
    if(static_cast<double>(sizeof(T)) * mem_r2l / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem_r2l = std::numeric_limits<double>::infinity();
        ops_r2l = std::numeric_limits<double>::infinity();
    }
    return std::array{ops_t2b + mem_t2b, ops_l2r + mem_l2r, ops_r2l + mem_r2l};
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 2> StateFinite<Scalar>::get_transfer_matrix(const std::vector<size_t> &sites, std::string_view side) const {
    auto  t_trf        = tid::tic_scope("trf");
    auto  chiL         = get_mps_site(sites.front()).get_chiL();
    auto  chiR         = get_mps_site(sites.back()).get_chiR();
    auto  costs        = get_transfer_matrix_costs<T>(sites, side);
    auto  min_cost_idx = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
    auto &threads      = tenx::threads::get();
    if constexpr(debug_transfer_matrix) tools::log->trace("cost_t2b {} | cost_s2s {} ({})", costs[0], costs[1], side);
    if(min_cost_idx == 0 /* top to bottom */) {
        if constexpr(debug_transfer_matrix) tools::log->trace("from top");
        auto mps                  = get_multisite_mps<T>(sites, true);
        auto dim                  = std::array{mps.dimension(1) * mps.dimension(2), mps.dimension(1) * mps.dimension(2)};
        auto res                  = Eigen::Tensor<T, 2>(dim);
        res.device(*threads->dev) = mps.conjugate().contract(mps, tenx::idx({0}, {0})).reshape(dim);
        return res;
    } else {
        auto trf_temp = Eigen::Tensor<T, 4>(); // Will accumulate the sites
        auto M        = Eigen::Tensor<T, 3>();
        auto trf_tmp4 = Eigen::Tensor<T, 4>(); // Scratch space for contractions
        auto trf_tmp5 = Eigen::Tensor<T, 5>(); // Scratch space for contractions

        /*
         * We accumulate the transfer matrix such that it has the same index ordering going from left or right,
         * so that the caches are compatible without having to transpose
         *  left     right
         * 0-----2  0-----2
         *    |        |
         * 1-----3  1-----3
         *
         */
        auto trf_cache = get_optimal_trf_from_cache<T>(sites, side);
        // auto side      = trf_cache.has_value() ? trf_cache->side : (min_cost_idx == 1 ? "l2r" : "r2l");

        if(side.starts_with('l') /* left to right */) {
            if constexpr(debug_transfer_matrix) tools::log->trace("from left");
            for(const auto &i : sites) {
                if(trf_cache.has_value() and i <= trf_cache->pos) {
                    if(i == trf_cache->pos) trf_temp = trf_cache->trf.get();
                    continue;
                }
                const auto &mps = get_mps_site(i);
                if(i == sites.front()) {
                    // Could be an A, AC or B. Either way we need the first site to include the left schmidt values
                    // If it is the only site, we also need it to include the right schmidt values.
                    bool use_multisite = mps.get_label() == "B" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    trf_temp.resize(std::array{dim[1], dim[1], dim[2], dim[2]});
                    trf_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).shuffle(std::array{0, 2, 1, 3});
                } else {
                    // This site could be A, AC or B. Only A lacks schmidt values on the right, so we use multisite when the last site is A.
                    bool use_multisite = i == sites.back() and mps.get_label() == "A";
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto trf_dim       = trf_temp.dimensions();
                    auto new_dim       = std::array{trf_dim[0], trf_dim[1], mps_dim[2], mps_dim[2]};
                    trf_tmp4.resize(new_dim);
                    trf_tmp4.device(*threads->dev) = trf_temp.contract(M.conjugate(), tenx::idx({2}, {1})).contract(M, tenx::idx({2, 3}, {1, 0}));
                    trf_temp                       = std::move(trf_tmp4);
                }
                save_trf_into_cache<T>(trf_temp, sites, i, "l2r");
            }
        } else if(side.starts_with('r') /* right to left */) {
            if constexpr(debug_transfer_matrix) tools::log->trace("from right");
            for(const auto &i : iter::reverse(sites)) {
                if(trf_cache.has_value() and i >= trf_cache->pos) {
                    if(i == trf_cache->pos) trf_temp = trf_cache->trf.get();
                    continue;
                }
                const auto &mps = get_mps_site(i);
                if(i == sites.back()) {
                    // Could be an A, AC or B. Either way we need the last site to include the right schmidt values
                    // If it is the only site, we also need it to include the left schmidt values.
                    bool use_multisite = mps.get_label() == "A" or sites.size() == 1;
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto dim           = M.dimensions();
                    trf_temp.resize(std::array{dim[1], dim[1], dim[2], dim[2]});
                    trf_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).shuffle(std::array{0, 2, 1, 3});

                    // auto dim = M.dimensions();
                    // trf_temp.resize(std::array{dim[2], dim[2], dim[1], dim[1]});
                    // trf_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).shuffle(std::array{1, 3, 0, 2});

                } else {
                    // This site could be A, AC or B. Only B lacks schmidt values on the left, so we use multisite when the first site is B.
                    bool use_multisite = i == sites.front() and mps.get_label() == "B";
                    M                  = use_multisite ? get_multisite_mps<T>({i}) : mps.template get_M_as<T>();
                    auto mps_dim       = M.dimensions();
                    auto trf_dim       = trf_temp.dimensions();
                    auto tm5_dim       = std::array{mps_dim[0], mps_dim[1], trf_dim[0], trf_dim[2], trf_dim[3]};
                    auto new_dim       = std::array{mps_dim[1], mps_dim[1], trf_dim[2], trf_dim[3]};
                    trf_tmp5.resize(tm5_dim);
                    trf_tmp5.device(*threads->dev) = M.contract(trf_temp, tenx::idx({2}, {1}));
                    trf_temp.resize(new_dim);
                    trf_temp.device(*threads->dev) = M.conjugate().contract(trf_tmp5, tenx::idx({0, 2}, {0, 2}));
                }
                save_trf_into_cache<T>(trf_temp, sites, i, "r2l");
            }
        }

        auto new_dim = std::array{chiL, chiR, chiL, chiR};
        trf_tmp4.resize(new_dim);
        trf_tmp4.device(*threads->dev) = trf_temp.shuffle(std::array{0, 2, 1, 3});
        return trf_tmp4.reshape(std::array{chiL * chiR, chiL * chiR});
    }
}

template<typename Scalar>
template<typename T>
double StateFinite<Scalar>::get_transfer_matrix_cost(const std::vector<size_t> &sites, std::string_view side,
                                                     const std::optional<TrfCacheEntry<T>> &trf_cache) const {
    auto   bonds = get_bond_dims(sites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
    auto   chiL  = get_mps_site(sites.front()).get_chiL();
    auto   chiR  = get_mps_site(sites.back()).get_chiR();
    double ops   = 0.0;
    double mem   = 0.0;
    if(side.starts_with('l')) {
        for(const auto &[i, pos] : iter::enumerate(sites)) {
            if(trf_cache.has_value() and side == trf_cache->side and pos <= trf_cache->pos) continue;
            // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
            if(pos == sites.front() and i < bonds.size()) {
                ops += static_cast<double>(chiL * chiL * bonds[i] * bonds[i] * 2); // The left-most site
                mem = std::max(mem, static_cast<double>(chiL * chiL * bonds[i] * bonds[i]));
            } else { // Appending sites in the interior
                auto bondR = pos == sites.back() ? chiR : bonds.at(i);
                ops += static_cast<double>(chiL * chiL * bonds[i - 1] * bonds[i - 1] * bondR * 2); // Upper mps
                ops += static_cast<double>(chiL * chiL * bonds[i - 1] * bondR * bondR * 4);        // Lower mps part 1 of 2
                ops += static_cast<double>(chiL * chiL * bondR * bondR * 2);                       // Lower mps part 2 of 2
                auto tmp1 = static_cast<double>(chiL * chiL * bonds[i - 1] * bondR * 2);
                auto tmp2 = std::max(tmp1, static_cast<double>(chiL * chiL * bondR * bondR * 4));
                auto tmp3 = std::max(tmp2, static_cast<double>(chiL * chiL * bondR * bondR));
                mem       = std::max(mem, tmp2 + tmp3);
            }
        }
    }
    if(side.starts_with('r')) {
        for(const auto &[i, pos] : iter::enumerate_reverse(sites)) {
            if(trf_cache.has_value() and side == trf_cache->side and pos >= trf_cache->pos) continue;
            // bonds[i] is always a bond directly to the right of pos, except for the last pos, where we use chiR instead
            if(pos == sites.back() and i - 1 < bonds.size()) {
                ops += static_cast<double>(chiR * chiR * bonds[i - 1] * bonds[i - 1] * 2); // The right-most site
                mem = std::max(mem, static_cast<double>(chiR * chiR * bonds[i - 1] * bonds[i - 1]));
            } else { // Appending sites in the interior
                auto bondL = pos == sites.front() ? chiL : bonds.at(i - 1);
                ops += static_cast<double>(chiR * chiR * bonds[i] * bonds[i] * bondL * 2); // Upper mps
                ops += static_cast<double>(chiR * chiR * bonds[i] * bondL * bondL * 4);    // Lower mps part 1 of 2
                ops += static_cast<double>(chiR * chiR * bondL * bondL * 2);               // Lower mps part 2 of 2
                auto tmp1 = static_cast<double>(chiR * chiR * bonds[i] * bondL * 2);
                auto tmp2 = std::max(tmp1, static_cast<double>(chiR * chiR * bondL * bondL * 4));
                auto tmp3 = std::max(tmp2, static_cast<double>(chiR * chiR * bondL * bondL));
                mem       = std::max(mem, tmp2 + tmp3);
            }
        }
    }
    if(static_cast<double>(sizeof(T)) * mem / std::pow(1024.0, 3.0) >= settings::precision::max_cache_gbts) {
        mem = std::numeric_limits<double>::infinity();
        ops = std::numeric_limits<double>::infinity();
    }
    return ops + mem;
}

template<typename Scalar>
template<typename T>
std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs(const std::vector<size_t> &sites, std::string_view side) const {
    if(sites.empty()) throw except::logic_error("get_transfer_matrix_cost: sites is empty");
    auto cites = num::range<size_t>(sites.front(), sites.back() + 1); // Contiguous list of all sites
    if(sites != cites) throw except::logic_error("get_transfer_matrix_cost: sites is not contiguous: {}", sites);
    // We have a contiguous set
    auto bonds   = get_bond_dims(sites); // One fewer than sites, unless there is only one site, and then this is the bond to the right of that site.
    auto chiL    = get_mps_site(sites.front()).get_chiL();
    auto chiR    = get_mps_site(sites.back()).get_chiR();
    auto spindim = std::pow(2.0, sites.size());
    // Calculate the numerical cost for contracting top to bottom or side to side
    auto ops_t2b = spindim * static_cast<double>(chiL * chiR * chiL * chiR); // This is the last step, but we will add earlier steps below.
    auto mem_t2b =
        static_cast<double>(chiL * chiR * chiL * chiR); // This is the number of elements in the last object we will compare with earlier steps below.
    for(size_t i = 0; i < sites.size(); ++i) {
        auto key = generate_cache_key(sites, sites.front() + i, "l2r");
        if(has_cached_mps<T>(key)) continue;
        if(i == 0 and sites.size() == 1) { // It will append either chiL or chiR but we take the worst case scenario here
            ops_t2b += static_cast<double>(2l * chiL * chiR * std::max(chiL, chiR));
            mem_t2b = std::max(mem_t2b, static_cast<double>(2l * chiL * chiR));
        }
        if(i + 1 < bonds.size()) { // TTT...T*T
            ops_t2b += std::pow(2.0, i + 1) * static_cast<double>(chiL * bonds[i] * bonds[i + 1] * 2l);
            mem_t2b = std::max(mem_t2b, std::pow(2.0, i + 1) * static_cast<double>(chiL * bonds[i + 1] * 2l));
        } else if(i + 1 == sites.size() and i == bonds.size()) { // The last site
            ops_t2b += std::pow(2.0, i + 1) * static_cast<double>(2l * chiL * bonds[i - 1] * chiR);
            mem_t2b = std::max(mem_t2b, std::pow(2.0, i + 1) * static_cast<double>(2l * chiL * chiR));
        }
    }
    auto cost_t2b = ops_t2b + mem_t2b;

    if(side.starts_with('l')) {
        auto trf_cacheL = get_optimal_trf_from_cache<T>(sites, "l2r");
        auto cost_l2r   = get_transfer_matrix_cost(sites, "l2r", trf_cacheL);
        if(trf_cacheL.has_value()) trf_cacheL->cost = cost_l2r;
        return std::array{cost_t2b, cost_l2r};
    } else if(side.starts_with('r')) {
        auto trf_cacheR = get_optimal_trf_from_cache<T>(sites, "r2l");
        auto cost_r2l   = get_transfer_matrix_cost(sites, "r2l", trf_cacheR);
        if(trf_cacheR.has_value()) trf_cacheR->cost = cost_r2l;
        return std::array{cost_t2b, cost_r2l};
    } else {
        throw except::logic_error("get_transfer_matrix_costs: invalid side: {}", side);
    }
}

template<typename Scalar>
template<typename T>
typename StateFinite<Scalar>::template optional_tensor4ref<T> StateFinite<Scalar>::load_trf_from_cache(const std::string &key) const {
    if(key.empty()) return {};
    auto it = std::find_if(get_cache<T>.trf.begin(), get_cache<T>.trf.end(), [&key](const auto &elem) -> bool { return elem.first == key; });
    if(it != get_cache<T>.trf.end()) {
        if constexpr(debug_cache) tools::log->trace("load_trf_from_cache<{}>: cache_hit: {} | {} | {}", sfinae::type_name<T>(), key, it->second.dimensions());
        return std::cref(it->second);
    }
    return std::nullopt;
}

template<typename Scalar>
template<typename T>
typename StateFinite<Scalar>::template optional_tensor4ref<T> StateFinite<Scalar>::load_trf_from_cache(const std::vector<size_t> &sites, const size_t pos,
                                                                                                       std::string_view side) const {
    if(sites.empty()) return {};
    assert(pos >= sites.front());
    assert(pos <= sites.back());
    auto key = generate_cache_key(sites, pos, side);
    return load_trf_from_cache<T>(key);
}

template<typename Scalar>
template<typename T>
void StateFinite<Scalar>::save_trf_into_cache(const Eigen::Tensor<T, 4> &trf, const std::string &key) const {
    if(key.empty()) return;
    auto it = std::find_if(get_cache<T>().trf.rbegin(), get_cache<T>().trf.rend(), [&key](const auto &elem) -> bool { return elem.first == key; });
    if constexpr(debug_cache) {
        // if(!cache.trf_real.contains(key)) tools::log->trace("save_trf_into_cache: key: {} | {}", key, trf.dimensions());
        if(it == get_cache<T>().trf.rend()) tools::log->trace("save_trf_into_cache<{}>: key: {} | {}", sfinae::type_name<T>(), key, trf.dimensions());
    }
    // cache.trf_real[key] = trf;
    if(it == get_cache<T>().trf.rend()) get_cache<T>().trf.emplace_back(std::make_pair(key, trf));
    shrink_cache();
}

template<typename Scalar>
template<typename T>
void StateFinite<Scalar>::save_trf_into_cache(const Eigen::Tensor<T, 4> &trf, const std::vector<size_t> &sites, size_t pos, std::string_view side) const {
    if(sites.empty()) return;
    if(side.empty()) return;
    assert(pos >= sites.front());
    assert(pos <= sites.back());
    auto key = generate_cache_key(sites, pos, side);
    if(side.starts_with('l') and key.ends_with("L]")) return;   // It cannot grow l2r any more
    if(side.starts_with('r') and key.starts_with("[L")) return; // It cannot grow r2l any more
    save_trf_into_cache<T>(trf, key);
}

template<typename Scalar>
template<typename T>
std::optional<typename StateFinite<Scalar>::template TrfCacheEntry<T>> StateFinite<Scalar>::get_optimal_trf_from_cache(const std::vector<size_t> &sites,
                                                                                                                       std::string_view           side) const {
    // We want to find the cheapest cache entry to start from, that has the most sites contracted into it already
    std::optional<TrfCacheEntry<T>> cacheEntry = std::nullopt;
    if(side.starts_with('l')) {
        for(const auto &posR : iter::reverse(sites)) { // posL is fixed, move the posR cursor towards posL
            auto key        = generate_cache_key(sites, posR, "l2r");
            auto nremaining = sites.back() - posR;
            auto ncontained = sites.size() - nremaining;
            // if(auto it = cache.trf_real.find(key); it != cache.trf_real.end()) {
            auto it = std::find_if(get_cache<T>().trf.rbegin(), get_cache<T>().trf.rend(), [&key](const auto &elem) -> bool { return elem.first == key; });
            if(it != get_cache<T>().trf.rend()) {
                if constexpr(debug_cache)
                    tools::log->trace("get_optimal_trf_from_cache<{}>: cache hit: pos {} | {} | sites {}", sfinae::type_name<T>(), posR, side, sites);
                cacheEntry = TrfCacheEntry<T>{.pos        = posR,
                                              .side       = "l2r",
                                              .key        = key,
                                              .ncontained = ncontained,
                                              .nremaining = nremaining,
                                              .cost       = std::numeric_limits<double>::quiet_NaN(),
                                              .trf        = std::cref(it->second)};
                break;
            }
        }
    }
    if(side.starts_with('r')) {
        for(const auto &posL : sites) { // posR is fixed, move the posL cursor towards posR
            auto key        = generate_cache_key(sites, posL, "r2l");
            auto ncontained = sites.back() - posL + 1;
            auto nremaining = sites.size() - ncontained;
            // if(auto it = cache.trf_real.find(key); it != cache.trf_real.end()) {
            auto it = std::find_if(get_cache<T>().trf.rbegin(), get_cache<T>().trf.rend(), [&key](const auto &elem) -> bool { return elem.first == key; });
            if(it != get_cache<T>().trf.rend()) {
                if constexpr(debug_cache)
                    tools::log->trace("get_optimal_trf_from_cache<{}>: cache hit: pos {} | {} | sites {}", sfinae::type_name<T>(), posL, side, sites);
                cacheEntry = TrfCacheEntry<T>{.pos        = posL,
                                              .side       = "r2l",
                                              .key        = key,
                                              .ncontained = ncontained,
                                              .nremaining = nremaining,
                                              .cost       = std::numeric_limits<double>::quiet_NaN(),
                                              .trf        = std::cref(it->second)};
                break;
            }
        }
    }
    cacheEntry->cost = get_transfer_matrix_cost<T>(sites, side, cacheEntry);
    return cacheEntry;
}

template<typename Scalar>
template<typename T>
std::optional<typename StateFinite<Scalar>::template TrfCacheEntry<T>> StateFinite<Scalar>::get_optimal_trf_from_cache(const std::vector<size_t> &sites) const {
    // We want to inspect the cache to find out which is the cheapest cache entry to start from.
    // We are looking for the longest cache entry (in number of sites).
    std::optional<TrfCacheEntry<T>> cacheL = get_optimal_trf_from_cache<T>(sites, "l2r");
    std::optional<TrfCacheEntry<T>> cacheR = get_optimal_trf_from_cache<T>(sites, "r2l");

    // Return the cache entry that would be cheapest to complete
    if(cacheL.has_value() and cacheR.has_value()) {
        if constexpr(debug_cache) tools::log->trace("get_optimal_trf_from_cache: comparing cacheL {} | cacheR {}", cacheL->key, cacheR->key);
        if(cacheL->cost < cacheR->cost) { return cacheL; }
        if(cacheR->cost < cacheL->cost) { return cacheR; }

        // Compare the number of sites remaining
        if(cacheL->nremaining < cacheR->nremaining) { return cacheL; }
        if(cacheR->nremaining < cacheL->nremaining) { return cacheR; }
        // In this case there are an equal number of remaining sites to contract.
        // The cheapest is the one with the smallest static dimension on the respective sides.
        if(cacheL->trf.get().dimension(0) < cacheR->trf.get().dimension(2)) { return cacheL; }
        if(cacheR->trf.get().dimension(2) < cacheL->trf.get().dimension(0)) { return cacheR; }
        // If those are also equal, then take the one with the smallest dynamic dimension.
        // If those are equal too... then it doesn't matter
        if(cacheL->trf.get().dimension(2) <= cacheR->trf.get().dimension(0)) { return cacheL; }
        if(cacheR->trf.get().dimension(0) <= cacheL->trf.get().dimension(2)) { return cacheR; }
    }
    if(cacheL.has_value()) return cacheL;
    if(cacheR.has_value()) return cacheR;
    return std::nullopt;
}

template<typename Scalar>
template<typename T>
typename StateFinite<Scalar>::template optional_tensor3ref<T> StateFinite<Scalar>::get_cached_mps(const std::string &key) const {
    auto &cache = get_cache<T>();
    auto  it    = std::find_if(cache.mps.begin(), cache.mps.end(), [&](const auto &elem) -> bool { return elem.first == key; });
    if(it != cache.mps.end()) return std::cref(it->second);
    return std::nullopt;
}

template<typename Scalar>
template<typename T>
bool StateFinite<Scalar>::has_cached_mps(const std::string &key) const {
    auto &cache = get_cache<T>();
    auto  it    = std::find_if(cache.mps.rbegin(), cache.mps.rend(), [&](const auto &elem) -> bool { return elem.first == key; });
    if constexpr(debug_cache)
        if(it != cache.mps.rend()) tools::log->trace("multisite_mps: cache_hit: {}", key);
    return it != cache.mps.rend();
}
