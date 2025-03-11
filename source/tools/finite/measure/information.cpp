
#include "../measure.h"
#include "config/settings.h"
#include "debug/info.h"
#include "general/iter.h"
#include "io/fmt_custom.h"
#include "math/eig.h"
#include "math/float.h"
#include "math/linalg/matrix.h"
#include "math/linalg/tensor.h"
#include "math/num.h"
#include "math/stat.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "qm/mpo.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/common/split.h"
#include "tools/finite/mps.h"
#include <Eigen/src/Core/GlobalFunctions.h>
#include <Eigen/SVD>
#include <span>

namespace settings {
    constexpr bool debug_subsystem_entropy = false;
}

template<typename ContainerType>
double count_finite(const ContainerType &arr) {
    double count = 0;
    for(const auto &a : arr) {
        if(std::isnan(a)) continue;
        if(std::isinf(a)) continue;
        count += 1;
    }
    return count;
}

template<typename ContainerType>
double nansum(const ContainerType &arr) {
    if(arr.size() == 0) return std::numeric_limits<double>::quiet_NaN();
    double sum   = 0;
    double count = 0;
    for(const auto &a : arr) {
        if(std::isnan(a)) continue;
        sum += a;
        count += 1;
    }
    if(count == 0) return std::numeric_limits<double>::quiet_NaN();
    return sum;
}

template<typename ContainerType>
double nanmean(const ContainerType &arr) {
    if(arr.size() == 0) return std::numeric_limits<double>::quiet_NaN();
    double sum   = 0;
    double count = 0;
    for(const auto &a : arr) {
        if(std::isnan(a)) continue;
        sum += a;
        count += 1;
    }
    if(count == 0) return std::numeric_limits<double>::quiet_NaN();
    return sum / count;
}

std::vector<size_t> get_subsystem_complement(size_t length, const std::vector<size_t> &subsystem) {
    // Assume the subsystem is contiguous and sorted
    if(subsystem.empty()) return num::range<size_t>(0, length);
    std::vector<size_t> complement = num::range<size_t>(0, subsystem.front());
    if(subsystem.back() + 1 != length) {
        auto right = num::range<size_t>(subsystem.back() + 1, length);
        complement.insert(complement.end(), right.begin(), right.end());
    }
    // for(size_t pos = 0; pos < length; ++pos) {
    //     if(std::find(subsystem.begin(), subsystem.end(), pos) == subsystem.end()) complement.emplace_back(pos);
    // }
    return complement;
}

double tools::finite::measure::entanglement_entropy_log2(const StateFinite &state, size_t nsites /* sites to the left of the partition */) {
    auto t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    if(nsites == 0ul) return 0.0;
    auto pos_pl1 = static_cast<size_t>(state.get_position<long>() + 1l);
    auto pos_tgt = 0ul;
    if(pos_pl1 < nsites) {
        // Get the L of the B-site at pos == nsites-1
        pos_tgt = nsites - 1ul;
    } else if(pos_pl1 == nsites) {
        // Get the LC of the AC site at pos == nsites-1
        pos_tgt = nsites - 1ul;
    } else {
        // Get the L of the A (or AC) -site at pos == nsites
        pos_tgt = nsites;
    }
    const auto &mps = state.get_mps_site(pos_tgt);
    if(mps.isCenter()) {
        auto &LC = mps.get_LC();
        auto  SE = Eigen::Tensor<cx64, 0>(-LC.square().contract(LC.square().log2().eval(), tenx::idx({0}, {0})));
        return std::abs(SE(0));
    } else {
        auto &L  = mps.get_L();
        auto  SE = Eigen::Tensor<cx64, 0>(-L.square().contract(L.square().log2().eval(), tenx::idx({0}, {0})));
        return std::abs(SE(0));
    }
}

std::vector<double> tools::finite::measure::entanglement_entropies_log2(const StateFinite &state) {
    auto                t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    std::vector<double> entanglement_entropies;
    entanglement_entropies.reserve(state.get_length() + 1);
    if(not state.has_center_point()) entanglement_entropies.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        auto                  &L  = mps->get_L();
        Eigen::Tensor<cx64, 0> SE = -L.square().contract(L.square().log2().eval(), tenx::idx({0}, {0}));
        entanglement_entropies.emplace_back(std::abs(SE(0)));
        if(mps->isCenter()) {
            auto &LC = mps->get_LC();
            SE       = -LC.square().contract(LC.square().log2().eval(), tenx::idx({0}, {0}));
            entanglement_entropies.emplace_back(std::abs(SE(0)));
        }
    }
    if(entanglement_entropies.size() != state.get_length() + 1) throw except::logic_error("entanglement_entropies.size() should be length+1");
    if(entanglement_entropies.front() != 0.0) throw except::logic_error("First entropy should be 0. Got: {:.16f}", entanglement_entropies.front());
    if(entanglement_entropies.back() != 0.0) throw except::logic_error("Last entropy should be 0. Got: {:.16f}", entanglement_entropies.back());
    return entanglement_entropies;
}

double tools::finite::measure::subsystem_entanglement_entropy_log2(const StateFinite &state, const std::vector<size_t> &sites, Precision prec,
                                                                   size_t eig_max_size = -1ul, std::string_view side = "") {
    if(sites.empty()) return 0;
    bool sites_is_contiguous = sites == num::range<size_t>(sites.front(), sites.back() + 1);
    if(not sites_is_contiguous) { throw except::logic_error("sites are not contiguous: {}", sites); }
    // If sites is contiguous starting at 0, then we can return a bipartite entropy
    if(sites == num::range<size_t>(0ul, sites.size())) { return tools::finite::measure::entanglement_entropy_log2(state, sites.size()); }
    // If sites is contiguous ending at L-1, then we can return a bipartite entropy
    auto L = state.get_length<size_t>();
    if(sites == num::range<size_t>(L - sites.size(), L)) { return tools::finite::measure::entanglement_entropy_log2(state, L - sites.size()); }
    if(eig_max_size > 10000) throw except::runtime_error("subsystem_entanglement_entropy_log2: eig_max_size invalid range (<= 10000). Got: {}", eig_max_size);
    bool is_real = state.is_real();
    auto cites   = get_subsystem_complement(L, sites); // Complement of sites

    // Determine whether to use the density matrix, its complement or the transfer matrix approach
    // Since the eigenvalue decomposition is the most limiting factor, we use an upper limit by setting its cost to infinity when above the threshold
    auto chiL         = state.get_mps_site(sites.front()).get_chiL();
    auto chiR         = state.get_mps_site(sites.back()).get_chiR();
    auto eig_size_rho = static_cast<double>(std::pow(2.0, sites.size()));
    auto eig_size_cmp = static_cast<double>(std::pow(2.0, cites.size())); // The complement of rho
    auto eig_size_trf = static_cast<double>(chiL * chiR);
    auto eig_cost_rho = eig_size_rho <= static_cast<double>(eig_max_size) ? std::pow(eig_size_rho, 3.0) : std::numeric_limits<double>::infinity();
    auto eig_cost_cmp = eig_size_cmp <= static_cast<double>(eig_max_size) ? std::pow(eig_size_cmp, 3.0) : std::numeric_limits<double>::infinity();
    auto eig_cost_trf = eig_size_trf <= static_cast<double>(eig_max_size) ? std::pow(eig_size_trf, 3.0) : std::numeric_limits<double>::infinity();

    auto eig_sizes = std::array{eig_size_rho, eig_size_cmp, eig_size_trf};
    // auto eig_costs = std::array{eig_cost_rho, eig_cost_cmp, eig_cost_trf};

    std::array<double, 3> mat_costs_rho = {};
    std::array<double, 3> mat_costs_cmp = {};
    std::array<double, 2> mat_costs_trf = {};
    switch(prec) {
        case Precision::SINGLE: {
            mat_costs_rho = is_real ? state.get_reduced_density_matrix_cost<fp32>(sites) : state.get_reduced_density_matrix_cost<cx32>(sites);
            mat_costs_cmp = is_real ? state.get_reduced_density_matrix_cost<fp32>(cites) : state.get_reduced_density_matrix_cost<cx32>(cites);
            mat_costs_trf = is_real ? state.get_transfer_matrix_costs<fp32>(sites, side) : state.get_transfer_matrix_costs<cx32>(sites, side);
            break;
        }
        case Precision::DOUBLE: {
            mat_costs_rho = is_real ? state.get_reduced_density_matrix_cost<fp64>(sites) : state.get_reduced_density_matrix_cost<cx64>(sites);
            mat_costs_cmp = is_real ? state.get_reduced_density_matrix_cost<fp64>(cites) : state.get_reduced_density_matrix_cost<cx64>(cites);
            mat_costs_trf = is_real ? state.get_transfer_matrix_costs<fp64>(sites, side) : state.get_transfer_matrix_costs<cx64>(sites, side);
            break;
        }
    }

    auto min_cost_rho_idx = std::distance(mat_costs_rho.begin(), std::min_element(mat_costs_rho.begin(), mat_costs_rho.end()));
    auto min_cost_cmp_idx = std::distance(mat_costs_cmp.begin(), std::min_element(mat_costs_cmp.begin(), mat_costs_cmp.end()));
    auto min_cost_trf_idx = std::distance(mat_costs_trf.begin(), std::min_element(mat_costs_trf.begin(), mat_costs_trf.end()));
    auto mat_cost_rho     = mat_costs_rho.at(safe_cast<size_t>(min_cost_rho_idx));
    auto mat_cost_cmp     = mat_costs_cmp.at(safe_cast<size_t>(min_cost_cmp_idx));
    auto mat_cost_trf     = mat_costs_trf.at(safe_cast<size_t>(min_cost_trf_idx));
    auto mat_costs        = std::array{mat_cost_rho, mat_cost_cmp, mat_cost_trf};
    auto tot_costs        = std::array{eig_cost_rho + mat_cost_rho, eig_cost_cmp + mat_cost_cmp, eig_cost_trf + mat_cost_trf};

    // auto names        = std::array{"rho", "cmp", "trf"};
    // auto min_eig_cost_itr = std::min_element(eig_costs.begin(), eig_costs.end());
    // auto min_eig_cost_idx = std::distance(eig_costs.begin(), min_eig_cost_itr);
    // auto min_eig_cost_val = eig_costs[min_eig_cost_idx];

    // auto min_mat_cost_itr = std::min_element(mat_costs.begin(), mat_costs.end());
    // auto min_mat_cost_idx = std::distance(mat_costs.begin(), min_mat_cost_itr);
    // auto min_mat_cost_val = mat_costs[min_mat_cost_idx];

    auto min_cost_itr = std::min_element(tot_costs.begin(), tot_costs.end());
    auto min_cost_idx = std::distance(tot_costs.begin(), min_cost_itr);
    auto min_cost_val = tot_costs.at(safe_cast<size_t>(min_cost_idx));

    if(std::isinf(min_cost_val)) {
        tools::log->trace("mode ???: eig {} | mat {} | chi {} {} | sites {} | costs {}", eig_sizes, mat_costs, chiL, chiR, sites, min_cost_val);
        return std::numeric_limits<double>::quiet_NaN(); // No cost is smaller than max mat size
    }

    auto                  solver   = eig::solver();
    auto                  evs      = Eigen::ArrayXd(); // Eigenvalues
    [[maybe_unused]] auto mat_time = 0.0;
    [[maybe_unused]] auto eig_time = 0.0;
    switch(prec) {
        case Precision::SINGLE: {
            if(is_real) {
                Eigen::Tensor<fp32, 2> mat;
                if(min_cost_idx == 0) {
                    mat      = state.get_reduced_density_matrix<fp32>(sites);
                    mat_time = tid::get("rho").get_last_interval();
                } else if(min_cost_idx == 1) {
                    mat      = state.get_reduced_density_matrix<fp32>(cites);
                    mat_time = tid::get("rho").get_last_interval();
                } else if(min_cost_idx == 2) {
                    mat      = state.get_transfer_matrix<fp32>(sites, side);
                    mat_time = tid::get("trf").get_last_interval();
                }
                solver.eig<eig::Form::SYMM>(mat.data(), mat.dimension(0), eig::Vecs::OFF);
                evs = eig::view::get_eigvals<fp32>(solver.result).cast<fp64>(); // Eigenvalues
            } else {
                Eigen::Tensor<cx32, 2> mat;
                if(min_cost_idx == 0) {
                    mat      = state.get_reduced_density_matrix<cx32>(sites);
                    mat_time = tid::get("rho").get_last_interval();
                } else if(min_cost_idx == 1) {
                    mat      = state.get_reduced_density_matrix<cx32>(sites);
                    mat_time = tid::get("rho").get_last_interval();
                } else if(min_cost_idx == 2) {
                    mat      = state.get_transfer_matrix<cx32>(sites, side);
                    mat_time = tid::get("trf").get_last_interval();
                }
                solver.eig<eig::Form::SYMM>(mat.data(), mat.dimension(0), eig::Vecs::OFF);
                evs = eig::view::get_eigvals<fp32>(solver.result).cast<fp64>(); // Eigenvalues
            }
            break;
        }
        case Precision::DOUBLE: {
            if(is_real) {
                Eigen::Tensor<fp64, 2> mat;
                if(min_cost_idx == 0) {
                    mat      = state.get_reduced_density_matrix<fp64>(sites);
                    mat_time = tid::get("rho").get_last_interval();
                    // if(debug::mem_hwm_in_mb() > 10000) throw except::runtime_error("Exceeded 5G high water mark after rho");
                } else if(min_cost_idx == 1) {
                    mat      = state.get_reduced_density_matrix<fp64>(cites);
                    mat_time = tid::get("rho").get_last_interval();
                    // if(debug::mem_hwm_in_mb() > 10000) throw except::runtime_error("Exceeded 5G high water mark after cmp");
                } else if(min_cost_idx == 2) {
                    mat      = state.get_transfer_matrix<fp64>(sites, side);
                    mat_time = tid::get("trf").get_last_interval();
                }
                solver.eig<eig::Form::SYMM>(mat.data(), mat.dimension(0), eig::Vecs::OFF);
                evs = eig::view::get_eigvals<fp64>(solver.result); // Eigenvalues
            } else {
                Eigen::Tensor<cx64, 2> mat;
                if(min_cost_idx == 0) {
                    mat      = state.get_reduced_density_matrix<cx64>(sites);
                    mat_time = tid::get("rho").get_last_interval();
                } else if(min_cost_idx == 1) {
                    mat      = state.get_reduced_density_matrix<cx64>(sites);
                    mat_time = tid::get("rho").get_last_interval();
                } else if(min_cost_idx == 2) {
                    mat      = state.get_transfer_matrix<cx64>(sites, side);
                    mat_time = tid::get("trf").get_last_interval();
                }
                solver.eig<eig::Form::SYMM>(mat.data(), mat.dimension(0), eig::Vecs::OFF);
                evs = eig::view::get_eigvals<fp64>(solver.result); // Eigenvalues
            }
            break;
        }
    }

    eig_time                       = solver.result.meta.time_total;
    fp64 entanglement_entropy_log2 = 0;
    for(const auto &e : evs) {
        if(e > 0) entanglement_entropy_log2 += -e * std::log2(e); // We use log base 2 for information
    }
    // if(eig_time > 1.0)
    if constexpr(settings::debug_subsystem_entropy)
        tools::log->trace("mode {} side {} | eig {} (max {}) | mat {} | chi {} {} | S {:.8f} | cch {::.2f} GB | mat {:.3e} s | eig {:.3e} s | sites {}",
                          min_cost_idx, side, eig_sizes, eig_max_size, mat_costs, chiL, chiR, entanglement_entropy_log2, state.get_cache_sizes(), mat_time,
                          eig_time, sites);
    // if(debug::mem_hwm_in_mb() > 10000) throw except::runtime_error("Exceeded 5G high water mark after eig");

    // auto mmps                            = state.get_multisite_mps(sites);
    // auto dims                            = mmps.dimensions();
    // auto mmps2                           = Eigen::Tensor<fp64, 2>(mmps.reshape(tenx::array2{dims[0], dims[1] * dims[2]}).real());
    // auto svd_solver                      = svd::solver();
    // auto [u, sv, v]                      = svd_solver.decompose(mmps2);
    // double entanglement_entropy_log2_new = 0;
    // for(long sidx = 0; sidx < sv.size(); ++sidx) {
    //     auto sval = sv[sidx];
    //     if(sval > 0) entanglement_entropy_log2_new += -sval * sval * std::log2(sval * sval); // We use log base 2 for information
    // }
    // tools::log->info("old: {:.16f} | new:{:.16f} | diff: {:.16f}", entanglement_entropy_log2, entanglement_entropy_log2_new,
    //                  entanglement_entropy_log2 - entanglement_entropy_log2_new);

    return entanglement_entropy_log2;
}

std::pair<std::deque<std::vector<size_t>>, std::deque<std::vector<size_t>>> get_missing_subsystems(const Eigen::ArrayXXd &see) {
    auto subsystemsL = std::deque<std::vector<size_t>>(); // list of missing subsystems closest to the left edge
    auto subsystemsR = std::deque<std::vector<size_t>>(); // list of missing subsystems closest to the right edge
    auto len         = see.rows();
    for(long off = 0; off < len; ++off) {
        for(long ext = 1; off + ext < len; ++ext) {
            if(std::isnan(see(ext - 1, off))) {
                long pos_mid = off + ext / 2;
                if(pos_mid <= len / 2) {
                    subsystemsL.emplace_back(num::range<size_t>(off, off + ext));
                } else {
                    subsystemsR.emplace_back(num::range<size_t>(off, off + ext));
                }
            }
        }
    }
    // The left subsystems are already sorted in ascending distance from the left edge and growing subsystem size
    // The right subsystems must be sorted in ascending distance from the right edge, and growing subsystem size
    std::sort(subsystemsR.begin(), subsystemsR.end(), [](const auto &sa, const auto &sb) -> bool {
        if(sa.back() == sb.back()) return sa.size() < sb.size();
        return sa.back() > sb.back();
    });
    return {subsystemsL, subsystemsR};
}

struct SeeProgress {
    Eigen::ArrayXXd info;
    double          icom;
    double          bits_total = 0;
    double          bits_found = 0;
    double          bits_error = 0;
    double          bits_minus = 0;
    double          progress   = 0;
};

SeeProgress check_see_progress(const Eigen::ArrayXXd &see, LogPolicy log_policy = LogPolicy::SILENT, spdlog::level::level_enum lvl = spdlog::level::debug) {
    // Check the information lattice
    auto sp       = SeeProgress();
    auto L        = static_cast<double>(see.rows());
    sp.info       = tools::finite::measure::information_lattice(see);
    sp.icom       = tools::finite::measure::information_center_of_mass(sp.info);
    sp.bits_total = L;
    sp.bits_found = sp.info.isNaN().select(0, sp.info).sum();
    sp.bits_error = 1 - sp.bits_found / sp.bits_total;
    sp.bits_minus = (sp.info < 0).select(sp.info, 0).sum();
    sp.progress =
        (see.isNaN()).select(0.0, Eigen::ArrayXXd::Ones(see.rows(), see.cols())).sum() / (L * (L + 1.0) / 2.0); // Found entries in the top left triangle
    if(log_policy == LogPolicy::VERBOSE) {
        tools::log->log(lvl, "see: \n{}\n", linalg::matrix::to_string(see, 10));
        tools::log->log(lvl, "info: \n{}\n", linalg::matrix::to_string(sp.info, 6));
        tools::log->log(lvl, "il: \n{}\n", linalg::matrix::to_string(tools::finite::measure::information_per_scale(sp.info), 6));
    }
    if(log_policy != LogPolicy::SILENT) {
        tools::log->log(lvl, " -- progress {:<6.2f}% bits {:<20.16f}/{} err {:.2e} icom {:.16f} neg {:.1e} mem[rss {:<.1f} hwm {:<.1f}]MB", sp.progress * 100,
                        sp.bits_found, sp.bits_total, sp.bits_error, sp.icom, sp.bits_minus, debug::mem_rss_in_mb(), debug::mem_hwm_in_mb());
    }
    return sp;
}

/*! Calculates the subsystem entanglement entropies,
 * starting with the cheapest entries and progressively harder until a threshold number of bits have been found.
 * Use max_bit_error to set the threshold:
 *  - use a positive value to indicate relative error
 *  - use a negative value to indicate absolute error (i.e. how many missing bits you can accept)
 */

Eigen::ArrayXXd tools::finite::measure::subsystem_entanglement_entropies_log2(const StateFinite &state, InfoPolicy ip) {
    if(has_flag(ip.cachePolicy, CachePolicy::READ) and state.measurements.subsystem_entanglement_entropies.has_value())
        return state.measurements.subsystem_entanglement_entropies.value();
    auto t_see        = tid::tic_scope("see", tid::level::normal);
    ip.bits_max_error = ip.bits_max_error.value_or(settings::storage::dataset::subsystem_entanglement_entropies::bits_err);
    ip.eig_max_size   = ip.eig_max_size.value_or(settings::storage::dataset::subsystem_entanglement_entropies::eig_size);
    ip.svd_max_size   = ip.svd_max_size.value_or(settings::storage::dataset::subsystem_entanglement_entropies::bond_lim);
    ip.svd_trnc_lim   = ip.svd_trnc_lim.value_or(settings::storage::dataset::subsystem_entanglement_entropies::trnc_lim);
    ip.precision      = ip.precision.value_or(settings::storage::dataset::subsystem_entanglement_entropies::precision);
    ip.cachePolicy    = ip.cachePolicy.value_or(settings::storage::dataset::subsystem_entanglement_entropies::cache);
    auto length       = state.get_length<size_t>();
    auto state_temp   = state;
    auto sites_temp   = num::range<size_t>(0, length);

    // Start by normalizing the state.
    // For some reason this step helps with the precision of entropies that we obtain later.
    // If skipped, the entropy for a subsystem and its complement may differ up to 1e-3.
    // Note from later: this may have been fixed now.
    // Another note: We should not truncate the state here at all (e.g. by setting a bond limit). Do that before calling this function instead.
    // Likewise, we shouldn't truncate during swaps, since that introduces an uncontrolled error of in the entropies and subsequent information.
    // We may get away with using slightly higher truncation_limit during the swap process, to get a controlled error.
    auto svd_cfg = svd::config(settings::get_bond_max(state.get_algorithm()), ip.svd_trnc_lim);
    tools::log->debug("subsystem_entanglement_entropies_log2: Normalizing state | svd: {}", svd_cfg.to_string());
    finite::mps::normalize_state(state_temp, svd_cfg, NormPolicy::ALWAYS);

    // finite::mps::move_center_point_to_pos_dir(state_temp, 0, 1, ip.svd_cfg);
    tools::log->info("subsystem_entanglement_entropies_log2: calculating | max bit err {:.2e} | max size eig {} svd {} trnc {:.2e} | prec {}",
                     ip.bits_max_error.value(), ip.eig_max_size.value(), ip.svd_max_size.value(), ip.svd_trnc_lim.value(), enum2sv(ip.precision.value()));

    constexpr auto  nan = std::numeric_limits<double>::quiet_NaN();
    auto            len = state_temp.get_length<long>();
    auto            bee = measure::entanglement_entropies_log2(state_temp); // bipartite entanglement entropies
    Eigen::ArrayXXd see = Eigen::ArrayXXd::Zero(len, len) * nan;            // susbsystem entanglement entropies

    // We begin with the low-hanging fruit (1): Bipartite entanglement entropies
    for(long ext = 1; ext <= len; ++ext) { // Assume off == 0
        auto idx        = safe_cast<size_t>(ext);
        see(ext - 1, 0) = bee[idx];
        if(len - ext >= 1) see(len - ext - 1, ext) = bee[idx];
    }
    check_see_progress(see, LogPolicy::DEBUG, spdlog::level::debug);

    // Next we take small subsystems
    // We can take subsystems on the left side first and discard their caches before moving onto those on the right side.
    auto [subsystemsL, subsystemsR] = get_missing_subsystems(see);
    {
        auto posL = subsystemsL.empty() ? -1ul : subsystemsL.front().front();
        for(const auto &subsystem : subsystemsL) {
            auto off          = safe_cast<long>(subsystem.front());
            auto ext          = safe_cast<long>(subsystem.size());
            see(ext - 1, off) = subsystem_entanglement_entropy_log2(state_temp, subsystem, ip.precision.value(), ip.eig_max_size.value(), "left");
            if(!std::isnan(see(ext - 1, off)) and see(ext - 1, off) < -1e-8) throw except::runtime_error("Negative entropy: {:.16f}", see(ext - 1, off));
            if(posL != safe_cast<size_t>(off) or &subsystem == &subsystemsL.back()) {
                check_see_progress(see, LogPolicy::DEBUG);
                posL = safe_cast<size_t>(off);
            }
        }
    }
    {
        auto posR = subsystemsR.empty() ? -1ul : subsystemsR.front().back();
        for(const auto &subsystem : subsystemsR) {
            auto off          = safe_cast<long>(subsystem.front());
            auto ext          = safe_cast<long>(subsystem.size());
            see(ext - 1, off) = subsystem_entanglement_entropy_log2(state_temp, subsystem, ip.precision.value(), ip.eig_max_size.value(), "right");
            if(!std::isnan(see(ext - 1, off)) and see(ext - 1, off) < -1e-8) throw except::runtime_error("Negative entropy: {:.16f}", see(ext - 1, off));
            if(posR != subsystem.back() or &subsystem == &subsystemsR.back()) {
                check_see_progress(see, LogPolicy::DEBUG);
                posR = subsystem.back();
            }
        }
    }
    check_see_progress(see, LogPolicy::DEBUG);

    // Refresh the missing subsystems
    std::tie(subsystemsL, subsystemsR) = get_missing_subsystems(see);
    for(const auto &s : subsystemsL) tools::log->info("subsystemL {} -> see({},{})", s, s.size() - 1, s.front());
    for(const auto &s : subsystemsR) tools::log->info("subsystemR {} -> see({},{})", s, s.size() - 1, s.front());
    auto posL      = subsystemsL.empty() ? -1ul : subsystemsL.front().front();
    auto posR      = subsystemsR.empty() ? -1ul : subsystemsR.front().back();
    auto state_l2r = state_temp;
    auto sites_l2r = sites_temp;
    auto state_r2l = state_temp;
    auto sites_r2l = sites_temp;
    // long loop_counter = 0;
    while(!subsystemsL.empty() or !subsystemsR.empty()) {
        if(posL > 2) {
            // We need at least the first two columns of see to avoid getting nans in the first column of info
            auto sp = check_see_progress(see, LogPolicy::DEBUG);
            if(ip.bits_max_error >= 0 and sp.bits_error <= ip.bits_max_error) break;
            if(ip.bits_max_error < 0 and std::abs(sp.bits_total - sp.bits_found) <= std::abs(ip.bits_max_error.value())) break;
        }
        while(!subsystemsL.empty()) {
            // Starting from the original state, the idea is to move the subsystems to the left edge successively
            auto subsystem = subsystemsL.front();
            if(posL != subsystem.front()) {
                // Reset
                state_l2r = state_temp;
                sites_l2r = sites_temp;
                posL      = subsystem.front();
                break; // Go to right subsystems
            }

            // Repeatedly move the left-most site in the subsystem, "off" steps to the left. E.g.:  012[34]5 --> [3]012[4]5 --> [34]0125
            bool subsystem_success = true;
            for(const auto &[idx, pos] : iter::enumerate(subsystem)) {
                if(sites_l2r[idx] == pos) continue; // The site is already in place
                for(long i = safe_cast<long>(pos); i > static_cast<long>(idx); --i) {
                    // Make sure that the swap problem size is smaller than the highest.
                    // If the swap needs to truncate the state, we get an uncontrolled error in the information lattice.
                    const auto &mpsL = state_l2r.get_mps_site(i - 1);
                    const auto &mpsR = state_l2r.get_mps_site(i);
                    const auto  rows = static_cast<double>(mpsL.get_chiL());
                    const auto  cols = static_cast<double>(mpsR.get_chiR());
                    if(rows * cols > ip.svd_max_size.value() * ip.svd_max_size.value()) {
                        // Abort! Go to the next subsystem.
                        tools::log->info("swap l2r failed between sites {} (dim {}) and {} (dim {}): SVD would exceed bond_lim {} | subsystem {}", i - 1,
                                         mpsL.dimensions(), i, mpsR.dimensions(), ip.svd_max_size.value(), subsystem);
                        subsystem_success = false;
                        break;
                    }
                    mps::swap_sites(state_l2r, safe_cast<size_t>(i - 1), safe_cast<size_t>(i), sites_l2r, GateMove::OFF, std::nullopt);
                }
                if(!subsystem_success) break;
            }
            auto ext = safe_cast<long>(subsystem.size());
            if(subsystem_success) {
                // Check that the subsystem is actually where we expect
                auto sites_l2r_span = std::span(sites_l2r.begin(), subsystem.size());
                bool subsystem_ok   = std::equal(sites_l2r_span.begin(), sites_l2r_span.end(), subsystem.begin(), subsystem.end());
                if(!subsystem_ok)
                    throw except::logic_error("The subsystem was not moved to the edge:\n"
                                              "\t sites_l2r: {::2}\n"
                                              "\t subsystem: {::2}",
                                              sites_l2r, subsystem);

                // Now we can read off the missing entropy
                see(ext - 1, safe_cast<long>(posL)) =
                    subsystem_entanglement_entropy_log2(state_l2r, num::range<size_t>(0, ext), ip.precision.value(), -1ul, "");
            }
            subsystemsL.pop_front();
            tools::log->info("swap l2r subs [{}-{}] | see({},{})={:.6f} | sites {::2} | bonds {::3}", subsystem.front(), subsystem.back(), ext - 1, posL,
                             see(ext - 1, safe_cast<long>(posL)), sites_l2r, measure::bond_dimensions(state_l2r));

            if(!subsystem_success) {
                // Reset and pop entries that have the same posR
                while(!subsystemsL.empty() and posL == subsystemsL.front().front()) { subsystemsL.pop_front(); }
            }
        }

        if(posL > 2) {
            // We need at least the first two columns of see to avoid getting nans in the first column of info
            auto sp = check_see_progress(see, LogPolicy::DEBUG);
            if(ip.bits_max_error >= 0 and sp.bits_error <= ip.bits_max_error) break;
            if(ip.bits_max_error < 0 and std::abs(sp.bits_total - sp.bits_found) <= std::abs(ip.bits_max_error.value())) break;
        }
        while(!subsystemsR.empty()) {
            // Starting from the original state, the idea is to move the subsystems to the right edge successively
            auto subsystem = subsystemsR.front();
            if(posR != subsystem.back()) {
                // Reset
                state_r2l = state_temp;
                sites_r2l = sites_temp;
                posR      = subsystem.back();
                break; // Go to left subsystems
            }

            // Repeatedly move the right-most site in the subsystem, posR steps to the right. E.g.:  012[34]5 --> 012[3]5[4] --> 0125[34]
            long nswap             = len - safe_cast<long>(posR) - 1l;
            bool subsystem_success = true;
            for(const auto &[idx, pos] : iter::enumerate_reverse(subsystem)) {
                size_t rdx = subsystem.size() - idx - 1ul;       // 0,1,2... subsystem.size()-1
                size_t rof = safe_cast<size_t>(len) - rdx - 1ul; // L-1, L-2, L-3...,  Index of pos starting from the right edge
                if(sites_r2l[rof] == pos) continue;              // The site is already in place
                for(long i = safe_cast<long>(pos); i < safe_cast<long>(pos) + nswap; ++i) {
                    // Make sure that the swap problem size is smaller than the highest.
                    // If the swap needs to truncate the state, we get an uncontrolled error in the information lattice.
                    const auto &mpsL = state_r2l.get_mps_site(i);
                    const auto &mpsR = state_r2l.get_mps_site(i + 1);
                    const auto  rows = static_cast<double>(mpsL.get_chiL());
                    const auto  cols = static_cast<double>(mpsR.get_chiR());
                    if(rows * cols > ip.svd_max_size.value() * ip.svd_max_size.value()) {
                        // Abort! Go to the next subsystem.
                        // Abort! Go to the next subsystem.
                        tools::log->info("swap r2l failed between sites {} (dim {}) and {} (dim {}): SVD would exceed bond_lim {} | subsystem {}", i,
                                         mpsL.dimensions(), i + 1, mpsR.dimensions(), ip.svd_max_size.value(), subsystem);
                        subsystem_success = false;
                        break;
                    }
                    mps::swap_sites(state_r2l, safe_cast<size_t>(i), safe_cast<size_t>(i + 1), sites_r2l, GateMove::OFF, std::nullopt);
                }
                if(!subsystem_success) break;
            }
            auto ext = safe_cast<long>(subsystem.size());
            auto off = safe_cast<long>(subsystem.front());
            if(subsystem_success) {
                // Check that the subsystem is actually where we expect
                auto sites_r2l_span = std::span(sites_r2l.begin() + safe_cast<long>(length - subsystem.size()), subsystem.size());
                bool subsystem_ok   = std::equal(sites_r2l_span.begin(), sites_r2l_span.end(), subsystem.begin(), subsystem.end());
                if(!subsystem_ok)
                    throw except::logic_error("The subsystem was not moved to the R edge:\n"
                                              "\t sites_r2l: {::2}\n"
                                              "\t subsystem: {::2}",
                                              sites_r2l, subsystem);
                // Now we can read off the missing entropy
                see(ext - 1, off) =
                    subsystem_entanglement_entropy_log2(state_r2l, num::range<size_t>(length - safe_cast<size_t>(ext), length), ip.precision.value(), -1ul, "");
            }
            subsystemsR.pop_front();
            tools::log->info("swap r2l subs [{}-{}] | see({},{})={:.6f} | sites {::2} | bonds {::3}", subsystem.front(), subsystem.back(), ext - 1, off,
                             see(ext - 1, off), sites_r2l, measure::bond_dimensions(state_r2l));
            if(!subsystem_success) {
                // Pop entries that have the same posR
                while(!subsystemsR.empty() and posR == subsystemsR.front().back()) { subsystemsR.pop_front(); }
            }
        }
        // loop_counter++;
    }

    if(has_flag(ip.cachePolicy, CachePolicy::WRITE)) {
        state.measurements.see_time                         = t_see->get_last_interval();
        state.measurements.subsystem_entanglement_entropies = see;
    }
    check_see_progress(see, LogPolicy::DEBUG, spdlog::level::info);
    return see;
}

Eigen::ArrayXXd tools::finite::measure::information_lattice(const Eigen::ArrayXXd &SEE) {
    auto            Ldim        = SEE.rows();
    Eigen::ArrayXXd infolattice = Eigen::ArrayXXd::Zero(Ldim, Ldim);
    for(long nidx = 0; nidx < Ldim; ++nidx) /* The offset */ {
        for(long lidx = 0; lidx < Ldim; ++lidx) /* The extent, or "scale" */ {
            if(lidx + nidx >= Ldim) continue;
            if(lidx == 0) {
                infolattice(lidx, nidx) = 1 - SEE(lidx, nidx);
            } else if(lidx == 1) {
                infolattice(lidx, nidx) = -SEE(lidx, nidx) + SEE(lidx - 1, nidx) + SEE(lidx - 1, nidx + 1);
            } else {
                infolattice(lidx, nidx) = -SEE(lidx, nidx) + SEE(lidx - 1, nidx) + SEE(lidx - 1, nidx + 1) - SEE(lidx - 2, nidx + 1);
            }
        }
    }
    return infolattice; // Since SEE may contain NaN, this infolattice may contain NaN
}

Eigen::ArrayXXd tools::finite::measure::information_lattice(const StateFinite &state, InfoPolicy ip) {
    if(has_flag(ip.cachePolicy, CachePolicy::READ) and state.measurements.information_lattice.has_value())
        return state.measurements.information_lattice.value();
    auto SEE         = subsystem_entanglement_entropies_log2(state, ip);
    auto infolattice = information_lattice(SEE);
    if(has_flag(ip.cachePolicy, CachePolicy::WRITE)) state.measurements.information_lattice = infolattice;
    return infolattice;
}

Eigen::ArrayXd tools::finite::measure::information_per_scale(const Eigen::ArrayXXd &information_lattice) {
    return information_lattice.isNaN().select(0.0, information_lattice).rowwise().sum(); // Does not have NaN
}

Eigen::ArrayXd tools::finite::measure::information_per_scale(const StateFinite &state, InfoPolicy ip) {
    if(has_flag(ip.cachePolicy, CachePolicy::READ) and state.measurements.information_per_scale.has_value())
        return state.measurements.information_per_scale.value();
    auto           infolattice    = information_lattice(state, ip); // May have NaN
    Eigen::ArrayXd info_per_scale = information_per_scale(infolattice);
    if(has_flag(ip.cachePolicy, CachePolicy::WRITE)) state.measurements.information_per_scale = info_per_scale;
    // tools::log->info("info per scale: {::.3e}", state.measurements.information_per_scale.value());
    return info_per_scale;
}

double tools::finite::measure::information_bit_scale(const Eigen::ArrayXd &information_per_scale, double bit) {
    double idx = 0;
    for(idx = 0; idx < information_per_scale.size(); ++idx) {
        if(information_per_scale.topRows(idx + 1).sum() > bit) return idx;
    }
    return idx;
}

/*! For masses m_i at coordinates x_i, the center of mass is defined as
 *      sum_i m_i * x_i
 *      ---------------
 *      sum_i m_i
 */
double tools::finite::measure::information_center_of_mass(const Eigen::ArrayXXd &information_lattice) {
    Eigen::ArrayXd il = information_lattice.isNaN().select(0.0, information_lattice).rowwise().sum();
    auto           l  = Eigen::ArrayXd::LinSpaced(il.size(), 0.0, static_cast<double>(il.size() - 1l));
    auto           ic = il.cwiseProduct(l).sum() / il.sum();
    return ic;
}
/*! For masses m_i at coordinates x_i, the center of mass is defined as
 *      sum_i m_i * x_i
 *      ---------------
 *      sum_i m_i
 */
double tools::finite::measure::information_center_of_mass(const Eigen::ArrayXd &information_per_scale) {
    auto il = information_per_scale.isNaN().select(0.0, information_per_scale).eval();
    auto l  = Eigen::ArrayXd::LinSpaced(il.size(), 0.0, static_cast<double>(il.size() - 1l));
    auto ic = il.cwiseProduct(l).sum() / il.sum();
    return ic;
}

/*! For masses m_i at coordinates x_i, the center of mass is defined as
 *      sum_i m_i * x_i
 *      ---------------
 *      sum_i m_i
 */
double tools::finite::measure::information_center_of_mass(const StateFinite &state, InfoPolicy ip) {
    return information_center_of_mass(information_lattice(state, ip));
}

InfoAnalysis tools::finite::measure::information_lattice_analysis(const StateFinite &state, InfoPolicy ip) {
    InfoAnalysis info_analysis                     = {};
    info_analysis.ip                               = ip;
    info_analysis.subsystem_entanglement_entropies = subsystem_entanglement_entropies_log2(state, ip);

    info_analysis.info_lattice = information_lattice(info_analysis.subsystem_entanglement_entropies);
    if(has_flag(ip.cachePolicy, CachePolicy::WRITE)) state.measurements.information_lattice = info_analysis.info_lattice;

    info_analysis.info_per_scale = information_per_scale(info_analysis.info_lattice);
    if(has_flag(ip.cachePolicy, CachePolicy::WRITE)) state.measurements.information_per_scale = info_analysis.info_per_scale;

    info_analysis.icom = information_center_of_mass(info_analysis.info_per_scale);
    if(has_flag(ip.cachePolicy, CachePolicy::WRITE)) state.measurements.information_center_of_mass = info_analysis.icom;

    info_analysis.bits_found    = info_analysis.info_per_scale.sum();
    info_analysis.scale_bit_one = information_bit_scale(info_analysis.info_per_scale, 1);
    info_analysis.scale_bit_two = information_bit_scale(info_analysis.info_per_scale, 2);
    info_analysis.scale_bit_mid = information_bit_scale(info_analysis.info_per_scale, state.get_length<double>() / 2.0);
    info_analysis.scale_bit_pen = information_bit_scale(info_analysis.info_per_scale, state.get_length<double>() - 1.0);
    info_analysis.scale_bit_all = information_bit_scale(info_analysis.info_per_scale, state.get_length<double>());

    tools::log->debug("scale_bit_one : {:.16f}", info_analysis.scale_bit_one);
    tools::log->debug("scale_bit_two : {:.16f}", info_analysis.scale_bit_two);
    tools::log->debug("scale_bit_mid : {:.16f}", info_analysis.scale_bit_mid);
    tools::log->debug("scale_bit_pen : {:.16f}", info_analysis.scale_bit_pen);
    tools::log->debug("scale_bit_all : {:.16f}", info_analysis.scale_bit_all);
    return info_analysis;
}

/*! Calculates xi from the expectation value of the geometric distribution, such that p = 1-exp(-1/xi) */
double tools::finite::measure::information_xi_from_geometric_dist(const StateFinite &state, InfoPolicy ip) {
    Eigen::ArrayXd il = information_per_scale(state, ip);
    auto           L  = il.size();
    if(il.size() <= 2) return std::numeric_limits<double>::quiet_NaN();
    Eigen::ArrayXd l    = Eigen::ArrayXd::LinSpaced(L, 0.0, static_cast<double>(L - 1l));
    Eigen::ArrayXd mask = Eigen::ArrayXd::Ones(L); // Mask to select the interesting interval

    bool has_extended_info = il.bottomRows(L / 2).sum() >= 0.9; // Both topological and thermal states have >= 1 bits beyond L/2.
    bool decay_starts_at_0 = il[0] > il[1];
    bool decay_starts_at_1 = il[1] >= il[0] and il[1] > il[2];

    if(has_extended_info) mask.bottomRows(L / 2) = 0; // If there is a topological bit at l ~ L? we should only consider l in [0,L/2]
    if(decay_starts_at_1) mask[0] = 0;                // If the decay starts at l==1, take the geometric distribution starting at l==1

    auto ic = (il * mask).cwiseProduct(l).sum() / (il * mask).sum(); // information center of mass (or <l>)
    auto p  = 1.0;                                                   // Probability p for the geometric distribution

    if(decay_starts_at_0) {
        p = 1 / (1 + ic); // The geometric decay starts at l == 0
    } else if(decay_starts_at_1) {
        p = 1 / ic; // The geometric decay starts at l == 1
    }
    auto xi = -1.0 / std::log(1 - p);
    // auto N  = decay_starts_at_1 ? il[1] : il[0]; // The initial value of the exponential decay
    // auto lstart = decay_starts_at_1 ? 1.0 : 0.0;
    // auto y_geom = Eigen::ArrayXd((il * mask).sum() * p * pow(1 - p, l - lstart));
    // auto y_expd = Eigen::ArrayXd(N * mask * exp(-(l - lstart) / xi));
    // tools::log->info("info_xi_from_geom: ic       : {:.16f}", ic);
    // tools::log->info("info_xi_from_geom: p        : {:.16f}", p);
    // tools::log->info("info_xi_from_geom: xi       : {:.16f}", xi);
    // tools::log->info("info_xi_from_geom: y_info   : {::.16f}", il);
    // tools::log->info("info_xi_from_geom: y_geom   : {::.16f}", y_geom);
    // tools::log->info("info_xi_from_geom: y_expfit : {::.16f}", y_expd);
    return xi;
}

/*! Calculates xi from the slope of the decay in the exponential distribution in log-scale */
double tools::finite::measure::information_xi_from_avg_log_slope(const StateFinite &state, InfoPolicy ip) {
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::ArrayXd il  = information_per_scale(state, ip);
    if(il.size() <= 3) return nan;
    auto           L    = il.size();
    Eigen::ArrayXd l    = Eigen::ArrayXd::LinSpaced(L, 0.0, static_cast<double>(L - 1l));
    Eigen::ArrayXd mask = Eigen::ArrayXd::Ones(L); // Mask to select the interesting interval

    bool has_extended_info = il.bottomRows(L / 2).sum() >= 0.9; // Both topological and thermal states have >= 1 bits beyond L/2.
    bool decay_starts_at_1 = il[1] >= il[0] and il[1] > il[2];

    if(has_extended_info) mask.bottomRows(L / 2 - 1) = nan; // If there is a topological bit at l ~ L? we should only consider l in [0,L/2]
    if(decay_starts_at_1) mask[0] = nan;                    // If the decay starts at l==1, take the geometric distribution starting at l==1
    mask = (il < 1e-15).select(nan, mask);                  // Filter tiny noisy values

    auto il_log      = Eigen::ArrayXd(Eigen::log(il * mask));
    auto il_log_diff = Eigen::ArrayXd(il_log.bottomRows(L - 1) - il_log.topRows(L - 1));
    auto xi          = -1.0 / nanmean(il_log_diff);

    // auto lstart = decay_starts_at_1 ? 1.0 : 0.0;
    // auto istart = decay_starts_at_1 ? 1 : 0;
    // auto y_expd = Eigen::ArrayXd(il[istart] * mask * exp(-(l - lstart) / xi));
    // tools::log->info("info_xi_from_expd: xi            : {:.16f}", xi);
    // tools::log->info("info_xi_from_expd: y_info        : {::.16f}", il);
    // tools::log->info("info_xi_from_expd: y_expfit      : {::.16f}", y_expd);
    // tools::log->info("info_xi_from_expd: il_log        : {::.6f}", il_log);
    // tools::log->info("info_xi_from_expd: il_log_diff   : {::.6f}", il_log_diff);
    // tools::log->info("info_xi_from_expd: il_log_diff⁻¹ : {::.6f}", il_log_diff.cwiseInverse());
    return xi;
}

double tools::finite::measure::information_xi_from_exp_fit(const StateFinite &state, InfoPolicy ip) {
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::ArrayXd il  = information_per_scale(state, ip);
    if(il.size() <= 3) return nan;
    auto           L    = il.size();
    Eigen::ArrayXd l    = Eigen::ArrayXd::LinSpaced(L, 0.0, static_cast<double>(L - 1l));
    Eigen::ArrayXd mask = Eigen::ArrayXd::Ones(L); // Mask to select the interesting interval

    bool has_extended_info = il.bottomRows(L / 2).sum() >= 0.9; // Both topological and thermal states have >= 1 bits beyond L/2.
    bool decay_starts_at_1 = il[1] >= il[0] and il[1] > il[2];

    if(has_extended_info) mask.bottomRows(L / 2 - 1) = 0; // If there is a topological bit at l ~ L? we should only consider l in [0,L/2]
    if(decay_starts_at_1) mask[0] = 0;                    // If the decay starts at l==1, take the geometric distribution starting at l==1
    mask = (il < 1e-15).select(0, mask);                  // Filter tiny noisy values

    // Collect the well-defined points in the domain
    auto Y = std::vector<double>();
    auto X = std::vector<double>();
    for(const auto &[i, m] : iter::enumerate(mask))
        if(m == 1) {
            Y.emplace_back(std::log(il[static_cast<long>(i)]));
            X.emplace_back(l[static_cast<long>(i)]);
        }
    tools::log->info("X : {}", X);
    tools::log->info("Y : {::.16f}", Y);
    auto linfit = stat::linearFit(X, Y);
    return -1.0 / linfit.slope;
}
