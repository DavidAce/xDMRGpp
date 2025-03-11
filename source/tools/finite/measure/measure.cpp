
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
#include "math/tenx.h"
#include "qm/mpo.h"
#include "qm/spin.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/common/split.h"
#include "tools/finite/mpo.h"
#include "tools/finite/mps.h"
#include "tools/finite/ops.h"
#include <fmt/ranges.h>

namespace settings {
    constexpr bool debug_expval = false;
}

size_t tools::finite::measure::length(const TensorsFinite &tensors) { return tensors.get_length(); }
size_t tools::finite::measure::length(const StateFinite &state) { return state.get_length(); }

double tools::finite::measure::norm(const StateFinite &state, bool full) {
    if(state.measurements.norm) return state.measurements.norm.value();
    cx64 norm;
    auto t_norm = tid::tic_scope("norm", tid::level::highest);
    if(not full) {
        // We know the all sites are normalized. We can check that the current position is normalized
        const auto  pos = std::clamp(state.get_position<long>(), 0l, state.get_length<long>());
        const auto &mps = state.get_mps_site(pos);
        tools::log->trace("Measuring norm using site {} with dimensions {}", pos, mps.dimensions());
        norm = tools::common::contraction::contract_mps_norm(mps.get_M());
    } else {
        tools::log->trace("Measuring norm on full chain");
        Eigen::Tensor<cx64, 2> chain;
        Eigen::Tensor<cx64, 2> temp;
        bool                   first   = true;
        auto                  &threads = tenx::threads::get();

        for(const auto &mps : state.mps_sites) {
            const auto &M = mps->get_M();
            if(first) {
                chain = tools::common::contraction::contract_mps_partial<std::array{0l, 1l}>(M);
                first = false;
                continue;
            }
            temp.resize(tenx::array2{M.dimension(2), M.dimension(2)});
            temp.device(*threads->dev) = chain.contract(M, tenx::idx({0}, {1})).contract(M.conjugate(), tenx::idx({0, 1}, {1, 0}));

            chain = std::move(temp);
        }
        norm = tenx::MatrixMap(chain).trace();
    }
    if(std::abs(norm - 1.0) > settings::precision::max_norm_error) tools::log->debug("norm: far from unity: {:.16f}{:+.16f}i", norm.real(), norm.imag());
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}

long tools::finite::measure::bond_dimension_current(const StateFinite &state) {
    if(state.measurements.bond_dim) return state.measurements.bond_dim.value();
    if(state.has_center_point())
        state.measurements.bond_dim = state.current_bond().dimension(0);
    else
        state.measurements.bond_dim = 1;
    return state.measurements.bond_dim.value();
}

long tools::finite::measure::bond_dimension_midchain(const StateFinite &state) {
    if(state.measurements.bond_mid) return state.measurements.bond_mid.value();
    state.measurements.bond_mid = state.get_midchain_bond().dimension(0);
    return state.measurements.bond_mid.value();
}

std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite &state, size_t pos) {
    auto t_bond = tid::tic_scope("bond_dimensions", tid::level::highest);
    assert(pos < state.get_length<size_t>());
    return {state.mps_sites[pos]->get_chiL(), state.mps_sites[pos]->get_chiR()};
}

std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite &state) {
    if(state.measurements.bond_dimensions) return state.measurements.bond_dimensions.value();
    auto              t_bond = tid::tic_scope("bond_dimensions", tid::level::highest);
    std::vector<long> bond_dimensions;
    bond_dimensions.reserve(state.get_length() + 1);
    if(not state.has_center_point()) bond_dimensions.emplace_back(state.mps_sites.front()->get_chiL());
    for(const auto &mps : state.mps_sites) {
        bond_dimensions.emplace_back(mps->get_L().dimension(0));
        if(mps->isCenter()) { bond_dimensions.emplace_back(mps->get_LC().dimension(0)); }
    }
    if(bond_dimensions.size() != state.get_length() + 1) throw except::logic_error("bond_dimensions.size() should be length+1");
    state.measurements.bond_dimensions = bond_dimensions;
    return state.measurements.bond_dimensions.value();
}

std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite &state) {
    // Here we get the bond dimensions of the bonds that were merged into the full state in the last step
    // For instance, if the active sites are {2,3,4,5,6} this returns the 4 bonds connecting {2,3}, {3,4}, {4,5} and {5,6}
    // If active sites is just {4}, it returns the bond between {4,5} when going left or right, or between {3,4} wen going right to left
    auto t_chi = tid::tic_scope("bond_merged", tid::level::highest);
    if(state.active_sites.empty()) return {};
    if(state.active_sites.size() == 1) {
        // In single-site DMRG the active site is a center "AC" site:
        //  * Going left-to-right, the forward (right) bond is expanded, and this same bond is truncated when merging
        //  * Going right-to-left, the forward (left) bond is expanded (L), but LC is still the one truncated when merging.
        if(state.get_direction() == +1) return {state.get_mps_site(state.active_sites[0]).get_chiR()};
        if(state.get_direction() == -1) return {state.get_mps_site(state.active_sites[0]).get_chiL()};
    }
    if(state.active_sites.size() == 2) return {state.get_mps_site(state.active_sites[0]).get_chiR()};
    std::vector<long> bond_dimensions;
    for(const auto &pos : state.active_sites) {
        if(&pos == &state.active_sites.front()) continue;
        const auto &mps = state.get_mps_site(pos);
        bond_dimensions.push_back(mps.get_chiL());
    }
    return bond_dimensions;
}

double tools::finite::measure::entanglement_entropy(const Eigen::Tensor<cx64, 1> &L) {
    auto t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    auto S     = Eigen::Tensor<cx64, 0>(-L.square().contract(L.square().log().eval(), tenx::idx({0}, {0})));
    return std::abs(S(0));
}

double tools::finite::measure::entanglement_entropy_current(const StateFinite &state) {
    if(state.measurements.entanglement_entropy_current) return state.measurements.entanglement_entropy_current.value();
    if(state.has_center_point()) {
        state.measurements.entanglement_entropy_current = entanglement_entropy(state.current_bond());
    } else
        state.measurements.entanglement_entropy_current = 0;
    return state.measurements.entanglement_entropy_current.value();
}

double tools::finite::measure::entanglement_entropy_midchain(const StateFinite &state) {
    if(state.measurements.entanglement_entropy_midchain) return state.measurements.entanglement_entropy_midchain.value();
    state.measurements.entanglement_entropy_midchain = entanglement_entropy(state.get_midchain_bond());
    return state.measurements.entanglement_entropy_midchain.value();
}

std::vector<double> tools::finite::measure::entanglement_entropies(const StateFinite &state) {
    if(state.measurements.entanglement_entropies) return state.measurements.entanglement_entropies.value();
    auto                t_ent = tid::tic_scope("neumann_entropy", tid::level::highest);
    std::vector<double> entanglement_entropies;
    entanglement_entropies.reserve(state.get_length() + 1);
    if(not state.has_center_point()) entanglement_entropies.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        entanglement_entropies.emplace_back(entanglement_entropy(mps->get_L()));
        if(mps->isCenter()) {
            entanglement_entropies.emplace_back(entanglement_entropy(mps->get_LC()));
            state.measurements.entanglement_entropy_current = entanglement_entropies.back();
        }
    }
    if(entanglement_entropies.size() != state.get_length() + 1) throw except::logic_error("entanglement_entropies.size() should be length+1");
    if(entanglement_entropies.front() != 0.0) throw except::logic_error("First entropy should be 0. Got: {:.16f}", entanglement_entropies.front());
    if(entanglement_entropies.back() != 0.0) throw except::logic_error("Last entropy should be 0. Got: {:.16f}", entanglement_entropies.back());
    state.measurements.entanglement_entropy_midchain = entanglement_entropies[state.get_length<size_t>() / 2];
    state.measurements.entanglement_entropies        = entanglement_entropies;
    return state.measurements.entanglement_entropies.value();
}

std::vector<double> tools::finite::measure::renyi_entropies(const StateFinite &state, double q) {
    auto inf = std::numeric_limits<double>::infinity();
    if(q == 1.0) return entanglement_entropies(state);
    if(q == 2.0 and state.measurements.renyi_2) return state.measurements.renyi_2.value();
    if(q == 3.0 and state.measurements.renyi_3) return state.measurements.renyi_3.value();
    if(q == 4.0 and state.measurements.renyi_4) return state.measurements.renyi_4.value();
    if(q == inf and state.measurements.renyi_inf) return state.measurements.renyi_inf.value();
    auto                t_ren = tid::tic_scope("renyi_entropy", tid::level::highest);
    std::vector<double> renyi_q;
    renyi_q.reserve(state.get_length() + 1);
    if(not state.has_center_point()) renyi_q.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        const auto            &L = mps->get_L();
        Eigen::Tensor<cx64, 0> RE;
        if(q == inf)
            RE(0) = -2.0 * std::log(L(0));
        else
            RE = 1.0 / cx64(1.0 - q, 0.0) * L.pow(cx64(2.0 * q, 0.0)).sum().log();
        renyi_q.emplace_back(std::abs(RE(0)));
        if(mps->isCenter()) {
            const auto &LC = mps->get_LC();
            if(q == inf)
                RE(0) = -2.0 * std::log(LC(0));
            else
                RE = 1.0 / cx64(1.0 - q, 0.0) * LC.pow(cx64(2.0 * q, 0.0)).sum().log();
            renyi_q.emplace_back(std::abs(RE(0)));
        }
    }
    if(renyi_q.size() != state.get_length() + 1) throw except::logic_error("renyi_q.size() should be length+1");
    if(q == 2.0) {
        state.measurements.renyi_2 = renyi_q;
        return state.measurements.renyi_2.value();
    }
    if(q == 3.0) {
        state.measurements.renyi_3 = renyi_q;
        return state.measurements.renyi_3.value();
    }
    if(q == 4.0) {
        state.measurements.renyi_4 = renyi_q;
        return state.measurements.renyi_4.value();
    }
    if(q == inf) {
        state.measurements.renyi_inf = renyi_q;
        return state.measurements.renyi_inf.value();
    }
    return renyi_q;
}

double tools::finite::measure::renyi_entropy_midchain(const StateFinite &state, double q) {
    auto inf = std::numeric_limits<double>::infinity();
    if(q == 1.0) return entanglement_entropy_midchain(state);
    auto                   t_ren = tid::tic_scope("renyi_entropy_midchain", tid::level::highest);
    auto                  &LC    = state.get_midchain_bond();
    Eigen::Tensor<cx64, 0> renyi_q;
    if(q == inf)
        renyi_q(0) = -2.0 * std::log(LC(0));
    else
        renyi_q = 1.0 / cx64(1.0 - q, 0.0) * LC.pow(cx64(2.0 * q, 0.0)).sum().log();
    return std::abs(renyi_q(0));
}

std::array<double, 3> tools::finite::measure::spin_components(const StateFinite &state) {
    if(state.measurements.spin_components) return state.measurements.spin_components.value();
    double spin_x                      = measure::spin_component(state, qm::spin::half::sx);
    double spin_y                      = measure::spin_component(state, qm::spin::half::sy);
    double spin_z                      = measure::spin_component(state, qm::spin::half::sz);
    state.measurements.spin_components = {spin_x, spin_y, spin_z};
    return state.measurements.spin_components.value();
}

double tools::finite::measure::spin_component(const StateFinite &state, const Eigen::Matrix2cd &paulimatrix) {
    auto t_spn       = tid::tic_scope("spin", tid::level::highest);
    auto [mpo, L, R] = qm::mpo::pauli_mpo(paulimatrix);
    Eigen::Tensor<cx64, 3> temp;
    for(const auto &mps : state.mps_sites) {
        tools::common::contraction::contract_env_mps_mpo(temp, L, mps->get_M(), mpo);
        L = temp;
    }

    if(L.dimensions() != R.dimensions()) throw except::runtime_error("spin_component(): L and R dimension mismatch");
    Eigen::Tensor<cx64, 0> spin_tmp = L.contract(R, tenx::idx({0, 1, 2}, {0, 1, 2}));
    double                 spin     = std::real(spin_tmp(0));
    return spin;
}

double tools::finite::measure::spin_component(const StateFinite &state, std::string_view axis) {
    if(axis.find('x') != std::string_view::npos) return measure::spin_component(state, qm::spin::half::sx);
    if(axis.find('y') != std::string_view::npos) return measure::spin_component(state, qm::spin::half::sy);
    if(axis.find('z') != std::string_view::npos) return measure::spin_component(state, qm::spin::half::sz);
    throw except::logic_error("unexpected axis [{}]", axis);
}

double tools::finite::measure::spin_alignment(const StateFinite &state, std::string_view axis) {
    if(not qm::spin::half::is_valid_axis(axis)) throw except::logic_error("unexpected axis [{}]", axis);
    auto spin_component_along_axis = tools::finite::measure::spin_component(state, qm::spin::half::get_pauli(axis));
    auto sign                      = qm::spin::half::get_sign(axis);
    return sign * spin_component_along_axis;
}

int tools::finite::measure::spin_sign(const StateFinite &state, std::string_view axis) {
    // The sign on the axis string is ignored here
    auto spin_component_along_axis = tools::finite::measure::spin_component(state, qm::spin::half::get_pauli(axis));
    return num::sign(spin_component_along_axis);
}

std::vector<double> tools::finite::measure::truncation_errors(const StateFinite &state) {
    if(state.measurements.truncation_errors) return state.measurements.truncation_errors.value();
    auto                t_chi = tid::tic_scope("trunc", tid::level::highest);
    std::vector<double> truncation_errors;
    if(not state.has_center_point()) truncation_errors.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        truncation_errors.emplace_back(mps->get_truncation_error());
        if(mps->isCenter()) truncation_errors.emplace_back(mps->get_truncation_error_LC());
    }
    if(truncation_errors.size() != state.get_length() + 1) throw except::logic_error("truncation_errors.size() should be length+1");
    state.measurements.truncation_errors = truncation_errors;
    return state.measurements.truncation_errors.value();
}

std::vector<double> tools::finite::measure::truncation_errors_active(const StateFinite &state) {
    // Here we get the truncation erros of the bonds that were merged into the full state in the last step
    // For instance, if the active sites are {2,3,4,5,6} this returns the 4 bonds connecting {2,3}, {3,4}, {4,5} and {5,6}
    // If active sites is just {4}, it returns the bond between {4,5} when going right, and {3,4} when going left.
    if(state.active_sites.empty()) return {};
    if(state.active_sites.size() == 1) {
        // In single-site DMRG the active site is a center "AC" site:
        // If we do forward expansion:aCq!eau3

        //  * Going left-to-right, the right bond (LC) is truncated after optimization
        //  * Going right-to-left, the left bond (L) is truncated after optimization.
        if(state.get_direction() == +1) return {state.get_mps_site(state.active_sites[0]).get_truncation_error_LC()};
        if(state.get_direction() == -1) return {state.get_mps_site(state.active_sites[0]).get_truncation_error()};
    }
    if(state.active_sites.size() == 2) return {state.get_mps_site(state.active_sites[0]).get_truncation_error_LC()};
    std::vector<double> truncation_errors;
    for(const auto &pos : state.active_sites) {
        if(&pos == &state.active_sites.front()) continue;
        const auto &mps = state.get_mps_site(pos);
        truncation_errors.push_back(mps.get_truncation_error());
    }
    return truncation_errors;
}

template<typename Scalar>
Eigen::Tensor<cx64, 1> tools::finite::measure::mps2tensor(const std::vector<std::unique_ptr<MpsSite>> &mps_sites, std::string_view name) {
    if constexpr(std::is_same_v<Scalar, cx64>) {
        bool all_real = std::all_of(mps_sites.begin(), mps_sites.end(), [](const auto &mps) -> bool { return mps->is_real(); });
        if(all_real) return mps2tensor<fp64>(mps_sites, name);
    }

    auto spindims = std::vector<long>();
    auto bonddims = std::vector<long>();
    for(auto &mps : mps_sites) {
        spindims.emplace_back(mps->spin_dim());
        bonddims.emplace_back(num::prod(spindims) * mps->get_chiR());
    }
    long  memsize = bonddims.empty() ? 0 : *std::max_element(bonddims.begin(), bonddims.end());
    auto  statev  = Eigen::Tensor<Scalar, 1>(memsize);
    auto  off1    = std::array<long, 1>{0};
    auto  ext1    = std::array<long, 1>{1};
    auto  ext2    = std::array<long, 2>{1, 1};
    auto &threads = tenx::threads::get();
    statev.slice(off1, ext1).setConstant(1.0);
    // For each site that we contract, the state vector grows by mps->spin_dim()
    // If 4 spin1/2 have been contracted, the state vector could have size 16x7 if the last chi was 7.
    // Then contracting the next site, with dimensions 2x7x9 will get you a 16x2x9 tensor.
    // Lastly, one should reshape it back into a 32 x 9 state vector

    for(auto &mps : mps_sites) {
        auto temp = Eigen::Tensor<Scalar, 2>(statev.slice(off1, ext1).reshape(ext2)); // Make a temporary copy of the state vector
        ext1      = {mps->spin_dim() * temp.dimension(0) * mps->get_chiR()};
        ext2      = {mps->spin_dim() * temp.dimension(0), mps->get_chiR()};
        if(ext1[0] > memsize) throw except::logic_error("mps2tensor [{}]: size of ext1[0] > memsize", name);
        //        tools::log->info("Contracting temp {} | M[{}]:{} | off1 {} | ext1 {} | ext2 {}", temp.dimensions(), mps->get_position(),
        //        mps->get_M().dimensions(),
        //                         off1, ext1, ext2);
        if constexpr(std::is_same_v<Scalar, cx64>) {
            statev.slice(off1, ext1).device(*threads->dev) = temp.contract(mps->get_M(), tenx::idx({1}, {1})).reshape(ext1);
        } else {
            Eigen::Tensor<fp64, 3> M                       = mps->get_M().real();
            statev.slice(off1, ext1).device(*threads->dev) = temp.contract(M, tenx::idx({1}, {1})).reshape(ext1);
        }
    }
    // Finally, we view a slice of known size 2^L
    statev      = statev.slice(off1, ext1);
    double norm = tenx::norm(statev);
    if(std::abs(norm - 1.0) > settings::precision::max_norm_error) { tools::log->warn("mps2tensor [{}]: Norm far from unity: {:.16f}", name, norm); }
    return statev.template cast<cx64>();
}

Eigen::Tensor<cx64, 1> tools::finite::measure::mps2tensor(const StateFinite &state) { return mps2tensor(state.mps_sites, state.get_name()); }

cx64 tools::finite::measure::expval_hamiltonian(const Eigen::Tensor<cx64, 3> &mps, const ModelFinite &model, const EdgesFinite &edges) {
    auto mpo  = model.get_mpo_active();
    auto env  = edges.get_ene_active();
    auto t_H2 = tid::tic_scope("H", tid::level::highest);

    return tools::finite::measure::expectation_value(mps, mps, mpo, env);
}

cx64 tools::finite::measure::expval_hamiltonian(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    auto mps = state.get_mps_active();
    auto mpo = model.get_mpo_active();
    auto env = edges.get_ene_active();
    auto t_H = tid::tic_scope("H", tid::level::highest);
    return tools::finite::measure::expectation_value(mps, mps, mpo, env);
}

cx64 tools::finite::measure::expval_hamiltonian(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                                const env_pair<const EnvEne &> &envs) {
    auto t_H = tid::tic_scope("H", tid::level::highest);
    return tools::finite::measure::expectation_value(mps, mps, mpo_refs, envs);
}

cx64 tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<cx64, 3> &mps, const ModelFinite &model, const EdgesFinite &edges) {
    auto mpo = model.get_mpo_active();
    auto env = edges.get_var_active();
    auto t_H = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value(mps, mps, mpo, env);
}
cx64 tools::finite::measure::expval_hamiltonian_squared(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    auto mps  = state.get_mps_active();
    auto mpo  = model.get_mpo_active();
    auto env  = edges.get_var_active();
    auto t_H2 = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value(mps, mps, mpo, env);
}
cx64 tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                                        const env_pair<const EnvVar &> &envs) {
    auto t_H2 = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value(mps, mps, mpo_refs, envs);
}

cx64 tools::finite::measure::expval_hamiltonian(const TensorsFinite &tensors) {
    return tools::finite::measure::expval_hamiltonian(*tensors.state, *tensors.model, *tensors.edges);
}

cx64 tools::finite::measure::expval_hamiltonian_squared(const TensorsFinite &tensors) {
    return tools::finite::measure::expval_hamiltonian_squared(*tensors.state, *tensors.model, *tensors.edges);
}

double tools::finite::measure::energy_minus_energy_shift(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                                         MeasurementsTensorsFinite *measurements) {
    if(measurements != nullptr and measurements->energy_minus_energy_shift) {
        if constexpr(!settings::debug_expval) {
            // Return the cache hit when not debugging. Otherwise, check that it is correct!
            // tools::log->trace("energy_minus_energy_shift: cache hit: {:.16f}", measurements->energy_minus_energy_shift.value());
            return measurements->energy_minus_energy_shift.value();
        }
    }
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    auto t_ene = tid::tic_scope("ene", tid::level::highest);
    if constexpr(settings::debug) tools::log->trace("Measuring energy: sites {}", state.active_sites);
    auto e_minus_ered = expval_hamiltonian(state, model, edges);
    if constexpr(settings::debug_expval) {
        const auto &multisite_mps = state.get_multisite_mps<cx64>();
        const auto &multisite_mpo = model.get_multisite_mpo<cx64>();
        const auto &multisite_env = edges.get_multisite_env_ene_blk();
        auto        edbg          = tools::common::contraction::expectation_value(multisite_mps, multisite_mpo, multisite_env.L, multisite_env.R);
        tools::log->trace("e_minus_ered: {:.16f}{:+.16f}i", e_minus_ered.real(), e_minus_ered.imag());
        tools::log->trace("e_minus_edbg: {:.16f}{:+.16f}i", edbg.real(), edbg.imag());
        if(measurements != nullptr and measurements->energy_minus_energy_shift) {
            tools::log->trace("e_minus_ehit: {:.16f}", measurements->energy_minus_energy_shift.value());
            assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < 1e-14);
        }
        assert(std::abs(e_minus_ered - edbg) < 1e-14);
    }

    assert(std::abs(std::imag(e_minus_ered)) < 1e-10);
    if(measurements != nullptr) measurements->energy_minus_energy_shift = std::real(e_minus_ered);
    return std::real(e_minus_ered);
}

double tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<cx64, 3> &multisite_mps, const ModelFinite &model, const EdgesFinite &edges,
                                                         std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite *measurements) {
    if(measurements != nullptr and measurements->energy_minus_energy_shift) {
        if constexpr(!settings::debug_expval) {
            // Return the cache hit when not debugging. Otherwise, check that it is correct!
            // tools::log->trace("energy_minus_energy_shift: cache hit: {:.16f}", measurements->energy_minus_energy_shift.value());
            return measurements->energy_minus_energy_shift.value();
        }
    }
    auto t_ene = tid::tic_scope("ene", tid::level::highest);
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    assert(num::all_equal(model.active_sites, edges.active_sites));
    // Check if we can contract directly or if we need to use the split method
    // Normally it's only worth splitting the multisite mps when it has more than 3 sites
    auto e_minus_ered = cx64(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    if(model.active_sites.size() <= 3) {
        // Contract directly
        const auto &mpo = model.get_multisite_mpo<cx64>();
        const auto &env = edges.get_multisite_env_ene_blk();
        if constexpr(settings::debug_expval)
            tools::log->trace("Measuring energy: multisite_mps dims {} | model sites {} dims {} | edges sites {} dims [L{} R{}]", multisite_mps.dimensions(),
                              model.active_sites, mpo.dimensions(), edges.active_sites, env.L.dimensions(), env.R.dimensions());
        e_minus_ered = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
        if constexpr(settings::debug_expval) {
            // Split the multisite mps first
            const auto mpos = model.get_mpo_active();
            const auto envs = edges.get_ene_active();
            const auto edbg = tools::finite::measure::expectation_value(multisite_mps, mpos, envs, svd_cfg);
            tools::log->trace("e_minus_ered: {:.16f}{:+.16f}i", e_minus_ered.real(), e_minus_ered.imag());
            tools::log->trace("e_minus_edbg: {:.16f}{:+.16f}i", edbg.real(), edbg.imag());
            if(measurements != nullptr and measurements->energy_minus_energy_shift) {
                tools::log->trace("e_minus_ehit: {:.16f}", measurements->energy_minus_energy_shift.value());
                assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < 1e-14);
            }
            assert(std::abs(e_minus_ered - edbg) < 1e-14);
        }
    } else {
        model.clear_cache();
        // Split the multisite mps first
        const auto mpos = model.get_mpo_active();
        const auto envs = edges.get_ene_active();
        tools::log->trace("Measuring energy: multisite_mps dims {} | sites {} | eshift {:.16f} | norm {:.16f}", multisite_mps.dimensions(), model.active_sites,
                          model.get_energy_shift_mpo(), tenx::norm(multisite_mps));
        e_minus_ered = tools::finite::measure::expectation_value(multisite_mps, mpos, envs, svd_cfg);
        if constexpr(settings::debug_expval) {
            const auto &mpo  = model.get_multisite_mpo<cx64>();
            const auto &env  = edges.get_multisite_env_ene_blk();
            const auto  edbg = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
            tools::log->trace("e_minus_ered: {:.16f}{:+.16f}i", e_minus_ered.real(), e_minus_ered.imag());
            tools::log->trace("e_minus_edbg: {:.16f}{:+.16f}i", edbg.real(), edbg.imag());
            if(measurements != nullptr and measurements->energy_minus_energy_shift) {
                tools::log->trace("e_minus_ehit: {:.16f}", measurements->energy_minus_energy_shift.value());
                assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < 1e-14);
            }
            assert(std::abs(e_minus_ered - edbg) < 1e-10);
        }
    }
    assert(std::abs(std::imag(e_minus_ered)) < 1e-10);
    if(measurements != nullptr) measurements->energy_minus_energy_shift = std::real(e_minus_ered);
    return std::real(e_minus_ered);
}

double tools::finite::measure::energy(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges, MeasurementsTensorsFinite *measurements) {
    if(measurements != nullptr and measurements->energy) return measurements->energy.value();
    // This measures the actual energy of the system regardless of the energy shift in the MPO's
    // If they are shifted, then
    //      "Actual energy" = (E - E_shift) + E_shift = (~0) + E_shift = E
    // Else
    //      "Actual energy" = (E - E_shift) + E_shift = E  + 0 = E
    auto e_minus_eshift = tools::finite::measure::energy_minus_energy_shift(state, model, edges, measurements);
    auto eshift         = model.get_energy_shift_mpo();
    auto energy         = e_minus_eshift + eshift;
    if(measurements != nullptr) measurements->energy = energy;
    return energy;
}

double tools::finite::measure::energy(const Eigen::Tensor<cx64, 3> &multisite_mps, const ModelFinite &model, const EdgesFinite &edges,
                                      std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite *measurements) {
    if(measurements != nullptr and measurements->energy) return measurements->energy.value();
    // This measures the actual energy of the system regardless of the energy shift in the MPO's
    // If they are shifted, then
    //      "Actual energy" = (E - E_shift) + E_shift = (~0) + E_shift = E
    // Else
    //      "Actual energy" = (E - E_shift) + E_shift = E  + 0 = E
    auto e_minus_eshift = tools::finite::measure::energy_minus_energy_shift(multisite_mps, model, edges, svd_cfg, measurements);
    auto eshift         = model.get_energy_shift_mpo();
    auto energy         = e_minus_eshift + eshift;
    if(measurements != nullptr) measurements->energy = energy;
    return energy;
}

double tools::finite::measure::energy_variance(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                               MeasurementsTensorsFinite *measurements) {
    // Here we show that the variance calculated with energy-shifted mpo²'s is equivalent to the usual way.
    // If mpo's are shifted in the mpo²:
    //      Var H = <(H-E_shf)²> - <H-E_shf>²     = <H²>  - 2<H>E_shf + E_shf² - (<H> - E_shf)²
    //                                            = H²    - 2*E*E_shf + E_shf² - E² + 2*E*E_shf - E_shf²
    //                                            = H²    - E²
    //      Note that in the last line, H²-E² is a subtraction of two large numbers --> catastrophic cancellation --> loss of precision.
    //      On the other hand Var H = <(H-E_shf)²> - energy_minus_energy_shift² = <(H-E_red)²> - ~dE², where both terms are always  << 1.
    //      The first term is computed from a double-layer of shifted mpo's.
    //      In the second term dE is usually very small, in fact identically zero immediately after an energy-reduction operation,
    //      but may grow if the optimization steps make significant progress refining E. Thus wethe first term is a good approximation to
    //      the variance by itself.
    //
    // Else, if E_shf = 0 (i.e. not shifted) we get the usual formula:
    //      Var H = <(H - 0)²> - <H - 0>² = H² - E²
    if(measurements != nullptr and measurements->energy_variance) return measurements->energy_variance.value();
    assert(not state.active_sites.empty());
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    if constexpr(settings::debug_expval) tools::log->trace("Measuring energy variance: sites {}", state.active_sites);
    auto E  = expval_hamiltonian(state, model, edges);
    auto E2 = E * E;
    auto H2 = expval_hamiltonian_squared(state, model, edges);
    // #pragma message "remove minus one for double parity shift test"
    // H2-=1.0;
    assert(std::abs(std::imag(H2)) < 1e-10);
    double var = std::abs(H2 - E2);
    if constexpr(settings::debug_expval) tools::log->trace("Variance |H2-E2| = |{:.16f} - {:.16f}| = {:.16f}", std::real(H2), E2, var);
    if(measurements != nullptr) measurements->energy_variance = var;
    return var;
}

double tools::finite::measure::energy_variance(const Eigen::Tensor<cx64, 3> &multisite_mps, const ModelFinite &model, const EdgesFinite &edges,
                                               std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite *measurements) {
    // Here we show that the variance calculated with energy-shifted mpo's is equivalent to the usual way.
    // If mpo's are shifted:
    //      Var H = <(H-E_shf)²> - <H-E_shf>²     = <H²>  - 2<H>E_shf + E_shf² - (<H> - E_shf)²
    //                                            = H²    - 2*E*E_shf + E_shf² - E² + 2*E*E_shf - E_shf²
    //                                            = H²    - E²
    //      Note that in the last line, H²-E² is a subtraction of two large numbers --> catastrophic cancellation --> loss of precision.
    //      On the other hand Var H = <(H-E_shf)²> - energy_minus_energy_shift² = <(H-E_red)²> - ~dE², where both terms are always  << 1.
    //      The first term is computed from a double-layer of shifted mpo's.
    //      In the second term dE is usually very small, in fact identically zero immediately after an energy-reduction operation,
    //      but may grow if the optimization steps make significant progress refining E. Thus wethe first term is a good approximation to
    //      the variance by itself.
    //
    // Else, if E_shf = 0 (i.e. not shifted) we get the usual formula:
    //      Var H = <(H - 0)²> - <H - 0>² = H² - E²
    if(measurements != nullptr and measurements->energy_variance) return measurements->energy_variance.value();
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    if(not num::all_equal(model.active_sites, edges.active_sites))
        throw std::runtime_error(
            fmt::format("Could not compute energy variance: active sites are not equal: model {} | edges {}", model.active_sites, edges.active_sites));
    double energy = tools::finite::measure::energy_minus_energy_shift(multisite_mps, model, edges, svd_cfg, measurements);
    double E2     = energy * energy;

    auto t_var = tid::tic_scope("var", tid::level::highest);

    // Check if we can contract directly or if we need to use the split method
    // Normally it's only worth splitting the multisite mps when it has more than 3 sites
    cx64 H2 = cx64(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    if(model.active_sites.size() <= 3) {
        // Direct contraction
        const auto &mpo2 = model.get_multisite_mpo_squared<cx64>();
        const auto &env2 = edges.get_multisite_env_var_blk();
        if constexpr(settings::debug)
            tools::log->trace("Measuring energy variance: state dims {} | model sites {} dims {} | edges sites {} dims [L{} R{}]", multisite_mps.dimensions(),
                              model.active_sites, mpo2.dimensions(), edges.active_sites, env2.L.dimensions(), env2.R.dimensions());

        if(multisite_mps.dimension(0) != mpo2.dimension(2))
            throw std::runtime_error(fmt::format("State and model have incompatible physical dimension: state dim {} | model dim {}",
                                                 multisite_mps.dimension(0), mpo2.dimension(2)));
        H2 = tools::common::contraction::expectation_value(multisite_mps, mpo2, env2.L, env2.R);
    } else {
        // Split the multisite mps first
        const auto mpos = model.get_mpo_active();
        const auto envs = edges.get_var_active();
        tools::log->trace("Measuring energy variance: state dims {} | sites {}", multisite_mps.dimensions(), model.active_sites);
        H2 = tools::finite::measure::expectation_value(multisite_mps, multisite_mps, mpos, envs, svd_cfg);
        if constexpr(settings::debug_expval) {
            const auto &mpo   = model.get_multisite_mpo_squared<cx64>();
            const auto &env   = edges.get_multisite_env_var_blk();
            const auto  H2dbg = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
            tools::log->trace("H2   : {:.16f}{:+.16f}i", H2.real(), H2.imag());
            tools::log->trace("H2dbg: {:.16f}{:+.16f}i", H2dbg.real(), H2dbg.imag());
            assert(std::abs(H2 - H2dbg) < 1e-10);
        }
    }
    assert(std::abs(std::imag(H2)) < 1e-10);
    // #pragma message "remove minus one for double parity shift test"
    // H2-=1.0;
    double var = std::abs(H2 - E2);
    // tools::log->info("Var H = H² - E² = {:.16f} - {:.16f} = {:.16f}", std::real(H2), E2, var);
    if(measurements != nullptr) measurements->energy_variance = var;
    return var;
}

double tools::finite::measure::energy_normalized(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges, double energy_min,
                                                 double energy_max, MeasurementsTensorsFinite *measurements) {
    return (tools::finite::measure::energy(state, model, edges, measurements) - energy_min) / (energy_max - energy_min);
}
double tools::finite::measure::energy_normalized(const Eigen::Tensor<cx64, 3> &multisite_mps, const ModelFinite &model, const EdgesFinite &edges,
                                                 double energy_min, double energy_max, std::optional<svd::config> svd_cfg,
                                                 MeasurementsTensorsFinite *measurements) {
    return (tools::finite::measure::energy(multisite_mps, model, edges, svd_cfg, measurements) - energy_min) / (energy_max - energy_min);
}

extern double tools::finite::measure::energy_shift(const TensorsFinite &tensors) { return tensors.model->get_energy_shift_mpo(); }

double tools::finite::measure::energy_minus_energy_shift(const TensorsFinite &tensors) {
    tensors.assert_edges_ene();
    return energy_minus_energy_shift(*tensors.state, *tensors.model, *tensors.edges, &tensors.measurements);
}

double tools::finite::measure::energy(const TensorsFinite &tensors) {
    if(not tensors.measurements.energy) {
        tensors.assert_edges_ene();
        tensors.measurements.energy = tools::finite::measure::energy(*tensors.state, *tensors.model, *tensors.edges, &tensors.measurements);
    }
    return tensors.measurements.energy.value();
}

double tools::finite::measure::energy_variance(const TensorsFinite &tensors) {
    if(not tensors.measurements.energy_variance) {
        tensors.assert_edges_var();
        tensors.measurements.energy_variance = tools::finite::measure::energy_variance(*tensors.state, *tensors.model, *tensors.edges, &tensors.measurements);
    }
    return tensors.measurements.energy_variance.value();
}

double tools::finite::measure::energy_normalized(const TensorsFinite &tensors, double emin, double emax) {
    tensors.assert_edges_ene();
    return tools::finite::measure::energy_normalized(*tensors.state, *tensors.model, *tensors.edges, emin, emax);
}

double tools::finite::measure::energy_minus_energy_shift(const StateFinite &state, const TensorsFinite &tensors, MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy_minus_energy_shift(state, *tensors.model, *tensors.edges, measurements);
}
double tools::finite::measure::energy(const StateFinite &state, const TensorsFinite &tensors, MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy(state, *tensors.model, *tensors.edges, measurements);
}

double tools::finite::measure::energy_variance(const StateFinite &state, const TensorsFinite &tensors, MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy_variance(state, *tensors.model, *tensors.edges, measurements);
}

double tools::finite::measure::energy_normalized(const StateFinite &state, const TensorsFinite &tensors, double emin, double emax,
                                                 MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy_normalized(state, *tensors.model, *tensors.edges, emin, emax, measurements);
}

double tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<cx64, 3> &mps, const TensorsFinite &tensors, std::optional<svd::config> svd_cfg,
                                                         MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy_minus_energy_shift(mps, *tensors.model, *tensors.edges, svd_cfg, measurements);
}
double tools::finite::measure::energy(const Eigen::Tensor<cx64, 3> &mps, const TensorsFinite &tensors, std::optional<svd::config> svd_cfg,
                                      MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy(mps, *tensors.model, *tensors.edges, svd_cfg, measurements);
}

double tools::finite::measure::energy_variance(const Eigen::Tensor<cx64, 3> &mps, const TensorsFinite &tensors, std::optional<svd::config> svd_cfg,
                                               MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy_variance(mps, *tensors.model, *tensors.edges, svd_cfg, measurements);
}

double tools::finite::measure::energy_normalized(const Eigen::Tensor<cx64, 3> &mps, const TensorsFinite &tensors, double emin, double emax,
                                                 std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite *measurements) {
    return tools::finite::measure::energy_normalized(mps, *tensors.model, *tensors.edges, emin, emax, svd_cfg, measurements);
}

double tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const Eigen::Tensor<cx64, 4> &mpo, const Eigen::Tensor<cx64, 3> &envL,
                                             const Eigen::Tensor<cx64, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpo, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    // auto E  = tools::common::contraction::expectation_value(mps, mpo, envL, envR);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}

double tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<Eigen::Tensor<cx64, 4>> &mpos,
                                             const Eigen::Tensor<cx64, 3> &envL, const Eigen::Tensor<cx64, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpos, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    // auto E  = tools::common::contraction::expectation_value(mps, mpos, envL, envR);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}

double tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                             const env_pair<const EnvEne &> &envs) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto mpo_vec = std::vector<Eigen::Tensor<cx64, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().MPO());
    return residual_norm(mps, mpo_vec, envs.L.get_block(), envs.R.get_block());
}

double tools::finite::measure::residual_norm(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                             const env_pair<const EnvVar &> &envs) {
    // Calculate the residual_norm r = |H²v - E²v|
    auto mpo_vec = std::vector<Eigen::Tensor<cx64, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().MPO2());
    return residual_norm(mps, mpo_vec, envs.L.get_block(), envs.R.get_block());
}

double tools::finite::measure::residual_norm_H1(const TensorsFinite &tensors) {
    const auto &mps = tensors.get_state().get_multisite_mps<cx64>();
    const auto &mpo = tensors.get_model().get_mpo_active();
    const auto &env = tensors.get_edges().get_ene_active();
    return residual_norm(mps, mpo, env);
}

double tools::finite::measure::residual_norm_H2(const TensorsFinite &tensors) {
    const auto &mps = tensors.get_state().get_multisite_mps<cx64>();
    const auto &mpo = tensors.get_model().get_mpo_active();
    const auto &env = tensors.get_edges().get_var_active();
    return residual_norm(mps, mpo, env);
}

double tools::finite::measure::residual_norm_full(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    // Calculate the residual_norm r = |Hv - Ev|, where H is the full Hamiltonian and v is the full mps
    // Note that the full residual norm is equal to the sqrt(Var(H)) = Std(H)
    tools::log->info("Calculating residual norm with full system");
    auto sites = num::range<size_t>(0, state.get_length());
    auto t     = state.get_multisite_mps<cx64>(sites);
    std::vector<Eigen::Tensor<cx64,4>> mpos;
    for(const auto & mpo : model.get_mpo(sites)) mpos.emplace_back(mpo.get().MPO());
    auto envs  = edges.get_env_ene_blk(sites.front(), sites.back());
    auto Ht    = tools::common::contraction::matrix_vector_product(t, mpos, envs.L, envs.R);
    auto Hv    = tenx::VectorMap(Ht);
    auto v     = tenx::VectorMap(t);
    tools::log->info("Calculating (Hv - v.dot(Hv) * v).norm() | v.size() == {} | Hv.size() == {}", v.size(), Hv.size());
    if(v.size() != Hv.size()) throw except::logic_error("Size mismatch: v.size() == {} | Hv.size() == {}", v.size(), Hv.size());
    return (Hv - v.dot(Hv) * v).norm();
}

cx64 tools::finite::measure::expectation_value(const StateFinite &state, const std::vector<LocalObservableOp> &ops) {
    if(state.mps_sites.empty()) throw std::runtime_error("expectation_value: state.mps_sites is empty");
    if(ops.empty()) throw std::runtime_error("expectation_value: ops is empty");
    if constexpr(settings::debug_expval) tools::log->trace("Measuring expectation_value <ops>");
    auto d0    = state.mps_sites.front()->get_chiL();
    auto chain = tenx::TensorCast(Eigen::MatrixXcd::Identity(d0, d0));
    if constexpr(settings::debug) {
        if(d0 != 1) tools::log->warn("expectation_value: chiL is not 1");
    }
    auto &threads = tenx::threads::get();
    for(const auto &mps : state.mps_sites) {
        Eigen::Tensor<cx64, 3> M   = mps->get_M();
        const auto             pos = mps->get_position<long>();
        for(const auto &op : iter::reverse(ops)) {
            // Reverse to apply operators right to left, if they are on the same position
            // i.e. ABC|psi> should apply C first and A last (if A and C are on the same site).
            if(op.used or op.pos != pos) continue;
            auto temp                  = Eigen::Tensor<cx64, 3>(op.op.dimension(0), M.dimension(1), M.dimension(2));
            temp.device(*threads->dev) = op.op.contract(M, tenx::idx({1}, {0}));
            M                          = std::move(temp);
            op.used                    = true;
        }
        auto temp                  = Eigen::Tensor<cx64, 2>(M.dimension(2), M.dimension(2));
        temp.device(*threads->dev) = chain.contract(M, tenx::idx({0}, {1})).contract(mps->get_M().conjugate(), tenx::idx({0, 1}, {1, 0}));
        chain                      = std::move(temp);
    }

    Eigen::Tensor<cx64, 0> expval = chain.trace();
    return expval.coeff(0);
}

cx64 tools::finite::measure::expectation_value(const StateFinite &state, const std::vector<LocalObservableMpo> &mpos) {
    if(state.mps_sites.empty()) throw std::runtime_error("expectation_value: state.mps_sites is empty");
    if(mpos.empty()) throw std::runtime_error("expectation_value: obs is empty");

    // Generate a string of mpos for each site. If a site has no local observable given, insert an identity MPO there.
    auto mpodims = mpos.front().mpo.dimensions();

    for(auto &ob : mpos) {
        if(ob.mpo.dimension(2) != ob.mpo.dimension(3)) throw except::runtime_error("expectation_value: given mpo's of unequal spin dimension up and down");
        if(ob.mpo.dimension(0) != ob.mpo.dimension(1)) throw except::runtime_error("expectation_value: given mpo's of unequal bond dimension left and right");
        if(ob.mpo.dimensions() != mpodims) throw except::runtime_error("expectation_value: given mpo's of unequal dimensions");
    }

    // Create compatible edges
    Eigen::Tensor<cx64, 1> Ledge(mpodims[0]); // The left  edge
    Eigen::Tensor<cx64, 1> Redge(mpodims[1]); // The right edge
    Ledge(mpodims[0] - 1) = 1;
    Redge(0)              = 1;
    Eigen::Tensor<cx64, 3> Ledge3, Redge3;
    {
        auto mpsdims = state.mps_sites.front()->dimensions();
        long mpsDim  = mpsdims[1];
        long mpoDim  = mpodims[0];
        Ledge3.resize(tenx::array3{mpsDim, mpsDim, mpoDim});
        Ledge3.setZero();
        for(long i = 0; i < mpsDim; i++) {
            std::array<long, 1> extent1                     = {mpoDim};
            std::array<long, 3> offset3                     = {i, i, 0};
            std::array<long, 3> extent3                     = {1, 1, mpoDim};
            Ledge3.slice(offset3, extent3).reshape(extent1) = Ledge;
        }
    }
    {
        auto mpsdims = state.mps_sites.back()->dimensions();
        long mpsDim  = mpsdims[2];
        long mpoDim  = mpodims[1];
        Redge3.resize(tenx::array3{mpsDim, mpsDim, mpoDim});
        Redge3.setZero();
        for(long i = 0; i < mpsDim; i++) {
            std::array<long, 1> extent1                     = {mpoDim};
            std::array<long, 3> offset3                     = {i, i, 0};
            std::array<long, 3> extent3                     = {1, 1, mpoDim};
            Redge3.slice(offset3, extent3).reshape(extent1) = Redge;
        }
    }

    // Generate an identity mpo with the same dimensions as the ones in obs
    Eigen::Tensor<cx64, 1> ones(mpodims[0] * mpodims[2]);
    ones.setConstant(cx64(1.0, 0.0));
    Eigen::Tensor<cx64, 4> mpoI = tenx::asDiagonal(ones).reshape(mpodims);

    // Start applying the mpo or identity on each site starting from Ledge3
    Eigen::Tensor<cx64, 3> temp;
    for(const auto &mps : state.mps_sites) {
        const auto  pos   = mps->get_position<long>();
        const auto &M     = mps->get_M();
        const auto  ob_it = std::find_if(mpos.begin(), mpos.end(), [&pos](const auto &ob) { return ob.pos == pos and not ob.used; });
        const auto &mpo   = ob_it != mpos.end() ? ob_it->mpo : mpoI; // Choose the operator or an identity
        if(ob_it != mpos.end()) ob_it->used = true;
        temp.resize(M.dimension(2), M.dimension(2), mpo.dimension(1));
        temp   = M.contract(Ledge3, tenx::idx({0}, {1})).contract(mpo, tenx::idx({0}, {1})).contract(M.conjugate(), tenx::idx({0, 1, 3}, {0, 2, 3}));
        Ledge3 = std::move(temp);
    }

    if(Ledge3.dimensions() != Redge3.dimensions())
        throw except::runtime_error("expectation_value: Ledge3 and Redge3 dimension mismatch: {} != {}", Ledge3.dimensions(), Redge3.dimensions());

    // Finish by contracting Redge3
    Eigen::Tensor<cx64, 0> expval = Ledge3.contract(Redge3, tenx::idx({0, 1, 2}, {0, 1, 2}));

    if(std::imag(expval(0)) > 1e-14)
        tools::log->warn("expectation_value: result has imaginary part: {:+8.2e}{:+8.2e} i", std::real(expval(0)), std::imag(expval(0)));
    //    if(std::imag(expval(0)) > 1e-8)
    //        throw except::runtime_error("expectation_value: result has imaginary part: {:+8.2e}{:+8.2e} i", std::real(expval(0)), std::imag(expval(0)));
    return expval(0);
}

cx64 tools::finite::measure::expectation_value(const StateFinite &state1, const StateFinite &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos) {
    /*!
     * Calculates <state1 | mpos | state2>
     */
    auto t_expval = tid::tic_scope("expval", tid::level::highest);
    if(!num::all_equal(state1.get_length(), state2.get_length(), mpos.size()))
        throw except::logic_error("Sizes are not equal: state1:{} state2:{} mpos:{}", state1.get_length(), state2.get_length(), mpos.size());
    auto L = mpos.size();
    if(state1.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state1 left bond dimension != 1: got {}", state1.get_mps_site(0).get_chiL());
    if(state2.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state2 left bond dimension != 1: got {}", state2.get_mps_site(0).get_chiL());
    if(mpos.front().dimension(0) != 1) throw except::logic_error("mpos left bond dimension != 1: got {}", mpos.front().dimension(0));
    if(state1.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state1 right bond dimension != 1: got {}", state1.get_mps_site(L - 1).get_chiR());
    if(state2.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state2 right bond dimension != 1: got {}", state2.get_mps_site(L - 1).get_chiR());
    if(mpos.back().dimension(1) != 1) throw except::logic_error("mpos right bond dimension != 1: got {}", mpos.back().dimension(1));
    Eigen::Tensor<cx64, 4> result, tmp;
    auto                  &threads = tenx::threads::get();
#pragma message "tools::finite::measure::expectation_value(state1,state2,mpo): add fp64 mode"

    for(size_t pos = 0; pos < L; ++pos) {
        Eigen::Tensor<cx64, 3> mps1 = state1.get_mps_site(pos).get_M().conjugate();
        const auto            &mps2 = state2.get_mps_site(pos).get_M();
        const auto            &mpo  = mpos[pos];
        if(pos == 0) {
            auto dim4 = tenx::array4{mpo.dimension(0) * mps1.dimension(1) * mps2.dimension(1), mps1.dimension(2), mpo.dimension(1), mps2.dimension(2)};
            auto shf6 = tenx::array6{0, 2, 4, 1, 3, 5};
            result.resize(dim4);
            result.device(*threads->dev) = mps1.contract(mpo, tenx::idx({0}, {2})).contract(mps2, tenx::idx({4}, {0})).shuffle(shf6).reshape(dim4);
            continue;
        }
        auto dim4 = tenx::array4{result.dimension(0), mps1.dimension(2), mpo.dimension(1), mps2.dimension(2)};
        tmp.resize(dim4);
        tmp.device(*threads->dev) =
            result.contract(mps1, tenx::idx({1}, {1})).contract(mpo, tenx::idx({1, 3}, {0, 2})).contract(mps2, tenx::idx({1, 4}, {1, 0}));
        result = std::move(tmp);
    }
    // In the end we should have a tensor of size 1 (if the state and mpo edges have dim 1).
    // We can extract and return this value
    if(result.size() != 1) tools::log->warn("expectation_value: result does not have size 1!");
    return result.coeff(0);
}

cx128 tools::finite::measure::expectation_value(const StateFinite &state1, const StateFinite &state2, const std::vector<Eigen::Tensor<cx128, 4>> &mpos_t) {
    /*!
     * Calculates <state1 | mpos | state2>
     */
    auto t_expval = tid::tic_scope("expval_t", tid::level::highest);
    if(!num::all_equal(state1.get_length(), state2.get_length(), mpos_t.size()))
        throw except::logic_error("Sizes are not equal: state1:{} state2:{} mpos_t:{}", state1.get_length(), state2.get_length(), mpos_t.size());
    auto L = mpos_t.size();
    if(state1.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state1 left bond dimension != 1: got {}", state1.get_mps_site(0).get_chiL());
    if(state2.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state2 left bond dimension != 1: got {}", state2.get_mps_site(0).get_chiL());
    if(mpos_t.front().dimension(0) != 1) throw except::logic_error("mpos_t left bond dimension != 1: got {}", mpos_t.front().dimension(0));
    if(state1.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state1 right bond dimension != 1: got {}", state1.get_mps_site(L - 1).get_chiR());
    if(state2.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state2 right bond dimension != 1: got {}", state2.get_mps_site(L - 1).get_chiR());
    if(mpos_t.back().dimension(1) != 1) throw except::logic_error("mpos_t right bond dimension != 1: got {}", mpos_t.back().dimension(1));
    Eigen::Tensor<cx128, 4> result, tmp;
    auto                    &threads = tenx::threads::get();
    for(size_t pos = 0; pos < L; ++pos) {
        const Eigen::Tensor<cx128, 3> mps1  = state1.get_mps_site(pos).get_M().conjugate().cast<cx128>();
        const Eigen::Tensor<cx128, 3> mps2  = state2.get_mps_site(pos).get_M().cast<cx128>();
        const auto                    &mpo_t = mpos_t[pos];
        if(pos == 0) {
            auto dim4 = tenx::array4{mpo_t.dimension(0) * mps1.dimension(1) * mps2.dimension(1), mps1.dimension(2), mpo_t.dimension(1), mps2.dimension(2)};
            auto shf6 = tenx::array6{0, 2, 4, 1, 3, 5};
            result.resize(dim4);
            result.device(*threads->dev) = mps1.contract(mpo_t, tenx::idx({0}, {2})).contract(mps2, tenx::idx({4}, {0})).shuffle(shf6).reshape(dim4);
            continue;
        }
        auto dim4 = tenx::array4{result.dimension(0), mps1.dimension(2), mpo_t.dimension(1), mps2.dimension(2)};
        tmp.resize(dim4);
        tmp.device(*threads->dev) =
            result.contract(mps1, tenx::idx({1}, {1})).contract(mpo_t, tenx::idx({1, 3}, {0, 2})).contract(mps2, tenx::idx({1, 4}, {1, 0}));
        result = std::move(tmp);
    }
    // In the end we should have a tensor of size 1 (if the state and mpo edges have dim 1).
    // We can extract and return this value
    if(result.size() != 1) tools::log->warn("expectation_value: result does not have size 1!");
    return result.coeff(0);
}

cx64 tools::finite::measure::expectation_value(const StateFinite &state1, const StateFinite &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos,
                                               const Eigen::Tensor<cx64, 1> &ledge, const Eigen::Tensor<cx64, 1> &redge) {
    /*!
     * Calculates <state1 | mpos | state2>
     */
    auto mpos_w_edge = mpo::get_mpos_with_edges(mpos, ledge, redge);
    return expectation_value(state1, state2, mpos_w_edge);
}

Eigen::Tensor<cx64, 1> tools::finite::measure::expectation_values(const StateFinite &state, const Eigen::Tensor<cx64, 2> &op) {
    tools::log->trace("Measuring local expectation values");
    long                   len = state.get_length<long>();
    Eigen::Tensor<cx64, 1> expvals(len);
    expvals.setZero();
    for(long pos = 0; pos < len; pos++) {
        LocalObservableOp ob1 = {op, pos};
        expvals(pos)          = expectation_value(state, {ob1});
    }
    return expvals;
}
Eigen::Tensor<cx64, 1> tools::finite::measure::expectation_values(const StateFinite &state, const Eigen::Matrix2cd &op) {
    Eigen::Tensor<cx64, 2> tensor_op = tenx::TensorMap(op);
    return expectation_values(state, tensor_op);
}

template<typename EnvType>
fp64 tools::finite::measure::expectation_value(const Eigen::Tensor<fp64, 3> &mpsBra, const Eigen::Tensor<fp64, 3> &mpsKet,
                                               const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvType> &envs) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet> by applying the mpos in series without splitting the mps first
     */
    auto t_expval = tid::tic_scope("expval", tid::level::highest);

    // Extract the correct tensors depending on EnvType
    Eigen::Tensor<fp64, 3>              envL = envs.L.get_block().real();
    Eigen::Tensor<fp64, 3>              envR = envs.R.get_block().real();
    std::vector<Eigen::Tensor<fp64, 4>> mpos_shf;

    assert(tenx::isReal(mpsBra));
    assert(tenx::isReal(mpsKet));
    assert(tenx::isReal(envL));
    assert(tenx::isReal(envR));

    for(size_t pos = 0; pos < mpos.size(); ++pos) {
        if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvEne>)
            mpos_shf.emplace_back(mpos[pos].get().MPO().real().shuffle(tenx::array4{2, 3, 0, 1}));
        else if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvVar>)
            mpos_shf.emplace_back(mpos[pos].get().MPO2().real().shuffle(tenx::array4{2, 3, 0, 1}));
        else
            static_assert(h5pp::type::sfinae::invalid_type_v<EnvType>);
        assert(tenx::isReal(mpos_shf.back()));
    }

    Eigen::Tensor<fp64, 3> mpoMpsKet = tools::common::contraction::matrix_vector_product(mpsKet, mpos_shf, envL, envR);
    return tools::common::contraction::contract_mps_overlap(mpsBra, mpoMpsKet);
}
template fp64 tools::finite::measure::expectation_value(const Eigen::Tensor<fp64, 3> &mpsBra, const Eigen::Tensor<fp64, 3> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvEne> &envs);
template fp64 tools::finite::measure::expectation_value(const Eigen::Tensor<fp64, 3> &mpsBra, const Eigen::Tensor<fp64, 3> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvVar> &envs);

template<typename EnvType>
cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet,
                                               const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvType> &envs) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet> by applying the mpos in series without splitting the mps first
     */
    bool mpsIsReal = tenx::isReal(mpsBra) and tenx::isReal(mpsKet);
    bool envIsReal = tenx::isReal(envs.L.get_block()) and tenx::isReal(envs.R.get_block());
    bool mpoIsReal = std::all_of(mpos.begin(), mpos.end(), [](const auto &mpo) -> bool { return mpo.get().is_real(); });
    if(mpsIsReal and envIsReal and mpoIsReal) {
        Eigen::Tensor<fp64, 3> mpsBraReal = mpsBra.real();
        Eigen::Tensor<fp64, 3> mpsKetReal = mpsKet.real();
        return expectation_value(mpsBraReal, mpsKetReal, mpos, envs);
    }
    auto t_expval = tid::tic_scope("expval", tid::level::highest);

    // Extract the correct tensors depending on EnvType
    const auto                         &envL = envs.L.get_block();
    const auto                         &envR = envs.R.get_block();
    std::vector<Eigen::Tensor<cx64, 4>> mpos_shf;
    for(size_t pos = 0; pos < mpos.size(); ++pos) {
        if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvEne>)
            mpos_shf.emplace_back(mpos[pos].get().MPO().shuffle(tenx::array4{2, 3, 0, 1}));
        else if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvVar>)
            mpos_shf.emplace_back(mpos[pos].get().MPO2().shuffle(tenx::array4{2, 3, 0, 1}));
        else
            static_assert(h5pp::type::sfinae::invalid_type_v<EnvType>);
    }
    Eigen::Tensor<cx64, 3> mpoMpsKet = tools::common::contraction::matrix_vector_product(mpsKet, mpos_shf, envL, envR);
    return tools::common::contraction::contract_mps_overlap(mpsBra, mpoMpsKet);
}
template cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvEne> &envs);
template cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvVar> &envs);

template<typename EnvType>
cx64 tools::finite::measure::expectation_value(const std::vector<std::reference_wrapper<const MpsSite>> &mpsBra,
                                               const std::vector<std::reference_wrapper<const MpsSite>> &mpsKet,
                                               const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvType> &envs) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet>
     */
    auto t_expval = tid::tic_scope("expval", tid::level::highest);

    assert(num::all_equal(mpsBra.size(), mpsKet.size(), mpos.size()));
    const auto            &envL = envs.L.get_block();
    const auto            &envR = envs.R.get_block();
    Eigen::Tensor<cx64, 3> resL = envL.shuffle(tenx::array3{0, 2, 1});
    Eigen::Tensor<cx64, 3> tmp;
    auto                   mporef  = std::optional<std::reference_wrapper<const Eigen::Tensor<cx64, 4>>>(std::nullopt);
    auto                  &threads = tenx::threads::get();
    for(size_t pos = 0; pos < mpos.size(); ++pos) {
        if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvEne>)
            mporef = mpos[pos].get().MPO();
        else if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvVar>)
            mporef = mpos[pos].get().MPO2();
        else
            static_assert(h5pp::type::sfinae::invalid_type_v<EnvType>);
        const auto &mpo = mporef->get();
        const auto &bra = mpsBra[pos].get().get_M();
        const auto &ket = mpsKet[pos].get().get_M();
        assert(resL.dimension(0) == bra.dimension(1));
        assert(resL.dimension(1) == mpo.dimension(0));
        assert(resL.dimension(2) == ket.dimension(1));
        assert(mpo.dimension(2) == ket.dimension(0));
        assert(mpo.dimension(3) == bra.dimension(0));

        auto dim3 = tenx::array3{ket.dimension(2), mpo.dimension(1), bra.dimension(2)};
        tmp.resize(dim3);
        tmp.device(*threads->dev) =
            resL.contract(ket, tenx::idx({0}, {1})).contract(mpo, tenx::idx({0, 2}, {0, 2})).contract(bra.conjugate(), tenx::idx({0, 3}, {1, 0}));
        resL = std::move(tmp);
    }

    Eigen::Tensor<cx64, 0> res = resL.contract(envR, tenx::idx({0, 1, 2}, {0, 2, 1}));
    return res.coeff(0);
}

template cx64 tools::finite::measure::expectation_value(const std::vector<std::reference_wrapper<const MpsSite>> &mpsBra,
                                                        const std::vector<std::reference_wrapper<const MpsSite>> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvEne> &envs);
template cx64 tools::finite::measure::expectation_value(const std::vector<std::reference_wrapper<const MpsSite>> &mpsBra,
                                                        const std::vector<std::reference_wrapper<const MpsSite>> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvVar> &envs);

template<typename EnvType>
cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet,
                                               const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvType> &envs,
                                               std::optional<svd::config> svd_cfg) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet>, where the mps are already multisite.
     * E.g. for 3 sites we have
     *
     *  |----0             chiL---[ket]---chiR            0----|
     *  |                          d^3                         |
     *  |               2           2           2              |
     *  |----2      0---|---1   0---|---1   0---|---1     2----|
     *  |               3           3           3              |
     *  |                          d^3                         |
     *  |----1             chiL---[bra]---chiR            1----|
     *
     * To apply the mpo's one by one efficiently, we need to split the mps using SVD first
     * Here we make the assumption that bra and ket are not necessarily equal
     */
    if(not svd_cfg) return expectation_value(mpsBra, mpsKet, mpos, envs);

    std::vector<long>   spin_dims_bra, spin_dims_ket;
    std::vector<size_t> positions;
    spin_dims_bra.reserve(mpos.size());
    spin_dims_ket.reserve(mpos.size());
    positions.reserve(mpos.size());
    for(const auto &mpo : mpos) {
        spin_dims_bra.emplace_back(mpo.get().get_spin_dimension());
        spin_dims_ket.emplace_back(mpo.get().get_spin_dimension());
        positions.emplace_back(mpo.get().get_position());
    }

    auto mpsBra_split = tools::common::split::split_mps(mpsBra, spin_dims_bra, positions, safe_cast<long>(positions.back()), svd_cfg);
    auto mpsKet_split = tools::common::split::split_mps(mpsKet, spin_dims_ket, positions, safe_cast<long>(positions.back()), svd_cfg);

    // Put them into a vector of reference wrappers for compatibility with the other expectation_value function
    auto mpsBra_refs = std::vector<std::reference_wrapper<const MpsSite>>(mpsBra_split.begin(), mpsBra_split.end());
    auto mpsKet_refs = std::vector<std::reference_wrapper<const MpsSite>>(mpsKet_split.begin(), mpsKet_split.end());

    return expectation_value(mpsBra_refs, mpsKet_refs, mpos, envs);
}

template cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvEne> &envs,
                                                        std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvVar> &envs,
                                                        std::optional<svd::config> svd_cfg);

template<typename EnvType>
cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpos,
                                               const env_pair<EnvType> &envs, std::optional<svd::config> svd_cfg) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet>, where the mps are already multisite.
     * E.g. for 3 sites we have
     *
     *  |----0             chiL---[ket]---chiR            0----|
     *  |                          d^3                         |
     *  |               2           2           2              |
     *  |----2      0---|---1   0---|---1   0---|---1     2----|
     *  |               3           3           3              |
     *  |                          d^3                         |
     *  |----1             chiL---[bra]---chiR            1----|
     *
     * To apply the mpo's one by one efficiently, we need to split the mps using SVD first
     * Here we make the assumption that bra and ket are not necessarily equal
     */
    if(not svd_cfg) return expectation_value(multisite_mps, multisite_mps, mpos, envs);

    std::vector<long>   spin_dims_bra, spin_dims_ket;
    std::vector<size_t> positions;
    spin_dims_bra.reserve(mpos.size());
    spin_dims_ket.reserve(mpos.size());
    positions.reserve(mpos.size());
    for(const auto &mpo : mpos) {
        spin_dims_bra.emplace_back(mpo.get().get_spin_dimension());
        spin_dims_ket.emplace_back(mpo.get().get_spin_dimension());
        positions.emplace_back(mpo.get().get_position());
    }
    if(positions.size() < 2) {
        tools::log->warn("expectation_value: skipped splitting a single-site multisite_mps");
        // No need to split. Also, splitting would require stashing artifacts from SVD for neighboring sites,
        // which we can't really do here.
        auto oneL = Eigen::Tensor<cx64, 1>(multisite_mps.dimension(1));
        auto oneR = Eigen::Tensor<cx64, 1>(multisite_mps.dimension(2));
        auto mps  = MpsSite(multisite_mps, oneL, positions.front(), 0.0, "AC");
        mps.set_LC(oneR);
        // Put them into a vector of reference wrappers for compatibility with the other expectation_value function
        auto mps_refs = std::vector<std::reference_wrapper<const MpsSite>>{mps};
        return expectation_value(mps_refs, mps_refs, mpos, envs);
    } else {
        // We can avoid splitting the mps by applying the mpos directly onto the mps in sequence
        // auto mpo_mps = tools::common::contraction::matrix_vector_product(multisite_mps, mpos, envs);
        // return tools::common::contraction::contract_mps_overlap(multisite_mps, mpo_mps);

        // Set the new center position in the interior of the set of positions, so we don't get stashes that need to be thrown away.
        auto mps_split = tools::common::split::split_mps(multisite_mps, spin_dims_bra, positions, safe_cast<long>(positions.front()), svd_cfg);
        // Put them into a vector of reference wrappers for compatibility with the other expectation_value function
        auto mps_refs = std::vector<std::reference_wrapper<const MpsSite>>(mps_split.begin(), mps_split.end());
        return expectation_value(mps_refs, mps_refs, mpos, envs);
    }
}

template cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3>                             &multisite_mps,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvEne> &envs,
                                                        std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value(const Eigen::Tensor<cx64, 3>                             &multisite_mps,
                                                        const std::vector<std::reference_wrapper<const MpoSite>> &mpos, const env_pair<EnvVar> &envs,
                                                        std::optional<svd::config> svd_cfg);

// cx64 tools::finite::measure::expectation_value(const std::vector<MpsSite> &mpsBra, const std::vector<MpsSite> &mpsKet, const std::vector<MpoSite> &mpos,
//                                                const env_pair<EnvVar> &envs) {
//     /*!
//      * Calculates <mpsBra | mpos | mpsKet>
//      */
//     auto t_expval = tid::tic_scope("expval", tid::level::highest);
//
//     assert(num::all_equal(mpsBra.size(), mpsKet.size(), mpos.size()));
//     assert(mpsBra.front().get_chiL() == envs.L.get_dims()[0]);
//     assert(mpsBra.back().get_chiR() == envs.R.get_dims()[0]);
//     assert(mpsKet.front().get_chiL() == envs.L.get_dims()[1]);
//     assert(mpsKet.back().get_chiR() == envs.R.get_dims()[1]);
//     assert(mpos.front().MPO2().dimension(0) == envs.L.get_dims()[2]);
//     //    if(state1.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state1 left bond dimension != 1: got {}", state1.get_mps_site(0).get_chiL());
//     //    if(state2.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state2 left bond dimension != 1: got {}", state2.get_mps_site(0).get_chiL());
//     //    if(mpos.front().dimension(0) != 1) throw except::logic_error("mpos left bond dimension != 1: got {}", mpos.front().dimension(0));
//     //    if(state1.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state1 right bond dimension != 1: got {}", state1.get_mps_site(L -
//     //    1).get_chiR()); if(state2.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state2 right bond dimension != 1: got {}",
//     //    state2.get_mps_site(L - 1).get_chiR()); if(mpos.back().dimension(1) != 1) throw except::logic_error("mpos right bond dimension != 1: got {}",
//     //    mpos.back().dimension(1));
//     Eigen::Tensor<cx64, 3> resL = envs.L.get_block();
//     Eigen::Tensor<cx64, 3> tmp;
//     auto                  &threads = tenx::threads::get();
//
//     for(size_t pos = 0; pos < mpos.size(); ++pos) {
//         const auto &bra = mpsBra[pos].get_M();
//         const auto &ket = mpsKet[pos].get_M();
//         const auto &mpo = mpos[pos].MPO2();
//
//         auto dim3 = tenx::array3{ket.dimension(2), mpo.dimension(1), bra.dimension(2)};
//         tmp.resize(dim3);
//         tmp.device(*threads->dev) = resL.contract(ket, tenx::idx({0}, {1}))
//                                         .contract(mpo, tenx::idx({1, 2}, {0, 2}))
//                                         .contract(bra.conjugate(), tenx::idx({0, 2}, {1, 0}))
//                                         .shuffle(tenx::array3{0, 2, 1});
//         resL = std::move(tmp);
//     }
//
//     Eigen::Tensor<cx64, 0> res = resL.contract(envs.R.get_block(), tenx::idx({0, 1, 2}, {0, 1, 2}));
//     return res.coeff(0);
// }

cx64 tools::finite::measure::correlation(const StateFinite &state, const Eigen::Tensor<cx64, 2> &op1, const Eigen::Tensor<cx64, 2> &op2, long pos1, long pos2) {
    if(pos1 == pos2) {
        // Stack the operators
        Eigen::Tensor<cx64, 2> op12 = op1.contract(op2, tenx::idx({0}, {1}));
        LocalObservableOp      ob12 = {op12, pos1};
        return expectation_value(state, {ob12});
    } else {
        // No need to stack
        LocalObservableOp ob1 = {op1, pos1};
        LocalObservableOp ob2 = {op2, pos2};
        return expectation_value(state, {ob1, ob2});
    }
}

Eigen::Tensor<cx64, 2> tools::finite::measure::correlation_matrix(const StateFinite &state, const Eigen::Tensor<cx64, 2> &op1,
                                                                  const Eigen::Tensor<cx64, 2> &op2) {
    if constexpr(settings::debug) tools::log->trace("Measuring correlation matrix");

    long                   len = state.get_length<long>();
    bool                   eq  = tenx::MatrixMap(op1) == tenx::MatrixMap(op2);
    Eigen::Tensor<cx64, 2> C(len, len);
    C.setZero();

    for(long pos_j = 0; pos_j < len; pos_j++) {
        for(long pos_i = pos_j; pos_i < len; pos_i++) {
            C(pos_i, pos_j) = correlation(state, op1, op2, pos_i, pos_j);
            if(eq)
                C(pos_j, pos_i) = C(pos_i, pos_j);
            else
                C(pos_j, pos_i) = correlation(state, op1, op2, pos_j, pos_i);
        }
    }
    return C;
}

Eigen::Tensor<cx64, 2> tools::finite::measure::opdm(const StateFinite &state) {
    /* We create a matrix of the form
     *
     *
     * R(i,j) =  | r++ r+- |
     *           | r-+ r-- |
     *
     * where
     *      r++ = r++(i,j) = ⟨sp(i) sz(i)sz(i+1)...sz(j-1) sm(j)⟩
     *      r+- = r+-(i,j) = ⟨sp(i) sz(i)sz(i+1)...sz(j-1) sp(j)⟩
     *      r-+ = r-+(i,j) = ⟨sm(i) sz(i)sz(i+1)...sz(j-1) sm(j)⟩
     *      r-- = r+-(i,j) = ⟨sm(i) sz(i)sz(i+1)...sz(j-1) sp(j)⟩
     *
     * and sp, sm, sz are the plus, minus and z pauli matrices at some site i.
     * Note that each r is an LxL matrix, so R must be a 2Lx2L matrix.
     * Note also that the signs are "crossed" intentionally: r++ has sp and sm, and so on.
     *
     */
    if(state.measurements.opdm) return state.measurements.opdm.value();
    tools::log->trace("Measuring the one-particle density matrix (OPDM)");
    long L   = state.get_length<long>();
    auto R   = Eigen::MatrixXcd(2 * L, 2 * L); // Allocate the full rho matrix "R"
    auto rpp = R.topLeftCorner(L, L);          // One quadrant of R: rho++
    auto rpm = R.topRightCorner(L, L);         // One quadrant of R: rho+-
    auto rmp = R.bottomLeftCorner(L, L);       // One quadrant of R: rho-+
    auto rmm = R.bottomRightCorner(L, L);      // One quadrant of R: rho--

    // Define pauli matrices sigma plus, minus and z.
    auto sp = tenx::TensorCast(qm::spin::half::sp);
    auto sm = tenx::TensorCast(qm::spin::half::sm);
    auto sz = tenx::TensorCast(qm::spin::half::sz);

    // Shorthand types
    using op_t       = LocalObservableOp;
    using opstring_t = std::vector<op_t>;
    for(long pos_i = 0; pos_i < L; ++pos_i) {
        for(long pos_j = pos_i; pos_j < L; ++pos_j) {
            // Create an operator string from pos_i to pos_j, where
            //      pos_i has sp (or sm)
            //      pos_j has sm (or sp)
            // Then insert sz from pos_i (including) to pos_j (excluding).
            auto opp = opstring_t{op_t{sp, pos_i}}; // adds s+_i
            auto opm = opstring_t{op_t{sp, pos_i}}; // adds s+_i
            auto omp = opstring_t{op_t{sm, pos_i}}; // adds s-_i
            auto omm = opstring_t{op_t{sm, pos_i}}; // adds s-_i
            if(pos_i < pos_j) {
                for(auto pos_x : num::range(pos_i, pos_j)) {
                    opp.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                    opm.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                    omp.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                    omm.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                }
            }
            opp.emplace_back(op_t{sm, pos_j}); // adds  s-_j
            opm.emplace_back(op_t{sp, pos_j}); // adds  s+_j
            omp.emplace_back(op_t{sm, pos_j}); // adds  s-_j
            omm.emplace_back(op_t{sp, pos_j}); // adds  s+_j

            // Calculate the expectation value of the operator string
            rpp(pos_i, pos_j) = expectation_value(state, opp);
            rpm(pos_i, pos_j) = expectation_value(state, opm);
            rmp(pos_i, pos_j) = expectation_value(state, omp);
            rmm(pos_i, pos_j) = expectation_value(state, omm);

            // Set the Hermitian conjugates on the opposite side
            if(pos_i != pos_j) {
                rpp(pos_j, pos_i) = std::conj(rpp(pos_i, pos_j));
                rpm(pos_j, pos_i) = std::conj(rmp(pos_i, pos_j)); // Mix pm <-> mp
                rmp(pos_j, pos_i) = std::conj(rpm(pos_i, pos_j)); // Mix mp <-> pm
                rmm(pos_j, pos_i) = std::conj(rmm(pos_i, pos_j));
            }
        }
    }
    //    tools::log->info("rho++: trace {:.16f}\n{}", rpp.trace(), linalg::matrix::to_string(rpp, 8));
    //    tools::log->info("rho+-: trace {:.16f}\n{}", rpm.trace(), linalg::matrix::to_string(rpm, 8));
    //    tools::log->info("rho-+: trace {:.16f}\n{}", rmp.trace(), linalg::matrix::to_string(rmp, 8));
    //    tools::log->info("rho--: trace {:.16f}\n{}", rmm.trace(), linalg::matrix::to_string(rmm, 8));
    // tools::log->debug("R    : trace {:.16f}", R.trace());
    if(not R.isApprox(R.conjugate().transpose())) throw except::logic_error("R is not hermitian");
    if(std::abs(R.trace() - static_cast<double>(L)) > 1e-8) throw std::runtime_error("R.trace() != L");
    state.measurements.opdm = tenx::TensorMap(R);
    return state.measurements.opdm.value();
}

Eigen::Tensor<double, 1> tools::finite::measure::opdm_spectrum(const StateFinite &state) {
    if(not state.measurements.opdm) state.measurements.opdm = opdm(state);
    if(not state.measurements.opdm_spectrum) {
        auto &opdm   = state.measurements.opdm.value();
        auto  solver = eig::solver();
        solver.eig<eig::Form::SYMM>(opdm.data(), opdm.dimension(0), eig::Vecs::OFF);
        state.measurements.opdm_spectrum = tenx::TensorCast(eig::view::get_eigvals<double>(solver.result));
        // tools::log->debug("OPDM spectrum: {::+9.4e}", tenx::span(state.measurements.opdm_spectrum.value()));
    }
    return state.measurements.opdm_spectrum.value();
}

double tools::finite::measure::structure_factor(const StateFinite &state, const Eigen::Tensor<cx64, 2> &correlation_matrix) {
    tools::log->trace("Measuring structure factor");
    if(correlation_matrix.dimension(0) != correlation_matrix.dimension(1))
        throw except::logic_error("Correlation matrix is not square: dims {}", correlation_matrix.dimensions());
    if(correlation_matrix.dimension(0) != state.get_length<long>())
        throw except::logic_error("Expected correlation matrix of size {}. Got {}", state.get_length<long>(), correlation_matrix.dimension(0));
    return tenx::MatrixMap(correlation_matrix).cwiseAbs2().colwise().sum().sum() / state.get_length<double>();
}

std::array<Eigen::Tensor<double, 1>, 3> tools::finite::measure::expectation_values_xyz(const StateFinite &state) {
    if(not state.measurements.expectation_values_sx) state.measurements.expectation_values_sx = measure::expectation_values(state, qm::spin::half::sx).real();
    if(not state.measurements.expectation_values_sy) state.measurements.expectation_values_sy = measure::expectation_values(state, qm::spin::half::sy).real();
    if(not state.measurements.expectation_values_sz) state.measurements.expectation_values_sz = measure::expectation_values(state, qm::spin::half::sz).real();
    return {state.measurements.expectation_values_sx.value(), state.measurements.expectation_values_sy.value(),
            state.measurements.expectation_values_sz.value()};
}

std::array<double, 3> tools::finite::measure::expectation_value_xyz(const StateFinite &state) {
    if constexpr(settings::debug) tools::log->trace("Measuring spin expectation_value_xyz");
    Eigen::Tensor<cx64, 2> sx  = tenx::TensorMap(qm::spin::half::sx);
    Eigen::Tensor<cx64, 2> sy  = tenx::TensorMap(qm::spin::half::sy);
    Eigen::Tensor<cx64, 2> sz  = tenx::TensorMap(qm::spin::half::sz);
    auto                   pos = (state.get_length<long>() - 1) / 2;
    return {measure::expectation_value(state, {LocalObservableOp{sx, pos}}).real(), measure::expectation_value(state, {LocalObservableOp{sy, pos}}).real(),
            measure::expectation_value(state, {LocalObservableOp{sz, pos}}).real()};
}

std::array<Eigen::Tensor<double, 2>, 3> tools::finite::measure::correlation_matrix_xyz(const StateFinite &state) {
    if constexpr(settings::debug) tools::log->trace("Measuring spin correlation_matrix_xyz");
    Eigen::Tensor<cx64, 2> sx = tenx::TensorMap(qm::spin::half::sx);
    Eigen::Tensor<cx64, 2> sy = tenx::TensorMap(qm::spin::half::sy);
    Eigen::Tensor<cx64, 2> sz = tenx::TensorMap(qm::spin::half::sz);
    if(not state.measurements.correlation_matrix_sx) state.measurements.correlation_matrix_sx = measure::correlation_matrix(state, sx, sx).real();
    if(not state.measurements.correlation_matrix_sy) state.measurements.correlation_matrix_sy = measure::correlation_matrix(state, sy, sy).real();
    if(not state.measurements.correlation_matrix_sz) state.measurements.correlation_matrix_sz = measure::correlation_matrix(state, sz, sz).real();
    return {state.measurements.correlation_matrix_sx.value(), state.measurements.correlation_matrix_sy.value(),
            state.measurements.correlation_matrix_sz.value()};
}

std::array<double, 3> tools::finite::measure::structure_factor_xyz(const StateFinite &state) {
    Eigen::Tensor<cx64, 2> sx = tenx::TensorMap(qm::spin::half::sx);
    Eigen::Tensor<cx64, 2> sy = tenx::TensorMap(qm::spin::half::sy);
    Eigen::Tensor<cx64, 2> sz = tenx::TensorMap(qm::spin::half::sz);
    if(not state.measurements.structure_factor_x) {
        auto correlation_matrix_sx = measure::correlation_matrix(state, sx, sx);
        if(not state.measurements.correlation_matrix_sx) state.measurements.correlation_matrix_sx = correlation_matrix_sx.real();
        state.measurements.structure_factor_x = measure::structure_factor(state, correlation_matrix_sx);
    }
    if(not state.measurements.structure_factor_y) {
        auto correlation_matrix_sy = measure::correlation_matrix(state, sy, sy);
        if(not state.measurements.correlation_matrix_sy) state.measurements.correlation_matrix_sy = correlation_matrix_sy.real();
        state.measurements.structure_factor_y = measure::structure_factor(state, correlation_matrix_sy);
    }
    if(not state.measurements.structure_factor_z) {
        auto correlation_matrix_sz = measure::correlation_matrix(state, sz, sz);
        if(not state.measurements.correlation_matrix_sz) state.measurements.correlation_matrix_sz = correlation_matrix_sz.real();
        state.measurements.structure_factor_z = measure::structure_factor(state, correlation_matrix_sz);
    }

    return {state.measurements.structure_factor_x.value(), state.measurements.structure_factor_y.value(), state.measurements.structure_factor_z.value()};
}
