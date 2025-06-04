#include "storage_info.h"
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include <fmt/core.h>
#include <string>
void StorageInfo::assert_well_defined() const {
    std::string msg;
    if(iter == -1ul) msg.append(" iter\n");
    if(step == -1ul) msg.append(" step\n");
    if(position == -1) msg.append(" position\n");
    if(direction == 0) msg.append(" direction\n");
    if(bond_lim == -1) msg.append(" bond_lim\n");
    if(bond_max == -1) msg.append(" bond_max\n");
    if(trnc_lim == -1) msg.append(" trnc_lim\n");
    if(algo_type == AlgorithmType::ANY) msg.append(" algo_type\n");
    if(algo_name.empty()) msg.append(" algo_name\n");
    if(state_name.empty()) msg.append(" state_name\n");
    if(not msg.empty()) throw except::logic_error("StorageInfo is not well defined. Missing fields: \n{}", msg);
}

StoragePolicy StorageInfo::get_table_storage_policy(std::string_view table_path) const {
    /* clang-format off */
    using namespace settings::storage::table;
    if(table_path.find("bonds")                       != std::string_view::npos) return bonds::policy;
    if(table_path.find("model")                       != std::string_view::npos) return model::policy;
    if(table_path.find("measurements")                != std::string_view::npos) return measurements::policy;
    if(table_path.find("status")                      != std::string_view::npos) return status::policy;
    if(table_path.find("memory")                      != std::string_view::npos) return memory::policy;
    if(table_path.find("timers")                      != std::string_view::npos) return timers::policy;
    if(table_path.find("entanglement_entropies")      != std::string_view::npos) return entanglement_entropies::policy;
    if(table_path.find("truncation_errors")           != std::string_view::npos) return truncation_errors::policy;
    if(table_path.find("bond_dimensions")             != std::string_view::npos) return bond_dimensions::policy;
    if(table_path.find("number_entropies")            != std::string_view::npos) return number_entropies::policy;
    if(table_path.find("renyi_entropies")             != std::string_view::npos) return renyi_entropies::policy;
    if(table_path.find("opdm_spectrum")               != std::string_view::npos) return opdm_spectrum::policy;
    if(table_path.find("expectation_values_s")        != std::string_view::npos) return expectation_values_spin_xyz::policy;
    if(table_path.find("random_unitary_circuit")      != std::string_view::npos) return random_unitary_circuit::policy;
    if(table_path.find("information_per_scale")       != std::string_view::npos) return information_per_scale::policy;
    if(table_path.find("information_center_of_mass")  != std::string_view::npos) return information_center_of_mass::policy;
    throw except::logic_error("Could not match storage policy for table: [{}]", table_path);
    /* clang-format on */
}

StoragePolicy StorageInfo::get_dataset_storage_policy(std::string_view dset_path) const {
    /* clang-format off */
    using namespace settings::storage::dataset;
    if(dset_path.find("lbit_analysis")                    != std::string_view::npos) return lbit_analysis::policy;
    if(dset_path.find("subsystem_entanglement_entropies") != std::string_view::npos) return subsystem_entanglement_entropies::policy;
    if(dset_path.find("information_lattice")              != std::string_view::npos) return information_lattice::policy;
    if(dset_path.find("number_probabilities")             != std::string_view::npos) return number_probabilities::policy;
    if(dset_path.find("expectation_values_spin")          != std::string_view::npos) return expectation_values_spin_xyz::policy;
    if(dset_path.find("correlation_matrix_s")             != std::string_view::npos) return correlation_matrix_spin_xyz::policy;
    if(dset_path.find("opdm")                             != std::string_view::npos) return opdm::policy;
    throw except::logic_error("Could not match storage policy for dataset: [{}]", dset_path);
    /* clang-format on */
}

StoragePolicy StorageInfo::get_mpo_storage_policy(std::string_view model_path) const {
    using namespace settings::storage::mpo;
    if(model_path.find("mpo") != std::string_view::npos) return model::policy;
    throw except::logic_error("Could not match storage policy for mpo: [{}]", model_path);
}

StoragePolicy StorageInfo::get_state_storage_policy() const {
    if(state_name == "state_emid") return settings::storage::mps::state_emid::policy;
    if(state_name == "state_emin") return settings::storage::mps::state_emin::policy;
    if(state_name == "state_emax") return settings::storage::mps::state_emax::policy;
    if(state_name == "state_real") return settings::storage::mps::state_real::policy;
    if(state_name == "state_lbit") return settings::storage::mps::state_lbit::policy;
    if(state_name == "state_eff") return settings::storage::mps::state_lbit::policy; // Reuse lbit

    // xDMRG states are numbered like "state_#". We should therefore get the policy simply based on the algorithm
    if(algo_type == AlgorithmType::xDMRG) return settings::storage::mps::state_emid::policy;
    throw except::logic_error("Failed to get the storage policy for state name: [{}]", state_name);
}

std::string StorageInfo::get_state_prefix() const {
    assert_well_defined();
    if(storage_event == StorageEvent::RBDS_STEP) return fmt::format("{}/{}/rbds", algo_name, state_name);
    if(storage_event == StorageEvent::RTES_STEP) return fmt::format("{}/{}/rtes", algo_name, state_name);
    return fmt::format("{}/{}", algo_name, state_name);
}
std::string StorageInfo::get_mps_prefix() const {
    assert_well_defined();
    std::string tag;
    auto        state_prefix   = get_state_prefix();
    auto        storage_policy = get_state_storage_policy();
    switch(storage_event) {
        case StorageEvent::ITERATION: {
            // TODO check storage level. If replace, just set "iter_last"
            if(has_flag(storage_policy, StoragePolicy::REPLACE) and has_flag(storage_policy, StoragePolicy::FINISH)) {
                return fmt::format("{}/mps", state_prefix, iter); // Just update the one at the final position
            } else if(has_flag(storage_policy, StoragePolicy::REPLACE)) {
                return fmt::format("{}/mps/iter_last", state_prefix, iter);
            } else {
                return fmt::format("{}/mps/iter_{}", state_prefix, iter);
            }
        }
        case StorageEvent::INIT: return fmt::format("{}/mps/init", state_prefix);
        case StorageEvent::FINISHED: return fmt::format("{}/mps", state_prefix);
        case StorageEvent::PROJECTION: return fmt::format("{}/mps/proj", state_prefix);
        case StorageEvent::BOND_UPDATE: return fmt::format("{}/mps/bond_{}", state_prefix, bond_lim);
        case StorageEvent::TRNC_UPDATE: return fmt::format("{}/mps/trnc_{:.2e}", state_prefix, trnc_lim);
        case StorageEvent::RBDS_STEP: return fmt::format("{}/rbds/mps_{}", state_prefix, bond_lim);
        case StorageEvent::RTES_STEP: return fmt::format("{}/rtes/mps_{:.2e}", state_prefix, trnc_lim);
        case StorageEvent::MODEL: throw except::logic_error("get_mps_prefix(): Invalid event: [MODEL]");
        case StorageEvent::NONE: throw except::logic_error("get_mps_prefix(): Invalid event: [NONE]");
        default: throw except::logic_error("Unhandled event: [{}]", enum2sv(storage_event));
    }
}

StorageInfo::StorageInfo(const AlgorithmStatus &status, std::string_view state_name, ModelType model_type)
    : iter(status.iter),                                     //
      step(status.step), position(status.position),          //
      direction(status.direction),                           //
      bond_lim(status.bond_lim),                             //
      bond_max(status.bond_max),                             //
      trnc_lim(status.trnc_lim),                             //
      algo_type(status.algo_type),                           //
      algo_stop(status.algo_stop),                           //
      algo_name(status.algo_type_sv()),                      //
      model_type(model_type),                                //
      storage_event(status.event),                           //
      state_name(state_name),                                //
      algorithm_has_finished(status.algorithm_has_finished), //
      algorithm_has_succeeded(status.algorithm_has_succeeded) {
    assert_well_defined();
}

StorageAttrs::StorageAttrs(const StorageInfo &sinfo) {
    iter          = sinfo.iter;
    step          = sinfo.step;
    bond_lim      = sinfo.bond_lim;
    bond_max      = sinfo.bond_max;
    trnc_lim      = sinfo.trnc_lim;
    storage_event = sinfo.storage_event;
}
bool StorageAttrs::operator==(const StorageAttrs &sinfo) {
    return iter == sinfo.iter and step == sinfo.step and bond_lim == sinfo.bond_lim and bond_max == sinfo.bond_max and trnc_lim == sinfo.trnc_lim and
           storage_event == sinfo.storage_event;
}

bool StorageAttrs::operator==(const StorageInfo &sinfo) {
    return iter == sinfo.iter and step == sinfo.step and bond_lim == sinfo.bond_lim and bond_max == sinfo.bond_max and trnc_lim == sinfo.trnc_lim and
           storage_event == sinfo.storage_event;
}
