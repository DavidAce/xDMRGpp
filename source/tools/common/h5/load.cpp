#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "io/hdf5_types.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include <h5pp/h5pp.h>


namespace tools::common::h5 {
    AlgorithmStatus getStatusRecord(const h5pp::File &h5file, std::string_view status_path, hsize_t offset) {
        // Read elements individually if they exist.
        // This allows us to add new fields to status and retain compatibility with older hdf5 files.
        AlgorithmStatus status;
        /* clang-format off */
        status.iter                         = h5file.readTableField<std::optional<size_t>>               (status_path, "iter",                       offset).value_or(0);
        status.step                         = h5file.readTableField<std::optional<size_t>>               (status_path, "step",                       offset).value_or(0);
        status.position                     = h5file.readTableField<std::optional<long>>                 (status_path, "position",                   offset).value_or(0);
        status.direction                    = h5file.readTableField<std::optional<int>>                  (status_path, "direction",                  offset).value_or(1);
        status.event                        = h5file.readTableField<std::optional<StorageEvent>>         (status_path, "event",                      offset).value_or(StorageEvent::NONE);
        status.opt_ritz                     = h5file.readTableField<std::optional<OptRitz>>              (status_path, "opt_ritz",                   offset).value_or(OptRitz::NONE);
        status.algo_type                    = h5file.readTableField<std::optional<AlgorithmType>>        (status_path, "algo_type",                  offset).value_or(AlgorithmType::ANY);
        status.algo_stop                    = h5file.readTableField<std::optional<AlgorithmStop>>        (status_path, "algo_stop",                  offset).value_or(AlgorithmStop::NONE);
        status.min_iters                    = h5file.readTableField<std::optional<size_t>>               (status_path, "min_iters",                  offset).value_or(0);
        status.bond_lim                     = h5file.readTableField<std::optional<long>>                 (status_path, "bond_lim",                   offset).value_or(0);
        status.bond_max                     = h5file.readTableField<std::optional<long>>                 (status_path, "bond_max",                   offset).value_or(0);
        status.bond_min                     = h5file.readTableField<std::optional<long>>                 (status_path, "bond_min",                   offset).value_or(0);
        status.trnc_lim                     = h5file.readTableField<std::optional<double>>               (status_path, "trnc_lim",                   offset).value_or(0);
        status.trnc_min                     = h5file.readTableField<std::optional<double>>               (status_path, "trnc_min",                   offset).value_or(0);
        status.trnc_max                     = h5file.readTableField<std::optional<double>>               (status_path, "trnc_max",                   offset).value_or(0);
        status.energy_min                   = h5file.readTableField<std::optional<double>>               (status_path, "energy_min",                 offset).value_or(std::numeric_limits<double>::quiet_NaN());
        status.energy_max                   = h5file.readTableField<std::optional<double>>               (status_path, "energy_max",                 offset).value_or(std::numeric_limits<double>::quiet_NaN());
        status.energy_tgt                   = h5file.readTableField<std::optional<double>>               (status_path, "energy_tgt",                 offset).value_or(std::numeric_limits<double>::quiet_NaN());
        status.energy_dens                  = h5file.readTableField<std::optional<double>>               (status_path, "energy_dens",                offset).value_or(std::numeric_limits<double>::quiet_NaN());
        status.energy_dens_target           = h5file.readTableField<std::optional<double>>               (status_path, "energy_dens_target",         offset).value_or(std::numeric_limits<double>::quiet_NaN());
        status.energy_variance_lowest       = h5file.readTableField<std::optional<double>>               (status_path, "energy_variance_lowest",     offset).value_or(1.0);
        status.energy_variance_max_digits   = h5file.readTableField<std::optional<size_t>>               (status_path, "energy_variance_max_digits", offset).value_or(0);
        status.energy_variance_prec_limit   = h5file.readTableField<std::optional<double>>               (status_path, "energy_variance_prec_limit", offset).value_or(0);
        status.mixing_factor          = h5file.readTableField<std::optional<double>>               (status_path, "mixing_factor",        offset).value_or(0.0);
        status.phys_time                    = h5file.readTableField<std::optional<h5pp::fstr_t<64>>>     (status_path, "phys_time",                  offset).value_or(h5pp::fstr_t<64>{});
        status.wall_time                    = h5file.readTableField<std::optional<double>>               (status_path, "wall_time",                  offset).value_or(0);
        status.algo_time                    = h5file.readTableField<std::optional<double>>               (status_path, "algo_time",                  offset).value_or(0);
        status.delta_t                      = h5file.readTableField<std::optional<h5pp::fstr_t<128>>>    (status_path, "delta_t",                    offset).value_or(h5pp::fstr_t<128>{});
        status.algorithm_has_finished       = h5file.readTableField<std::optional<bool>>                 (status_path, "algorithm_has_finished",     offset).value_or(false);
        status.algorithm_has_succeeded      = h5file.readTableField<std::optional<bool>>                 (status_path, "algorithm_has_succeeded",    offset).value_or(false);
        status.algorithm_has_to_stop        = h5file.readTableField<std::optional<bool>>                 (status_path, "algorithm_has_to_stop",      offset).value_or(false);
        status.algorithm_has_stuck_for      = h5file.readTableField<std::optional<size_t>>               (status_path, "algorithm_has_stuck_for",    offset).value_or(0);
        status.algorithm_saturated_for      = h5file.readTableField<std::optional<size_t>>               (status_path, "algorithm_saturated_for",    offset).value_or(0);
        status.algorithm_converged_for      = h5file.readTableField<std::optional<size_t>>               (status_path, "algorithm_converged_for",    offset).value_or(0);
        status.entanglement_converged_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "entanglement_converged_for", offset).value_or(0);
        status.entanglement_saturated_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "entanglement_saturated_for", offset).value_or(0);
        status.locinfoscale_saturated_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "locinfoscale_saturated_for", offset).value_or(0);
        status.variance_mpo_converged_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "variance_mpo_converged_for", offset).value_or(0);
        status.variance_mpo_saturated_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "variance_mpo_saturated_for", offset).value_or(0);
        status.variance_ham_converged_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "variance_ham_converged_for", offset).value_or(0);
        status.variance_ham_saturated_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "variance_ham_saturated_for", offset).value_or(0);
        status.variance_mom_converged_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "variance_mom_converged_for", offset).value_or(0);
        status.variance_mom_saturated_for   = h5file.readTableField<std::optional<size_t>>               (status_path, "variance_mom_saturated_for", offset).value_or(0);
        status.bond_limit_has_reached_max   = h5file.readTableField<std::optional<bool>>                 (status_path, "bond_limit_has_reached_max", offset).value_or(false);
        status.trnc_limit_has_reached_min   = h5file.readTableField<std::optional<bool>>                 (status_path, "trnc_limit_has_reached_min", offset).value_or(false);
        status.spin_parity_has_converged    = h5file.readTableField<std::optional<bool>>                 (status_path, "spin_parity_has_converged",  offset).value_or(false);
        status.trnc_error_has_converged     = h5file.readTableField<std::optional<bool>>                 (status_path, "trnc_error_has_converged",   offset).value_or(false);
        status.time_step_has_converged      = h5file.readTableField<std::optional<bool>>                 (status_path, "time_step_has_converged",    offset).value_or(false);
        /* clang-format on*/
        return status;
    }
    struct StatusTriplet {
        size_t iter;
        size_t step;
        StorageEvent event;
    }__attribute__((packed));
    static_assert(sizeof(StatusTriplet) == sizeof(StatusTriplet::iter)+sizeof(StatusTriplet::step)+sizeof(StatusTriplet::event));

    void load::status(const h5pp::File &h5file, std::string_view state_prefix, AlgorithmStatus &status, const MpsInfo &info) {
        auto tic        = tid::tic_scope("status", tid::level::highest);
        auto table_path = fmt::format("{}/status", state_prefix);
        if(h5file.linkExists(table_path)) {
            tools::log->info("Loading status from table: [{}]", table_path);
            if(info.iter != -1ul or info.step != -1ul or info.event != StorageEvent::NONE) {
                // Load the whole table and find the right entry
                auto recs = h5file.readTableField<std::vector<StatusTriplet>>(table_path, {"iter", "step", "event"}, h5pp::TableSelection::ALL);
                for(const auto &[offset, rec] : iter::enumerate_reverse(recs)) {
                    if(info.iter != 1ul and rec.iter != info.iter) continue;
                    if(info.step != 1ul and rec.step != info.step) continue;
                    if(info.event != StorageEvent::NONE and rec.event != info.event) continue;
                    // We have a match!
                    h5file.readTableRecords(status, table_path, offset); // Reads the last entry by default
                    // status = getStatusRecord(h5file, table_path, offset);
                    return;
                }
                throw except::load_error("Could not find status entry matching iter {} | step {} | event {}", info.iter, info.step, enum2sv(info.event));
            } else {
                h5file.readTableRecords(status, table_path, h5pp::TableSelection::LAST); // Reads the last entry by default
            }
        } else {
            throw except::runtime_error("Could not find table [status] for state [{}] in file [{}] at table path [{}]", state_prefix, h5file.getFilePath(),
                                        table_path);
        }
    }

    void load::status_old(const h5pp::File &h5file, std::string_view state_prefix, AlgorithmStatus &status, const MpsInfo &info) {
        auto tic        = tid::tic_scope("status", tid::level::highest);
        auto table_path = fmt::format("{}/status", state_prefix);
        if(h5file.linkExists(table_path)) {
            tools::log->info("Loading status from table: [{}]", table_path);
            if(info.iter != -1ul or info.step != -1ul or info.event != StorageEvent::NONE) {
                // Load the whole table and find the right entry
                auto statusRecords = h5file.readTableRecords<std::vector<AlgorithmStatus>>(table_path, h5pp::TableSelection::ALL);
                for(const auto &statusRecord : iter::reverse(statusRecords)) {
                    if(info.iter != 1ul and statusRecord.iter != info.iter) continue;
                    if(info.step != 1ul and statusRecord.step != info.step) continue;
                    if(info.event != StorageEvent::NONE and statusRecord.event != info.event) continue;
                    // We have a match!
                    status = statusRecord;
                    return;
                }
                throw except::load_error("Could not find status entry matching iter {} | step {} | event {}", info.iter, info.step, enum2sv(info.event));
            } else {
                h5file.readTableRecords(status, table_path, h5pp::TableSelection::LAST); // Reads the last entry by default
            }
        } else {
            throw except::runtime_error("Could not find table [status] for state [{}] in file [{}] at table path [{}]", state_prefix, h5file.getFilePath(),
                                        table_path);
        }
    }

    void load::timer(const h5pp::File &h5file, std::string_view state_prefix, [[maybe_unused]] const AlgorithmStatus &status) {
        auto table_path = fmt::format("{}/timers", state_prefix);
        if(h5file.linkExists(table_path)) {
            tools::log->info("Loading timers from table: [{}]", table_path);
            auto table = h5file.readTableRecords<std::vector<h5pp_ur::item>>(table_path, h5pp::TableSelection::ALL);
            for(const auto &t : table) {
                tools::log->trace("Loading {}", t.name);
                auto &t_ur = tid::get(t.name.c_str());
                t_ur.set_time(t.time);
                t_ur.set_level(static_cast<tid::level>(t.level));
                t_ur.set_count(t.count);
            }
        } else {
            tools::log->info("Could not load table: link does not exist: [{}]", table_path);
        }
    }

    void load::initial_state_attrs(const h5pp::File &h5file, std::string_view state_prefix, std::string &pattern) {
        auto dset_path = fmt::format("{}/initial_pattern", state_prefix); // For the old style initial_pattern
        if(h5file.linkExists(dset_path)) {
            pattern = h5file.readDataset<std::string>(dset_path);
            tools::log->info("Loading initial pattern: [{}]", pattern);
            if(pattern.size() != settings::model::model_size) {
                throw except::runtime_error("Loaded pattern size {} != model size {}", pattern.size(), settings::model::model_size);
            }
        }

        if(h5file.attributeExists(state_prefix, "initial_pattern")) {
            pattern = h5file.readAttribute<std::string>(state_prefix, "initial_pattern");
        } else {
            throw except::load_error("Could not load pattern: dset or attribute [initial_pattern] does not exist in: [{}]", state_prefix);
        }
        if(h5file.attributeExists(state_prefix, "initial_type")) {
            auto initial_type = h5file.readAttribute<std::string>(state_prefix, "initial_type");
            if(initial_type != enum2sv(settings::strategy::initial_type)) {
                throw except::load_error("Mismatching initial_type: file {} != config {}", initial_type, enum2sv(settings::strategy::initial_type));
            }
        }
        if(h5file.attributeExists(state_prefix, "initial_axis")) {
            auto initial_axis = h5file.readAttribute<std::string>(state_prefix, "initial_axis");
            if(initial_axis != settings::strategy::initial_axis) {
                throw except::load_error("Mismatching initial_axis: file {} != config {}", initial_axis, settings::strategy::initial_axis);
            }
        }
        if(h5file.attributeExists(state_prefix, "initial_state")) {
            auto initial_state = h5file.readAttribute<std::string>(state_prefix, "initial_state");
            if(initial_state != enum2sv(settings::strategy::initial_state)) {
                throw except::load_error("Mismatching initial_axis: file {} != config {}", initial_state, enum2sv(settings::strategy::initial_state));
            }
        }
    }
}
