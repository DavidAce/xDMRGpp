#include "algorithms/AlgorithmStatus.h"
#include "io/hdf5_types.h"
#include "math/cast.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateInfinite.h"
#include "tensors/TensorsInfinite.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/infinite/h5.h"
#include "tools/infinite/measure.h"
#include <h5pp/h5pp.h>

namespace tools::infinite::h5::save {
    void bootstrap_save_log(std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> &save_log, const h5pp::File &h5file,
                            const std::vector<std::string> &links) {
        if(save_log.empty()) {
            try {
                for(auto &link : links) {
                    if(h5file.linkExists(link)) {
                        auto step      = h5file.readAttribute<uint64_t>(link, "step");
                        auto iter      = h5file.readAttribute<uint64_t>(link, "iter");
                        save_log[link] = std::make_pair(iter, step);
                    }
                }
            } catch(const std::exception &ex) { tools::log->warn("Could not bootstrap save_log: {}", ex.what()); }
        }
    }

    template<typename Scalar>
    void bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<Scalar> &state) {
        // Checks if the current entry has already been saved
        // If it is empty because we are resuming, check if there is a log entry on file already
        auto tic          = tid::tic_token("state");
        auto bonds_prefix = fmt::format("{}/bonds", sinfo.get_state_prefix());

        // Check if the current entry has already been appended
        auto attrs = tools::common::h5::save::get_save_attrs(h5file, bonds_prefix);
        if(attrs == sinfo) return;

        h5file.writeDataset(state.LA(), bonds_prefix + "/L_A");
        h5file.writeDataset(state.LB(), bonds_prefix + "/L_B");
        h5file.writeDataset(state.LC(), bonds_prefix + "/L_C");
        tools::common::h5::save::set_save_attrs(h5file, bonds_prefix, sinfo);
    }
    template void bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<fp32> &state);
    template void bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<fp64> &state);
    template void bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<fp128> &state);
    template void bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<cx32> &state);
    template void bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<cx64> &state);
    template void bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<cx128> &state);

    template<typename Scalar>
    void state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<Scalar> &state) {
        // Checks if the current entry has already been saved
        // If it is empty because we are resuming, check if there is a log entry on file already
        auto tic        = tid::tic_token("state");
        auto mps_prefix = sinfo.get_mps_prefix();
        auto attrs      = tools::common::h5::save::get_save_attrs(h5file, mps_prefix);
        if(attrs == sinfo) return;
        h5file.writeDataset(state.A_bare(), mps_prefix + "/M_A");
        h5file.writeDataset(state.B(), mps_prefix + "/M_B");
        tools::common::h5::save::set_save_attrs(h5file, mps_prefix, sinfo);
    }
    template void state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<fp32> &state);
    template void state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<fp64> &state);
    template void state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<fp128> &state);
    template void state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<cx32> &state);
    template void state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<cx64> &state);
    template void state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<cx128> &state);

    template<typename Scalar>
    void edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<Scalar> &edges) {
        auto        tic = tid::tic_token("edges");
        const auto &ene = edges.get_env_ene_blk();
        const auto &var = edges.get_env_var_blk();
        h5file.writeDataset(ene.L, fmt::format("{}/eneL", sinfo.get_mps_prefix()));
        h5file.writeDataset(ene.R, fmt::format("{}/eneR", sinfo.get_mps_prefix()));
        h5file.writeDataset(var.L, fmt::format("{}/varL", sinfo.get_mps_prefix()));
        h5file.writeDataset(var.R, fmt::format("{}/varR", sinfo.get_mps_prefix()));
    }
    template void edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<fp32> &state);
    template void edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<fp64> &state);
    template void edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<fp128> &state);
    template void edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<cx32> &state);
    template void edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<cx64> &state);
    template void edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<cx128> &state);

    template<typename Scalar>
    /*! Write down the Hamiltonian model type and site info as attributes */
    void model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<Scalar> &model) {
        tools::log->trace("Writing Hamiltonian model");
        auto table_path = fmt::format("{}/model/hamiltonian", sinfo.algo_name);
        if(h5file.linkExists(table_path)) return tools::log->debug("The hamiltonian has already been written to [{}]", table_path);
        tools::log->trace("Storing table: [{}]", table_path);
        auto t_ham = tid::tic_token("save_hamiltonian");
        model.get_mpo_siteA().save_hamiltonian(h5file, table_path);
        model.get_mpo_siteB().save_hamiltonian(h5file, table_path);
        h5file.writeAttribute(enum2sv(settings::model::model_type), "model_type", table_path);
        h5file.writeAttribute(settings::model::model_size, "model_size", table_path);
    }
    template void model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<fp32> &state);
    template void model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<fp64> &state);
    template void model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<fp128> &state);
    template void model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<cx32> &state);
    template void model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<cx64> &state);
    template void model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<cx128> &state);

    template<typename Scalar>
    void mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<Scalar> &model) {
        // We do not expect the MPO's to change. Therefore, if they exist, there is nothing else to do here
        auto model_prefix = fmt::format("{}/model", sinfo.algo_name);
        if(h5file.linkExists(model_prefix)) return tools::log->trace("The model has already been written to [{}]", model_prefix);
        auto tic        = tid::tic_token("mpo");
        auto mpo_prefix = fmt::format("{}/mpo", model_prefix);
        tools::log->trace("Storing mpo [{}]", mpo_prefix);
        model.get_mpo_siteA().save_mpo(h5file, mpo_prefix);
        model.get_mpo_siteB().save_mpo(h5file, mpo_prefix);

        h5file.writeAttribute(2, "model_size", mpo_prefix);
        h5file.writeAttribute(enum2sv(settings::model::model_type), "model_type", mpo_prefix);

        h5file.writeAttribute(2, "model_size", model_prefix);
        h5file.writeAttribute(enum2sv(settings::model::model_type), "model_type", model_prefix);
    }
    template void mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<fp32> &state);
    template void mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<fp64> &state);
    template void mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<fp128> &state);
    template void mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<cx32> &state);
    template void mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<cx64> &state);
    template void mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<cx128> &state);

    template<typename Scalar>
    void measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<Scalar> &tensors, const AlgorithmStatus &status) {
        auto table_path = fmt::format("{}/measurements", sinfo.get_state_prefix());
        // Check if the current entry has already been appended
        static std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> save_log;
        auto                                                                  save_point = std::make_pair(sinfo.iter, sinfo.step);
        if(save_log[table_path] == save_point) return;

        log->trace("Appending to table: {}", table_path);
        if(not h5file.linkExists(table_path)) h5file.createTable(h5pp_table_measurements_infinite<Scalar>::get_h5t(), table_path, "measurements");

        typename h5pp_table_measurements_infinite<Scalar>::table measurement_entry{};
        const auto                                     &state = *tensors.state;

        measurement_entry.step                 = safe_cast<uint64_t>(sinfo.step);
        measurement_entry.iter                 = safe_cast<uint64_t>(sinfo.iter);
        measurement_entry.position             = safe_cast<int64_t>(sinfo.position);
        measurement_entry.length               = safe_cast<uint64_t>(tools::infinite::measure::length(tensors));
        measurement_entry.bond_dim             = tools::infinite::measure::bond_dimension(state);
        measurement_entry.bond_lim             = sinfo.bond_lim;
        measurement_entry.bond_max             = sinfo.bond_max;
        measurement_entry.entanglement_entropy = tools::infinite::measure::entanglement_entropy(state);
        measurement_entry.norm                 = tools::infinite::measure::norm(*tensors.state);
        if(sinfo.algo_type == AlgorithmType::iTEBD) {
            // MPO calculations do not make sense for iTEBD
            measurement_entry.energy_mpo                   = tools::infinite::measure::energy_mpo(tensors);
            measurement_entry.energy_per_site_mpo          = tools::infinite::measure::energy_per_site_mpo(tensors);
            measurement_entry.energy_variance_mpo          = tools::infinite::measure::energy_variance_mpo(tensors);
            measurement_entry.energy_variance_per_site_mpo = tools::infinite::measure::energy_variance_per_site_mpo(tensors);
        }
        measurement_entry.energy_per_site_ham          = tools::infinite::measure::energy_per_site_ham(tensors);
        measurement_entry.energy_per_site_mom          = tools::infinite::measure::energy_per_site_mom(tensors);
        measurement_entry.energy_variance_per_site_ham = tools::infinite::measure::energy_variance_per_site_ham(tensors);
        measurement_entry.energy_variance_per_site_mom = tools::infinite::measure::energy_variance_per_site_mom(tensors);
        measurement_entry.truncation_error             = tools::infinite::measure::truncation_error(state);
        measurement_entry.wall_time                    = status.wall_time;
        measurement_entry.phys_time                    = status.phys_time;
        measurement_entry.time_step                    = status.delta_t;
        auto t_app                                     = tid::tic_scope("append_table");
        h5file.appendTableRecords(measurement_entry, table_path);
        save_log[table_path] = save_point;
    }
    template void measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<fp32> &tensors, const AlgorithmStatus &status);
    template void measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<fp64> &tensors, const AlgorithmStatus &status);
    template void measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<fp128> &tensors, const AlgorithmStatus &status);
    template void measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<cx32> &tensors, const AlgorithmStatus &status);
    template void measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<cx64> &tensors, const AlgorithmStatus &status);
    template void measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<cx128> &tensors, const AlgorithmStatus &status);
}