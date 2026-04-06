#include "algorithms/AlgorithmStatus.h"
#include "config/loader.h"
#include "config/settings.h"
#include "config/threading.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/tenx/threads.h"
#include "mpi/mpi-tools.h"
#include "parse.h"
#include "settings.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/finite/h5.h"
#include "tools/finite/mps.h"
#include <h5pp/h5pp.h>
#include <math/linalg/matrix/to_string.h>
#include <omp.h>

namespace fs = std::filesystem;
using scalar_t = cx64;

std::vector<h5pp::fs::path> find_h5_dirs() {
    std::vector<h5pp::fs::path> result;
    if(mpi::world.id == 0) {
        tools::log->info("Checking for .h5 files in {}", settings::prefix);
        for(const auto &dir : h5pp::fs::recursive_directory_iterator(settings::prefix, h5pp::fs::directory_options::follow_directory_symlink)) {
            if(dir.is_directory()) {
                tools::log->info("Checking for .h5 files in {}", dir.path().string());
                // Check if this directory has any .h5 files
                if(!settings::filter.empty() and dir.path().string().find(settings::filter) == std::string::npos) { continue; }
                for(const auto &obj : h5pp::fs::directory_iterator(dir, h5pp::fs::directory_options::follow_directory_symlink)) {
                    if(obj.is_regular_file() and obj.path().extension() == ".h5") {
                        result.emplace_back(dir.path());
                        break;
                    }
                }
            }
        }
        std::sort(result.begin(), result.end());
        tools::log->debug("Found {} directories with .h5 files. Splitting into {} groups", result.size(), mpi::world.size);
    }
    mpi::scatter_r(result, 0);
    return result;
}

void save_data(h5pp::File &h5file, const StateFinite<scalar_t> &state, const AlgorithmStatus &status) {
    if(not state.position_is_inward_edge()) return;
    auto sinfo = StorageInfo(status, state.get_name(), settings::model::model_type);

    tools::log->debug("Writing to file: Event [{}] | state prefix [{}]", enum2sv(sinfo.storage_event), sinfo.get_state_prefix());

    // The file can be kept open during writes
    h5file.setKeepFileOpened();

    // The main results have now been written. Next we append data to tables
    // tools::finite::h5::save::bond_dimensions(h5file, sinfo, state);
    // tools::finite::h5::save::bonds(h5file, sinfo, state);
    // tools::finite::h5::save::truncation_errors(h5file, sinfo, state);
    // tools::finite::h5::save::entanglement_entropies(h5file, sinfo, state);
    // tools::finite::h5::save::subsystem_entanglement_entropies(h5file, sinfo, state);
    tools::finite::h5::save::information_lattice(h5file, sinfo, state);
    tools::finite::h5::save::information_per_scale(h5file, sinfo, state);
    tools::finite::h5::save::information_center_of_mass(h5file, sinfo, state);
    // tools::finite::h5::save::renyi_entropies(h5file, sinfo, state);
    // tools::finite::h5::save::number_entropies(h5file, sinfo, state);
    // tools::finite::h5::save::number_probabilities(h5file, sinfo, state);
    // tools::finite::h5::save::expectations(h5file, sinfo, state);
    // tools::finite::h5::save::correlations(h5file, sinfo, state);
    // tools::finite::h5::save::opdm(h5file, sinfo, state);
    // tools::finite::h5::save::opdm_spectrum(h5file, sinfo, state);
    // tools::common::h5::save::resume_attrs(h5file, sinfo); // Save attributes relevant for resuming

    // The file can now be closed
    h5file.setKeepFileClosed();
}

int main(int argc, char *argv[]) {
    Loader default_config(settings::config_default);
    settings::parse_fill(argc, argv);
    tools::log = tools::Logger::setLogger("fill", settings::console::loglevel, settings::console::timestamp);
    settings::configure_threads(); // Set up the number of openmp and std threads for Eigen Tensor
    mpi::init();

    auto expected_dsets = std::vector<std::string>{"xDMRG/state_emid/information_center_of_mass", "xDMRG/state_emid/rbds/information_center_of_mass",
                                                   "xDMRG/state_emid/rtes/information_center_of_mass", "common/finished_all"};
    auto h5dirs         = find_h5_dirs();
    using h5iter        = h5pp::fs::directory_iterator;
    for(const auto &h5dir : h5dirs) {
        for(const auto &file : h5iter(h5dir)) {
            if(!file.is_regular_file()) continue;
            if(file.path().extension() != ".h5") continue;
            // if(!settings::filter.empty() and file.path().string().find(settings::filter) == std::string::npos) { continue; }
            auto outfile = fs::path(fmt::format("{}/{}", settings::outdir, file.path().string().substr(settings::prefix.size()))).lexically_normal();
            try {
                tools::log->set_level(spdlog::level::info);
                tools::log->info("Filling: {}", outfile.string());
                tools::log->set_level(spdlog::level::warn);
                fs::create_directories(outfile.parent_path());
                fs::copy(file.path(), outfile, fs::copy_options::update_existing);
                auto h5tgt       = h5pp::File(outfile, h5pp::FileAccess::READWRITE);
                auto found_dsets = std::vector<std::string>();
                for(const auto &dset : expected_dsets) {
                    if(h5tgt.linkExists(dset)) found_dsets.emplace_back(dset);
                }
                if(found_dsets.size() == expected_dsets.size()) { continue; }
                if(h5tgt.readAttribute<uint8_t>("xDMRG/state_emid", "algorithm_has_finished") != 1) { continue; }
                auto config_filename = fmt::format("{}/{}", settings::prefix, h5tgt.readDataset<std::string>("common/config_filename"));
                if(config_filename != settings::input::config_filename) {
                    //  Try loading the given config file.
                    //  Note that there is a default "input/input.config" if none was given
                    Loader h5cfg(config_filename);
                    if(h5cfg.file_exists) {
                        h5cfg.load();
                        settings::load(h5cfg);
                    } else
                        throw except::runtime_error("Could not find config file: {}", config_filename); // Invalid file
                    tools::log->debug("Loaded config file: {}", config_filename);
                }
                settings::storage::output_filepath = outfile.string(); // Replace the output filepath with the new target path

                // Load the simulation from the file
                auto states_that_may_resume =
                    tools::common::h5::resume::find_states_that_may_resume(h5tgt, ResumePolicy::ALWAYS, AlgorithmType::xDMRG, "state_emid");
                if(states_that_may_resume.empty()) throw except::state_error("no resumable states were found");
                for(const auto &[state_prefix, algo_stop] : states_that_may_resume) {
                    tools::log->debug("Resuming [{}] | previous stop reason: {} | resume policy: {} ", state_prefix, enum2sv(algo_stop),
                                      enum2sv(settings::storage::resume_policy));
                    auto model_size = settings::model::model_size;
                    auto state      = StateFinite<scalar_t>(AlgorithmType::xDMRG, model_size, 0);
                    auto status     = AlgorithmStatus();

                    try {
                        // tools::finite::h5::load::simulation(h5tgt, state_prefix, tensors, status, AlgorithmType::xDMRG);
                        tools::finite::h5::load::MpsInfo info;
                        tools::finite::h5::load::state(h5tgt, state_prefix, state, info);
                        tools::common::h5::load::status(h5tgt, state_prefix, status, info);
                    } catch(const except::load_error &le) {
                        tools::log->error("{}", le.what());
                        continue;
                    }

                    // Our first task is to decide on a state name for the newly loaded state
                    // The simplest is to infer it from the state prefix itself
                    auto name = tools::common::h5::resume::extract_state_name(state_prefix);
                    state.set_name(name);

                    // Reload the bond and truncation error limits (could be different in the config compared to the status we just loaded)
                    status.trnc_min                   = settings::precision::svd_truncation_min;
                    status.trnc_max                   = settings::precision::svd_truncation_max;
                    status.bond_min                   = settings::get_bond_min(status.algo_type);
                    status.bond_max                   = settings::get_bond_max(status.algo_type);
                    status.trnc_limit_has_reached_min = false;
                    status.bond_limit_has_reached_max = false;

                    // Apply shifts and compress the model
                    tools::finite::mps::move_center_point_to_inward_edge(state);

                    // Load the subsystem entanglement entropy instead of calculating it (expensive!)
                    auto L    = state.get_length<long>();
                    auto sees = h5tgt.readDataset<std::optional<Eigen::Tensor<double, 3>>>("xDMRG/state_emid/subsystem_entanglement_entropies");
                    if(sees.has_value()) state.measurements.subsystem_entanglement_entropies = Eigen::Map<Eigen::ArrayXXd>(sees->data(), L, L);

                    status.event = StorageEvent::FINISHED;
                    save_data(h5tgt, state, status);

                    auto status_rbds = h5tgt.readTableRecords<std::vector<AlgorithmStatus>>("xDMRG/state_emid/rbds/status", h5pp::TableSelection::ALL);
                    sees             = h5tgt.readDataset<std::optional<Eigen::Tensor<double, 3>>>("xDMRG/state_emid/rbds/subsystem_entanglement_entropies");
                    for(const auto &[sidx, srec] : iter::enumerate(status_rbds)) {
                        state.clear_measurements();
                        state.clear_cache();
                        if(sees.has_value() and sees->dimension(2) > static_cast<long>(sidx))
                            state.measurements.subsystem_entanglement_entropies = Eigen::Map<Eigen::ArrayXXd>(sees->data() + sidx * L * L, L, L);
                        save_data(h5tgt, state, srec);
                    }

                    auto status_rtes = h5tgt.readTableRecords<std::vector<AlgorithmStatus>>("xDMRG/state_emid/rtes/status", h5pp::TableSelection::ALL);
                    sees             = h5tgt.readDataset<std::optional<Eigen::Tensor<double, 3>>>("xDMRG/state_emid/rtes/subsystem_entanglement_entropies");
                    for(const auto &[sidx, srec] : iter::enumerate(status_rtes)) {
                        state.clear_measurements();
                        state.clear_cache();
                        if(sees.has_value() and sees->dimension(2) > static_cast<long>(sidx))
                            state.measurements.subsystem_entanglement_entropies = Eigen::Map<Eigen::ArrayXXd>(sees->data() + sidx * L * L, L, L);
                        save_data(h5tgt, state, srec);
                    }
                }
            } catch(std::exception &e) { tools::log->info("Failed to fill file {}: {}", file.path().string(), e.what()); }
        }
    }
    mpi::barrier();
    mpi::finalize();
}
