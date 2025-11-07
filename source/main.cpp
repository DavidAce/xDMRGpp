#include "algorithms/AlgorithmLauncher.h"
#include "config/parse.h"
#include "config/settings.h"
#include "config/threading.h"
#include "debug/info.h"
#include "debug/stacktrace.h"
#include "env/environment.h"
#include "io/filesystem.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "tools/common/log.h"
#include <h5pp/h5pp.h>

void clean_up() {
    if(not settings::storage::use_temp_dir) return;
    if(fs::exists(settings::storage::tmp::hdf5_temp_path)) {
        try {
            tools::log->info("Cleaning up temporary file: [{}]", settings::storage::tmp::hdf5_temp_path);
            h5pp::hdf5::moveFile(settings::storage::tmp::hdf5_temp_path, settings::storage::tmp::hdf5_final_path, h5pp::FilePermission::REPLACE);
        } catch(const std::exception &err) { tools::log->info("Cleaning not needed: {}", err.what()); }
    }
    H5garbage_collect();
    H5Eprint(H5E_DEFAULT, stderr);
}

/*!
    \brief  Main function. Sets simulation parameters and excecutes the desired algorithms.
    \return an integer 0 upon exit success
*/
int main(int argc, char *argv[]) {
    settings::parse(argc, argv);

    tools::log = tools::Logger::setLogger("xDMRG++ main", settings::console::loglevel, settings::console::timestamp);

    // Set up the number of openmp and std threads for Eigen Tensor
    settings::configure_threads();

    // Seed with random::device initially (This also takes care of srand used by Eigen)
    // This is to make reproducible simulations
    rnd::seed(settings::input::seed);

    // print environment and git status
    tools::log->info("Hostname        : {}", debug::hostname());
    tools::log->info("Build hostname  : {}", env::build::hostname);
    tools::log->info("Git branch      : {}", env::git::branch);
    tools::log->info("    commit hash : {}", env::git::commit_hash);
    tools::log->info("    revision    : {}", env::git::revision);

    // Register termination codes and what to do in those cases
    debug::register_callbacks();

    // Make sure to move the file back from temp location
    std::atexit(clean_up);
    std::at_quick_exit(clean_up);

    // Initialize the launcher and run the algorithms
    AlgorithmLauncher().run_algorithms();

    return 0;
}
