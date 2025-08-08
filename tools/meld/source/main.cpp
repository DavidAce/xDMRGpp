#include "debug/exceptions.h"
#include "debug/stacktrace.h"
#include "env/environment.h"
#include "general/enums.h"
#include "general/human.h"
#include "general/prof.h"
#include "general/settings.h"
#include "meld-io/find.h"
#include "meld-io/h5db.h"
#include "meld-io/h5dbg.h"
#include "meld-io/h5io.h"
#include "meld-io/hash.h"
#include "meld-io/id.h"
#include "meld-io/logger.h"
#include "meld-io/meta.h"
#include "meld-io/parse.h"
#include "mpi/mpi-tools.h"
#include "tid/tid.h"
#include <CLI/CLI.hpp>
#include <cstdlib>
#include <h5pp/h5pp.h>
#include <string>

void clean_up() {
    if(not tools::h5io::h5_tmp_part_path.empty()) {
        try {
            tools::logger::log->info("Cleaning up temporary file: [{}]", tools::h5io::h5_tmp_part_path);
            h5pp::hdf5::moveFile(tools::h5io::h5_tmp_part_path, tools::h5io::h5_tgt_part_path, h5pp::FilePermission::REPLACE);
            tools::h5io::h5_tmp_part_path.clear();
        } catch(const std::exception &err) { tools::logger::log->info("Cleaning not needed: {}", err.what()); }
    }
    H5garbage_collect();
    H5Eprint(H5E_DEFAULT, stderr);
}

int main(int argc, char *argv[]) {
    // Here we use getopt to parse CLI input
    // Note that CLI input always override config-file values
    // wherever they are found (config file, h5 file)
    tools::logger::log = tools::logger::setLogger("h5mbl", 2);

    h5pp::fs::path              src_base = h5pp::fs::absolute("/mnt/WDB-AN1500/mbl_transition");
    std::vector<h5pp::fs::path> src_sims;
    std::vector<std::string>    src_reqs  = {"measurements"}; // Required links in source file
    std::string                 src_out   = "output";
    std::string                 src_match = "mbl_";
    std::string                 tgt_file  = "merged.h5";
    h5pp::fs::path              tgt_dir;
    h5pp::fs::path              tmp_dir        = h5pp::fs::absolute(fmt::format("/tmp/{}", tools::h5io::get_tmp_dirname(argv[0])));
    bool                        finished       = false;
    bool                        link_only      = false;
    bool                        lbit_only      = false;
    bool                        use_tmp        = false;
    size_t                      verbosity      = 2;
    size_t                      verbosity_h5pp = 2;
    size_t                      time_steps     = -1ul;
    size_t                      max_files      = 0ul;
    size_t                      max_dirs       = 0ul;
    long                        seed_min       = 0l;
    long                        seed_max       = std::numeric_limits<long>::max();
    std::string                 algo           = "";
    Model                       model          = Model::SDUAL;
    std::string                 state          = "state_*";
    bool                        replace        = false;
    unsigned int                compression    = 3;
    std::vector<std::string>    incfilter      = {};
    std::vector<std::string>    excfilter      = {};
    {
        using namespace h5pp;
        using namespace spdlog;
        CLI::App app;
        app.description("meld: Collect simulation data from xDMRG++ batch simulations");
        app.get_formatter()->column_width(80);
        app.option_defaults()->always_capture_default();
        std::map<std::string, Model>                           ModelMap{{"sdual", Model::SDUAL}, {"majorana", Model::MAJORANA}, {"lbit", Model::LBIT}};
        std::vector<std::pair<std::string, LogLevel>>          h5ppLevelMap{{"trace", LogLevel::trace}, {"debug", LogLevel::debug}, {"info", LogLevel::info}};
        std::vector<std::pair<std::string, level::level_enum>> SpdlogLevelMap{{"trace", level::trace}, {"debug", level::debug}, {"info", level::info}};

        /* clang-format off */
        app.add_option("-A,--algo"        , algo            , "Choose the algorithm type")->required()->check(CLI::IsMember({"fDMRG", "xDMRG", "fLBIT"}, CLI::ignore_case))->always_capture_default(false);
        app.add_option("-M,--model"       , model           , "Choose [majorana|sdual|lbit]")->required()->transform(CLI::CheckedTransformer(ModelMap, CLI::ignore_case))->always_capture_default(false);
        app.add_option("-S,--state"       , state           , "Choose the state name or pattern e.g. state_emid or state_*");
        app.add_option("-n,--tgtfile"     , tgt_file        , "The destination file name for the merge-file");
        app.add_option("-t,--tgtdir"      , tgt_dir         , "The destination directory for the merge-file");
        app.add_option("-b,--srcbase"     , src_base        , "The base directory for MBL simulation results");
        app.add_option("-o,--srcout"      , src_out         , "The name of the source directory where simulation files are found");
        app.add_option("-s,--srcsims"     , src_sims        , "Patterns to simulation directories from where to collect simulation files, e.g. 'lbit19-,lbit20-'")->required();
        app.add_option("--srcreqs"        , src_reqs        , "Required dataset names in each source file");
        app.add_flag  ("-f,--finished"    , finished        , "Require that src file has finished");
        app.add_flag  ("-l,--linkonly"    , link_only       , "Link only. Make the main file with external links to all the others");
        app.add_flag  ("--lbitonly"       , lbit_only       , "Collect lbit supports only");
        app.add_flag  ("-r,--replace"     , replace         , "Replace existing files");
        app.add_flag  ("-T,--usetemp"     , use_tmp         , "Use temp directory");
        app.add_option("--time-steps"     , time_steps      , "Expected number of time steps (skip files on mismatch)" );
        app.add_option("--maxfiles"       , max_files       , "Maximum number of .h5 files to collect in each set");
        app.add_option("--maxdirs"        , max_dirs        , "Maximum number of simulation sets");
        app.add_option("--maxseed"        , seed_max        , "Maximum seed number to collect");
        app.add_option("--minseed"        , seed_min        , "Minimum seed number to collect");
        app.add_option("--inc"            , incfilter       , "Include paths to .h5 matching any in this list");
        app.add_option("--exc"            , excfilter       , "Exclude paths to .h5 matching any in this list");
        app.add_option("-v,--log"         , verbosity       , "Log level")->transform(CLI::CheckedTransformer(h5ppLevelMap, CLI::ignore_case));
        app.add_option("-V,--logh5pp"     , verbosity_h5pp  , "Log level of h5pp")->transform(CLI::CheckedTransformer(h5ppLevelMap, CLI::ignore_case));
        app.add_option("-z,--compression" , compression     , "Compression level of h5pp")->check(CLI::Range(0,9));
        CLI11_PARSE(app, argc, argv);
        /* clang-format on */
    }

    for(auto &src_sim : src_sims) {
        if(src_sim.is_relative()) {
            auto matching_dirs = tools::io::find_dir<false>(src_base, src_sim.string(), src_out);
            if(matching_dirs.size() > 5) {
                std::string error_msg = h5pp::format("Too many directories match the pattern {}:\n", (src_base / src_sim).string());
                for(auto &dir : matching_dirs) error_msg.append(dir.string() + '\n');
                throw std::runtime_error(error_msg);
            }
            if(matching_dirs.empty())
                throw std::runtime_error(h5pp::format("No directories match the pattern: {} (subdir: [{}])", (src_base / src_sim).string(), src_out));
            // We have multiple matches. Append them
            for(auto &match : matching_dirs) src_sims.emplace_back(match);
        } else
            src_sim = h5pp::fs::canonical(src_sim);
    }
    // Now remove elements that do not exist
    src_sims.erase(std::remove_if(src_sims.begin(), src_sims.end(),
                                  [](const h5pp::fs::path &p) {
                                      return not h5pp::fs::exists(p); // put your condition here
                                  }),
                   src_sims.end());

    // Now generate the target dir if none was given
    if(tgt_dir.empty() and src_sims.size() == 1) {
        tgt_dir = h5pp::fs::absolute(src_sims.front() / "analysis/data");
    } else if(tgt_dir.is_relative() and src_sims.size() == 1) {
        tgt_dir = h5pp::fs::absolute(src_sims.front() / tgt_dir);
    } else if(src_sims.size() > 1)
        throw std::runtime_error("src_sims > 1 and no tgt_dir was given. Can't deduce where to put the target file.");
    if(tgt_dir.empty()) throw std::runtime_error("A target directory is required. Pass --tgtdir=<dir>");

    tools::logger::setLogLevel(tools::logger::log, verbosity);
    tools::logger::log->info("Started h5mbl from directory {}", h5pp::fs::current_path());
    if(src_sims.empty()) throw std::runtime_error("Source directories are required. Pass --srcsims=<pattern1,pattern2...>");
    for(auto &src_dir : src_sims) {
        if(not h5pp::fs::is_directory(src_dir)) throw std::runtime_error(fmt::format("Given source is not a directory: {}", src_dir.string()));
        tools::logger::log->info("Found source directory {}", src_dir.string());
    }

    auto t_h5mbl = tid::tic_scope("h5mbl");
    mpi::init();
    // Register termination codes and what to do in those cases
    debug::register_callbacks();
    if(mpi::world.id == 0) {
        std::atexit(tools::prof::print_profiling);
        std::at_quick_exit(tools::prof::print_profiling);
        std::atexit(tools::prof::print_mem_usage);
    }

    // Set file permissions
    auto perm = h5pp::FilePermission::READWRITE;
    if(replace) perm = h5pp::FilePermission::REPLACE;

    h5pp::fs::path tgt_path = tgt_dir / tgt_file; // File path for the main merge file
    tools::logger::log->info("Merge into target file {}", tgt_path.string());

    if(use_tmp) {
        std::atexit(clean_up);
        std::at_quick_exit(clean_up);
    }

    if(not link_only) {
        // Define which objects to consider for merging
        tools::h5db::Keys keys;
        switch(model) {
            case Model::SDUAL: {
                keys.models.emplace_back(ModelKey(algo, "model", "hamiltonian"));

                // A table records data from the last time step
                keys.tables.emplace_back(TableKey(algo, state, "", "status"));
                keys.tables.emplace_back(TableKey(algo, state, "", "memory"));
                keys.tables.emplace_back(TableKey(algo, state, "", "measurements"));
                keys.tables.emplace_back(TableKey(algo, state, "", "bond_dims"));
                keys.tables.emplace_back(TableKey(algo, state, "", "bond_dimensions"));
                keys.tables.emplace_back(TableKey(algo, state, "", "entanglement_entropies"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_2"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_3"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_4"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_inf"));
                keys.tables.emplace_back(TableKey(algo, state, "", "truncation_errors"));

                keys.fesups.emplace_back(FesUpKey(algo, state, "", "measurements"));
                keys.fesups.emplace_back(FesUpKey(algo, state, "", "bond_dims"));
                keys.fesups.emplace_back(FesUpKey(algo, state, "", "bond_dimensions"));
                keys.fesups.emplace_back(FesUpKey(algo, state, "", "entanglement_entropies"));
                keys.fesups.emplace_back(FesUpKey(algo, state, "", "truncation_errors"));
                keys.fesups.emplace_back(FesUpKey(algo, state, "", "memory"));
                keys.fesups.emplace_back(FesUpKey(algo, state, "", "status"));

                keys.dsets.emplace_back(DsetKey(algo, state, "", "correlation_matrix_sx", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "correlation_matrix_sy", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "correlation_matrix_sz", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "expectation_values_sx", Size::FIX, 1));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "expectation_values_sy", Size::FIX, 1));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "expectation_values_sz", Size::FIX, 1));
                break;
            }
            case Model::MAJORANA: {
                keys.models.emplace_back(ModelKey(algo, "model", "hamiltonian"));

                // A table records data from the last time step
                keys.tables.emplace_back(TableKey(algo, state, "", "status"));
                keys.tables.emplace_back(TableKey(algo, state, "", "memory"));
                keys.tables.emplace_back(TableKey(algo, state, "", "measurements"));
                keys.tables.emplace_back(TableKey(algo, state, "", "bond_dimensions"));
                keys.tables.emplace_back(TableKey(algo, state, "", "entanglement_entropies"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_2"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_3"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_4"));
                keys.tables.emplace_back(TableKey(algo, state, "", "renyi_entropies_inf"));
                keys.tables.emplace_back(TableKey(algo, state, "", "truncation_errors"));
                keys.tables.emplace_back(TableKey(algo, state, "", "opdm_spectrum"));
                keys.tables.emplace_back(TableKey(algo, state, "", "information_per_scale"));
                keys.tables.emplace_back(TableKey(algo, state, "", "information_center_of_mass"));

                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "measurements"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "bond_dimensions"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "entanglement_entropies"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "truncation_errors"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "memory"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "status"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "opdm_spectrum"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "information_per_scale"));
                keys.rbdses.emplace_back(RbdsKey(algo, state, "rbds", "information_center_of_mass"));

                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "measurements"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "bond_dimensions"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "entanglement_entropies"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "truncation_errors"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "memory"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "status"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "opdm_spectrum"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "information_per_scale"));
                keys.rteses.emplace_back(RtesKey(algo, state, "rtes", "information_center_of_mass"));

                keys.dsets.emplace_back(DsetKey(algo, state, "", "opdm", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "correlation_matrix_sx", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "correlation_matrix_sy", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "correlation_matrix_sz", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "expectation_values_sx", Size::FIX, 1));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "expectation_values_sy", Size::FIX, 1));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "expectation_values_sz", Size::FIX, 1));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "subsystem_entanglement_entropies", Size::FIX, 2));
                keys.dsets.emplace_back(DsetKey(algo, state, "", "information_lattice", Size::FIX, 2));

                //// keys.dsets.emplace_back(DsetKey(algo, state, "rtes", "subsystem_entanglement_entropies", Size::FIX, 2));
                //// keys.dsets.emplace_back(DsetKey(algo, state, "rbds", "subsystem_entanglement_entropies", Size::FIX, 2));

                break;
            }
            case Model::LBIT: {
                if(lbit_only) {
                    keys.models.emplace_back(ModelKey(algo, "model", "hamiltonian"));
                    keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "corrmat", Size::FIX, 0));
                    break;
                }
                keys.models.emplace_back(ModelKey(algo, "model", "hamiltonian"));

                // A table records data from the last time step
                keys.tables.emplace_back(TableKey(algo, state, "", "status"));
                keys.tables.emplace_back(TableKey(algo, state, "", "memory"));
                keys.tables.emplace_back(TableKey(algo, state, "", "bond_dimensions"));
                keys.tables.emplace_back(TableKey(algo, state, "", "truncation_errors"));
                // A crono records data from each time step
                keys.cronos.emplace_back(CronoKey(algo, state, "", "measurements", time_steps));
                //                keys.cronos.emplace_back(CronoKey(algo, state, "bond_dimensions", time_steps));
                keys.cronos.emplace_back(CronoKey(algo, state, "", "entanglement_entropies", time_steps));
                keys.cronos.emplace_back(CronoKey(algo, state, "", "number_entropies", time_steps));
                //                keys.cronos.emplace_back(CronoKey(algo, state, "truncation_errors", time_steps));

                // Last argument is the axis along which to build the time series
                keys.dsets.emplace_back(DsetKey(algo, state, "", "number_probabilities", Size::FIX, 3, SlabSelect::FULL));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "cls_avg_fit", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "cls_avg_rms", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "cls_avg_rsq", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "cls_typ_fit", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "cls_typ_rms", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "cls_typ_rsq", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "corrtyp", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "corravg", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "correrr", Size::FIX, 0));
                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "", "corrmat", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "corroff", Size::FIX, 0));
                keys.dsets.emplace_back(DsetKey(algo, "model", "", "opdm-eigv", Size::FIX, 1));

                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "decay_avg", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "decay_err", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "data", Size::FIX, 0));
                //                keys.dsets.emplace_back(DsetKey(algo, "model/lbits", "data_shifted", Size::FIX, 0));

                //            keys.dsets.emplace_back(DsetKey(algo, state, "finished", "schmidt_midchain", Size::VAR, Type::COMPLEX));
                //            keys.dsets.emplace_back(DsetKey(algo, state, "finished/profiling", "fLBIT.run", Size::FIX, Type::TID));
                break;
            }
            default: throw std::runtime_error("Invalid model");
        }

        auto h5dirs = tools::io::find_h5_dirs(src_sims, src_out, max_dirs, incfilter, excfilter);
        tools::logger::log->info("num h5dirs: {}", h5dirs.size());
        for(const auto &h5dir : h5dirs) tools::logger::log->debug(" -- {}", h5dir.string());

        std::unordered_map<std::string, FileStats> file_stats;
        uintmax_t                                  srcBytes = 0; // Count the total size of scanned files
        for(const auto &h5dir : h5dirs) {
            // Define a new partial hdf5 file to collect the source HDF5 files in this particular h5dir
            tools::h5io::h5_tgt_part_hash = tools::hash::std_hash(h5dir.string());
            tools::h5io::h5_tgt_part_path = h5pp::format("{}/{}.{}{}", tgt_dir.string(), h5pp::fs::path(tgt_path).stem().string(),
                                                         tools::h5io::h5_tgt_part_hash, h5pp::fs::path(tgt_path).extension().string());

            // Initialize the partial target file for this dir
            auto h5_tgt_part = h5pp::File();
            try {
                h5_tgt_part = h5pp::File(tools::h5io::h5_tgt_part_path, perm, verbosity_h5pp);
            } catch(const std::exception &ex) {
                H5Eprint(H5E_DEFAULT, stderr);
                tools::logger::log->error("Error opening target file: {}", ex.what());
                tools::logger::log->error("Replacing broken file: [{}]", tools::h5io::h5_tgt_part_path);
                h5_tgt_part = h5pp::File(tools::h5io::h5_tgt_part_path, h5pp::FilePermission::REPLACE, verbosity_h5pp);
            }
            //            auto h5_tgt = h5pp::File(h5_tgt_path, perm, verbosity_h5pp);
            tools::h5dbg::assert_no_dangling_ids(h5_tgt_part, __FUNCTION__, __LINE__); // Check that there are no open HDF5 handles
            //        hsize_t fsp_size; size_t pbs_size;
            //        H5Pget_file_space_page_size(H5Fget_create_plist(h5_tgt.openFileHandle()), &fsp_size);
            //        H5Pget_page_buffer_size(H5Fget_access_plist(h5_tgt.openFileHandle()),&pbs_size,nullptr,nullptr);
            //        tools::logger::log->info("fsp_size: {} | pbs_size {}", fsp_size, pbs_size);

            //            h5_tgt.setDriver_core();
            //    h5_tgt.setKeepFileOpened();
            h5_tgt_part.setCompressionLevel(compression);
            //        size_t rdcc_nbytes = 4*1024*1024;
            //        size_t rdcc_nslots = rdcc_nbytes * 100 / (320 * 64);
            //        H5Pset_cache(h5_tgt.plists.fileAccess, 1000, rdcc_nslots,rdcc_nbytes, 0.20 );
            //
            //        h5pp::hdf5::setTableSize(tableInfo, 2500);
            //        h5pp::hid::h5p dapl = H5Dget_access_plist(tableInfo.h5Dset.value());
            //        size_t rdcc_nbytes = 4*1024*1024;
            //        size_t rdcc_nslots = rdcc_nbytes * 100 / (tableInfo.recordBytes.value() * tableInfo.chunkSize.value());
            //        H5Pset_chunk_cache(dapl, rdcc_nslots, rdcc_nbytes, 0.5);
            //        tableInfo.h5Dset = h5pp::hdf5::openLink<h5pp::hid::h5d>(tableInfo.getLocId(), tableInfo.tablePath.value(), true, dapl);
            tools::h5dbg::print_dangling_ids(h5_tgt_part, __FUNCTION__, __LINE__);
            tools::h5io::h5_tgt_part_path = h5_tgt_part.getFilePath();
            tools::h5io::h5_tmp_part_path = h5_tgt_part.getFilePath();
            if(use_tmp) {
                tools::h5io::h5_tmp_part_path = (tmp_dir / tgt_file).string();
                tools::logger::log->info("Moving to {} -> {}", h5_tgt_part.getFilePath(), tools::h5io::h5_tmp_part_path);
                h5_tgt_part.moveFileTo(tools::h5io::h5_tmp_part_path, h5pp::FilePermission::REPLACE);
            }

            // Load database
            tools::h5db::TgtDb tgtdb;
            //    h5_tgt.setDriver_core();
            {
                //                auto keepOpen = h5_tgt_part.getFileHandleToken();
                tgtdb.file  = tools::h5db::loadFileDatabase(h5_tgt_part); // This database maps  src_name <--> FileId
                tgtdb.dset  = tools::h5db::loadDatabase<h5pp::DsetInfo>(h5_tgt_part, keys.dsets);
                tgtdb.table = tools::h5db::loadDatabase<h5pp::TableInfo>(h5_tgt_part, keys.tables);
                tgtdb.crono = tools::h5db::loadDatabase<BufferedTableInfo>(h5_tgt_part, keys.cronos);
                tgtdb.fesup = tools::h5db::loadDatabase<BufferedTableInfo>(h5_tgt_part, keys.fesups);
                tgtdb.rbds  = tools::h5db::loadDatabase<BufferedTableInfo>(h5_tgt_part, keys.rbdses);
                tgtdb.rtes  = tools::h5db::loadDatabase<BufferedTableInfo>(h5_tgt_part, keys.rteses);
                tgtdb.model = tools::h5db::loadDatabase<h5pp::TableInfo>(h5_tgt_part, keys.models);
            }
            {
                for(const auto &[infoKey, infoId] : tgtdb.table) infoId.info.assertReadReady();
                for(const auto &[infoKey, infoId] : tgtdb.dset) infoId.info.assertReadReady();
                //                for(const auto &[infoKey, infoId] : tgtdb.model) infoId.info.assertReadReady() ;
                for(const auto &[infoKey, infoId] : tgtdb.crono) infoId.info.assertReadReady();
                for(const auto &[infoKey, infoId] : tgtdb.fesup) infoId.info.assertReadReady();
                for(const auto &[infoKey, infoId] : tgtdb.rbds) infoId.info.assertReadReady();
                for(const auto &[infoKey, infoId] : tgtdb.rtes) infoId.info.assertReadReady();
            }
            uintmax_t tgtBytes = h5pp::fs::file_size(h5_tgt_part.getFilePath());

            // Collect and sort all the files in h5dir
            using h5iter = h5pp::fs::directory_iterator;
            std::vector<h5pp::fs::path> h5files;
            for(const auto &file : h5iter(h5dir)) {
                if(not file.is_regular_file()) continue;
                if(file.path().extension() != ".h5") continue;
                if(file.path().stem().string().find(src_match) == std::string::npos) continue;
                bool include = incfilter.empty() or std::all_of(incfilter.begin(), incfilter.end(),
                                                                [&file](const auto &s) { return file.path().string().find(s) != std::string::npos; });
                bool exclude =
                    std::any_of(excfilter.begin(), excfilter.end(), [&file](const auto &s) { return file.path().string().find(s) != std::string::npos; });
                if(exclude or not include) continue;
                h5files.emplace_back(file);
            }
            std::sort(h5files.begin(), h5files.end());

            tools::logger::log->info("num h5files: {}", h5files.size());
            // No barriers from now on: There can be a different number of files in h5files!
            for(const auto &src_item : h5files) {
                auto        t_src_item = tid::tic_scope("src_item");
                const auto &src_abs    = src_item;
                auto        t_pre      = tid::tic_scope("preamble");

                // Check which source root this belongs to
                // TODO: What does this do? Expand comment
                h5pp::fs::path src_dir;
                for(auto &src_can : src_sims) {
                    auto [it1, it2] = std::mismatch(src_can.begin(), src_can.end(), src_abs.begin());
                    if(it1 == src_can.end()) {
                        src_dir = src_can;
                        break;
                    }
                }
                if(src_dir.empty()) throw std::runtime_error("Could not infer root src_dir from src_abs");

                auto src_rel = h5pp::fs::relative(src_abs, src_dir);
                auto src_par = src_rel.parent_path();

                bool stats_exists = file_stats.find(src_par) != file_stats.end();
                if(not stats_exists) {
                    // Create new entry
                    file_stats[src_par].count = 0;
                    file_stats[src_par].files = h5files.size();
                } else if(max_files > 0 and file_stats[src_par].count >= max_files) {
                    tools::logger::log->debug("Max files reached in {}: {}", src_par, file_stats[src_par].count);
                    break;
                }

                // Append latest profiling information to table
                t_pre.toc();

                // We should now have enough to define a FileId
                auto src_hash = tools::hash::hash_file_meta(src_abs);
                auto src_seed = tools::parse::extract_digits_from_h5_filename<long>(src_rel.filename());
                if(src_seed != std::clamp<long>(src_seed, seed_min, seed_max)) {
                    //                    tools::logger::log->warn("Skipping seed {}: Valid are [{}-{}]", src_seed, seed_min, seed_max);
                    continue;
                }

                FileId fileId(src_seed, src_abs.string(), src_hash);
                // We check if it's in the file database
                auto status             = tools::h5db::getFileIdStatus(tgtdb.file, fileId);
                tgtdb.file[fileId.path] = fileId;

                // Update file stats
                file_stats[src_par].elaps = file_stats[src_par].count == 0 ? t_src_item->restart_lap() : t_src_item->get_lap();
                file_stats[src_par].count++;
                file_stats[src_par].bytes += h5pp::fs::file_size(src_abs.string());

                // Print file status
                srcBytes += h5pp::fs::file_size(src_abs.string());
                tgtBytes = h5pp::fs::file_size(h5_tgt_part.getFilePath());

                auto fmt_grp_bytes = tools::fmtBytes(true, file_stats[src_par].bytes, 1024, 1);
                auto fmt_src_bytes = tools::fmtBytes(true, srcBytes, 1024, 1);
                auto fmt_tgt_bytes = tools::fmtBytes(true, tgtBytes, 1024, 1);
                auto fmt_spd_bytes = fmt::format("{:.1f} files", file_stats[src_par].get_speed());
                if(tools::logger::log->level() <= 1) {
                    tools::logger::log->info(FMT_STRING("Found file: {} | {} | {} | count {} | src {} ({}) | tgt {} | {}/s"), src_rel.string(),
                                             enum2str(status), src_hash, file_stats[src_par].count, fmt_grp_bytes, fmt_src_bytes, fmt_tgt_bytes, fmt_spd_bytes);
                } else {
                    static size_t lastcount = 0;
                    if(file_stats[src_par].count < lastcount) lastcount = 0;
                    size_t filecounter = file_stats[src_par].count - lastcount;
                    if(t_h5mbl->get_lap() > 5.0 or file_stats[src_par].count % 1000 == 0 or file_stats[src_par].count == 1 or
                       file_stats[src_par].count == file_stats[src_par].files) {
                        tools::logger::log->info(FMT_STRING("Directory {} ({}) | count {} | src {} ({}) | tgt {} | {:.2f}/s"), h5dir.string(),
                                                 file_stats[src_par].files, file_stats[src_par].count, fmt_grp_bytes, fmt_src_bytes, fmt_tgt_bytes,
                                                 static_cast<double>(filecounter) / t_h5mbl->restart_lap());
                        lastcount = file_stats[src_par].count;
                    }
                }

                if(status == FileIdStatus::UPTODATE) continue;

                // If we've reached this point we will start reading from h5_src many times.
                auto       t_open = tid::tic_scope("open");
                h5pp::File h5_src;
                try {
                    h5_src = h5pp::File(src_abs.string(), h5pp::FilePermission::READONLY, verbosity_h5pp);
                    // h5_src.setDriver_core(false, 10 * 1024 * 1024);
                    // h5_src.setDriver_sec2();
                    //                 h5_src.setDriver_core();
                    // H5Pset_cache(h5_src.plists.fileAccess, 1000, 7919,rdcc_nbytes, 0.0 );
                    h5_src.setCloseDegree(H5F_close_degree_t::H5F_CLOSE_WEAK); // Delay closing ids related to this file.

                } catch(const std::exception &ex) {
                    tools::logger::log->warn("Skipping broken file: {}\n\tReason: {}\n", src_abs.string(), ex.what());
                    continue;
                }
                try {
                    if(not h5_src.linkExists("common/finished_all")) { throw except::load_error("could not find dataset [common/finished_all]"); }
                    if(finished and not h5_src.readDataset<bool>("common/finished_all")) { throw except::load_error("simulation has not finished"); }
                    for(const auto &req : src_reqs) {
                        auto path = h5pp::fs::path(req);
                        auto name = h5pp::fs::path(req).filename().string();
                        auto root = path.has_parent_path() ? path.parent_path().string() : "/";
                        if(h5_src.findLinks(name, root).empty()) throw except::load_error("missing required dataset: [{}]", req);
                    }
                    // TODO: REMOVE THIS CHECK
                    //                    auto model_size = h5_src.readAttribute<size_t>("/fLBIT/model/hamiltonian", "model_size");
                    //                    auto bond_lim   = h5_src.readAttribute<long>("/fLBIT/state_real/status", "bond_max");
                    //                    if(model_size >= 28 and bond_lim != 2048) { throw except::load_error("bond_max != 2048"); }
                } catch(const std::exception &ex) {
                    tools::logger::log->warn("skipped file: {}: [{}]", ex.what(), src_abs.string());
                    tools::h5io::saveFailedJob(h5_src, "invalid source file", ex);
                    continue;
                }

                t_open.toc();
                {
                    // Start by copying the environment metadata like git version. This should be the same for all files, so we only do it once
                    // if(not h5_tgt_part.linkExists(".env")) { h5_tgt_part.copyLinkFromFile(".env", h5_src.getFilePath(), ".env"); }
                    //                    auto tgtKeepOpen = h5_tgt_part.getFileHandleToken();
                    //                    auto srcKeepOpen = h5_src.getFileHandleToken();

                    switch(model) {
                        case Model::SDUAL: {
                            tools::h5io::merge<ModelId<sdual, nocircuit>>(h5_tgt_part, h5_src, fileId, keys, tgtdb);
                            break;
                        }
                        case Model::MAJORANA: {
                            tools::h5io::merge<ModelId<majorana, nocircuit>>(h5_tgt_part, h5_src, fileId, keys, tgtdb);
                            break;
                        }
                        case Model::LBIT: {
                            tools::h5io::merge<ModelId<lbit, lbit_circuit>>(h5_tgt_part, h5_src, fileId, keys, tgtdb);
                            break;
                        }
                    }
                }
                tools::logger::log->debug("mem[rss {:<.2f}|peak {:<.2f}|vm {:<.2f}]MB | file db size {}", tools::prof::mem_rss_in_mb(),
                                          tools::prof::mem_hwm_in_mb(), tools::prof::mem_vm_in_mb(), tgtdb.file.size());
            }
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.file);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.model);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.table);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.crono);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.fesup);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.rbds);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.rtes);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.dset);
            tools::h5db::saveDatabase(h5_tgt_part, tgtdb.env);

            tgtdb.file.clear();
            tgtdb.model.clear();
            tgtdb.table.clear();
            tgtdb.crono.clear();
            tgtdb.fesup.clear();
            tgtdb.rbds.clear();
            tgtdb.rtes.clear();
            tgtdb.dset.clear();
            tgtdb.env.clear();

            // TODO: Put the lines below in a "at quick exit" function
            tools::h5io::writeProfiling(h5_tgt_part);

            h5_tgt_part.flush();

            if(use_tmp) {
                tools::logger::log->info("Moving to {} -> {}", h5_tgt_part.getFilePath(), tools::h5io::h5_tgt_part_path);
                h5_tgt_part.moveFileTo(tools::h5io::h5_tgt_part_path, h5pp::FilePermission::REPLACE);
            }
            tools::logger::log->info("Results written to file {}", tools::h5io::h5_tgt_part_path);
            tools::h5dbg::assert_no_dangling_ids(h5_tgt_part, __FUNCTION__, __LINE__); // Check that there are no open HDF5 handles
        }
    }

    mpi::barrier();

    // Now id 1 can merge the files into one final file
    if(mpi::world.id == 0) {
        tools::logger::log->info("Creating main file for external links: {}", tgt_path);
        auto h5_tgt   = h5pp::File(tgt_path, h5pp::FilePermission::REPLACE, verbosity_h5pp);
        auto tgt_stem = h5pp::fs::path(tgt_path).stem().string();
        auto failpath = tgt_dir / "failed.job";
        auto failfile = std::ofstream(failpath.string(), std::ios::trunc | std::ios_base::out);

        for(const auto &obj : h5pp::fs::directory_iterator(tgt_dir, h5pp::fs::directory_options::follow_directory_symlink)) {
            // Take care hdf5 files
            if(obj.is_regular_file() and obj.path().extension() == ".h5") {
                if(obj.path().filename() != tgt_file and obj.path().stem().string().find(tgt_stem) != std::string::npos) {
                    // Found a file that we can link!
                    auto h5_ext = h5pp::File(obj.path().string(), h5pp::FilePermission::READONLY, verbosity_h5pp);
                    // Find the path to the algorithm in this external file
                    auto algo_groups = h5_ext.findGroups(algo, "/", -1, -1); // Need any number of hits (-1) at any recursive depth (-1)
                    if(algo_groups.empty()) {
                        tools::logger::log->error("Could not find algo group {} in external file {}: [{}]", algo, obj.path().string(), algo_groups);
                        continue;
                    }
                    tools::logger::log->info("algo: {} | algo_groups: {}", algo, algo_groups);
                    for(const auto &algo_group : algo_groups) {
                        if(algo_group.find(".env") != algo_group.npos) {
                            // We don't care about the environment group here
                            continue;
                        }
                        bool linkExists = h5_tgt.linkExists(algo_group);
                        if(linkExists) {
                            tools::logger::log->error("The link [{}] is already in the target file [{}]", algo_group, h5_tgt.getFileName());
                            continue;
                        }

                        auto tgt_link = h5pp::fs::proximate(obj.path(), tgt_dir);
                        tools::logger::log->info("Creating external link: {} -> {}", algo_group, tgt_link.string());

                        h5_tgt.createExternalLink(tgt_link.string(), algo_group, algo_group);
                    }
                }
            }
        }
        for(const auto &obj : h5pp::fs::directory_iterator(tgt_dir, h5pp::fs::directory_options::follow_directory_symlink)) {
            // Take care fail files
            if(obj.is_regular_file() and obj.path().extension() == ".job") {
                if(obj.path().filename() != failpath.filename() and obj.path().stem().string().find(tgt_stem) != std::string::npos) {
                    // Found a file with failed jobs!
                    tools::logger::log->info("Appending failed jobs {} -> {}", obj.path().string(), failpath.string());
                    std::ifstream failpart(obj.path().string(), std::ios::in);
                    failfile << failpart.rdbuf();
                }
            }
        }
    }

    mpi::barrier();
    mpi::finalize();
    return 0;
}
