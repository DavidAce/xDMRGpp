//
// Created by david on 2021-10-12.
//

#include "parse.h"
#include "loader.h"
#include "settings.h"
#include "tools/common/log.h"
#include <algorithm>
#include <cctype>
#include <string_view>
#include <CLI/CLI.hpp>
#include <h5pp/h5pp.h>

namespace {
    bool has_full_help_flag(int argc, char **argv) {
        for(int idx = 1; idx < argc; ++idx) {
            const std::string_view arg = argv[idx];
            if(arg == "-h" or arg == "--help") return true;
        }
        return false;
    }

    int parse_gpu_id(std::string value) {
        std::ranges::transform(value, value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if(value == "auto") return -1;
        std::size_t pos = 0;
        int         dev = std::stoi(value, &pos);
        if(pos != value.size()) throw CLI::ValidationError("--gpu-id", "expected 'auto', -1, or a non-negative integer");
        if(dev < -1) throw CLI::ValidationError("--gpu-id", "expected 'auto', -1, or a non-negative integer");
        return dev;
    }

    settings::ParseResult parse_app(CLI::App &app, int argc, char **argv) {
        try {
            app.parse(argc, argv);
        } catch(const CLI::ParseError &err) {
            return {settings::ParseAction::EXIT, app.exit(err)};
        }
        return {};
    }
}

template<typename T>
std::string filename_append_seed(const std::string &filename, const T number) {
    if constexpr(std::is_signed_v<T>)
        if(number < 0) return filename;
    if constexpr(std::is_unsigned_v<T>)
        if(number == std::numeric_limits<T>::max()) return filename;
    // Append the seed_model to the output filename
    h5pp::fs::path         oldFileName  = filename;
    h5pp::fs::path         newFileName  = filename;
    auto                   oldstem      = oldFileName.stem().string();
    std::string::size_type old_seed_pos = oldstem.find_first_of('_');
    std::string::size_type old_bitf_pos = std::string::npos;
    if(old_seed_pos != std::string::npos) { old_bitf_pos = oldstem.find_first_of('_', old_seed_pos + 1); }
    std::string old_prfx_str = oldstem.substr(0, oldstem.find_first_of("_."));
    std::string old_seed_str = "";
    std::string old_bitf_str = "";
    if(old_seed_pos != std::string::npos) { old_seed_str = oldstem.substr(old_seed_pos + 1, old_bitf_pos); }
    if(old_bitf_pos != std::string::npos) { old_bitf_str = oldstem.substr(old_bitf_pos + 1, std::string::npos); }
    if(old_seed_str == std::to_string(number)) return oldFileName.string();
    if(not old_seed_str.empty()) {
        throw except::file_error("Tried to append seed {} to filename, but another seed is already present: {}", number, oldFileName.string());
    }
    // The seed number is not present!
    std::string newName;
    if(old_bitf_str.empty()) {
        newName = fmt::format("{}_{}{}", old_prfx_str, number, oldFileName.extension().string());
    } else {
        newName = fmt::format("{}_{}_{}{}", old_prfx_str, number, old_bitf_str, oldFileName.extension().string());
    }
    newFileName.replace_filename(newName);
    tools::log->info("Appended seed [{}] to filename: [{}]", number, newFileName.string());
    return newFileName.string();
}

std::string filename_append_pattern(const std::string &filename, const std::string &pattern) {
    if(pattern.empty()) return filename;
    // Append the seed_model to the output filename
    h5pp::fs::path         oldFileName  = filename;
    h5pp::fs::path         newFileName  = filename;
    auto                   oldstem      = oldFileName.stem().string();
    std::string::size_type old_seed_pos = oldstem.find_first_of('_');
    std::string::size_type old_bitf_pos = std::string::npos;
    if(old_seed_pos != std::string::npos) { old_bitf_pos = oldstem.find_first_of('_', old_seed_pos + 1); }
    std::string old_prfx_str = oldstem.substr(0, oldstem.find_first_of("_."));
    std::string old_seed_str = "";
    std::string old_bitf_str = "";
    if(old_seed_pos != std::string::npos) { old_seed_str = oldstem.substr(old_seed_pos + 1, old_bitf_pos); }
    if(old_bitf_pos != std::string::npos) { old_bitf_str = oldstem.substr(old_bitf_pos + 1, std::string::npos); }
    if(old_bitf_str == pattern) return oldFileName.string();
    //    tools::log->info("old_seed_pos {}\n"
    //                     "old_seed_str {}\n"
    //                     "old_bitf_pos {}\n"
    //                     "old_bitf_str {}",
    //                     old_seed_pos, old_seed_str, old_bitf_pos, old_bitf_str);

    if(not old_bitf_str.empty()) {
        throw except::file_error("Tried to append pattern {} to filename, but another pattern is already present: {}", pattern, oldFileName.string());
    }
    // The pattern is not present!
    std::string newName;
    if(old_seed_str.empty()) {
        newName = fmt::format("{}_{}{}", old_prfx_str, pattern, oldFileName.extension().string());
    } else {
        newName = fmt::format("{}_{}_{}{}", old_prfx_str, old_seed_str, pattern, oldFileName.extension().string());
    }
    newFileName.replace_filename(newName);
    tools::log->info("Appended pattern [{}] to filename: [{}]", pattern, newFileName.string());
    return newFileName.string();
}

template<>
h5pp::LogLevel sv2enum<h5pp::LogLevel>(std::string_view item) {
    if(item == "trace") return h5pp::LogLevel::trace;
    if(item == "debug")
        return h5pp::LogLevel::debug;
    else
        return h5pp::LogLevel::info;
}

template<>
spdlog::level::level_enum sv2enum<spdlog::level::level_enum>(std::string_view item) {
    if(item == "trace") return spdlog::level::level_enum::trace;
    if(item == "debug")
        return spdlog::level::level_enum::debug;
    else
        return spdlog::level::level_enum::info;
}

// MWE: https://godbolt.org/z/jddxod53d
settings::ParseResult settings::parse(int argc, char **argv) {
    using namespace settings;
    using namespace h5pp;
    using namespace spdlog;

    auto s2e_log     = mapStr2Enum<spdlog::level::level_enum>("trace", "debug", "info");
    auto s2e_logh5pp = mapStr2Enum<h5pp::LogLevel>("trace", "debug", "info");
    auto s2e_model   = mapEnum2Str<ModelType>(ModelType::ising_tf_rf, ModelType::ising_sdual, ModelType::ising_majorana, ModelType::lbit);
    auto s2e_eigslib = mapEnum2Str<EigsLibrary>(EigsLibrary::ARPACK, EigsLibrary::SPECTRA, EigsLibrary::PRIMME, EigsLibrary::EIGSMPO, EigsLibrary::GRIT);
    auto s2e_gpu_policy = mapEnum2Str<GpuPolicy>(GpuPolicy::ON, GpuPolicy::OFF, GpuPolicy::TRY);
    auto gpu_id_text    = settings::cuda::gpu_id < 0 ? std::string{"auto"} : std::to_string(settings::cuda::gpu_id);
    int  dummy       = 0;

    auto preload = [&argc, &argv, &s2e_log]() -> ParseResult {
        CLI::App pre;
        pre.get_formatter()->column_width(90);
        pre.option_defaults()->always_capture_default();
        pre.allow_extras(true);
        pre.set_help_flag("--help-preload", "Help for preloading configuration");
        /* clang-format off */
        pre.add_option("-c,--config"                       , input::config_filename , "Path to a .cfg or .h5 file from a previous simulation");
        pre.add_option("-v,--log,--verbosity,--loglevel"   , console::loglevel      , "Log level of xDMRG++")->transform(CLI::CheckedTransformer(s2e_log, CLI::ignore_case))->type_name("ENUM");
        pre.add_option("--timestamp"                       , console::timestamp     , "Log timestamp");
        /* clang-format on */
        if(auto result = parse_app(pre, argc, argv); result.action == ParseAction::EXIT) return result;
        tools::log = tools::Logger::setLogger("xDMRG++ config", settings::console::loglevel, settings::console::timestamp);
        tools::log->info("Preloading {}", input::config_filename);
        //  Try loading the given config file.
        //  Note that there is a default "input/input.config" if none was given
        Loader dmrg_config(settings::input::config_filename);
        if(dmrg_config.file_exists) {
            dmrg_config.load();
            settings::load(dmrg_config);
        } else if(pre.get_option("--config")->empty()) {
            tools::log->warn("The default config file does not exist: {}", input::config_filename);
        } else
            throw except::runtime_error("Could not find config file: {}", settings::input::config_filename); // Invalid file
        return {};
    };
    const auto full_help_requested = has_full_help_flag(argc, argv);
    if(not full_help_requested)
        if(auto result = preload(); result.action == ParseAction::EXIT) return result;

    CLI::App app;
    app.description("xDMRG++: An MPS-based algorithm to calculate 1D quantum-states");
    app.get_formatter()->column_width(90);
    app.option_defaults()->always_capture_default();
    app.allow_extras(false);
    /* clang-format off */
    app.add_flag("--help-preload"                      , "Print help related to preloading configuration");
    app.add_option("-c,--config"                       , input::config_filename         , "Path to a .cfg or .h5 file from a previous simulation");
    app.add_option("-m,--model"                        , model::model_type              , "Select the Hamiltonian")->transform(CLI::CheckedTransformer(s2e_model, CLI::ignore_case));
    app.add_option("-b,--bitfield,--pattern"           , state::init::initial_pattern      , "Integer whose bitfield sets the initial product state. Negative is unused");
    app.add_option("-o,--outfile"                      , storage::output_filepath       , "Path to the output file. The seed number gets appended by default (see -x)");
    app.add_option("-s,--seed"                         , input::seed                    , "Positive number seeds the random number generator");
    app.add_option("-t,--threads"                      , threading::num_threads         , "Total number of threads (omp + std threads). Use env OMP_NUM_THREADS to control omp.");
    app.add_option("--eigslib"                         , solvers::eig::eigslib          , "Iterative eigensolver backend [ARPACK | SPECTRA | PRIMME | EIGSMPO | GRIT]")->transform(CLI::CheckedTransformer(s2e_eigslib, CLI::ignore_case))->type_name("ENUM");
    app.add_option("--gpu-policy"                      , cuda::gpu_policy               , "GPU contraction policy [ON | OFF | TRY]")->transform(CLI::CheckedTransformer(s2e_gpu_policy, CLI::ignore_case))->type_name("ENUM");
    app.add_option("--gpu-id"                          , gpu_id_text                    , "CUDA device id. Use auto or -1 to select the first working GPU");
    app.add_option("--gpu-switchsize"                  , cuda::gpu_switchsize           , "Minimum linear problem size before AUTO matvec may switch to the GPU");
    app.add_option("--gpu-max-alloc-fraction"          , cuda::gpu_max_alloc_fraction   , "Refuse GPU matvec when the estimated device allocation exceeds this fraction of free memory")->check(CLI::Range(0.0, 1.0));
    app.add_flag("--show-threads"                      , threading::show_threads        , "Show information about threading and exit immediately");
    app.add_flag  ("--append-seed, !--no-append-seed"  , storage::output_append_seed    , "Append seed to the output filename")->default_val(true);
    app.add_option("-z,--compression"                  , storage::compression_level     , "Compression level of h5pp")->check(CLI::Range(0,9));
    app.add_option("--resume-iter"                     , storage::file_resume_iter      , "Resume from iteration");
    app.add_option("--resume-name"                     , storage::file_resume_name      , "Resume from state matching this name");
    app.add_flag  ("-r,--resume"                                                        , "Resume simulation from last iteration");
    app.add_flag  ("--replace"                                                          , "Replace the output file and start from the beginning")->excludes("--resume", "--resume-iter", "--resume-name");
    app.add_flag  ("--revive"                                                           , "Replace the output file and start from the beginning")->excludes("--resume", "--resume-iter", "--resume-name", "--replace");
    app.add_option("-v,--log,--verbosity,--loglevel"   , console::loglevel              , "Log level of xDMRG++")->transform(CLI::CheckedTransformer(s2e_log, CLI::ignore_case))->type_name("ENUM");
    app.add_option("-V,--logh5pp"                      , console::logh5pp               , "Log level of h5pp")->transform(CLI::CheckedTransformer(s2e_logh5pp, CLI::ignore_case))->type_name("ENUM");
    app.add_option("--timestamp"                       , console::timestamp             , "Log timestamp");
    app.add_option("--dummyrange"                      , dummy                          , "Dummy")->check(CLI::Range(0,3));
    app.add_flag("--test-unwind"                       , test_unwind, "Throw an error to test stack unwinding");

    /* clang-format on */

    if(auto result = parse_app(app, argc, argv); result.action == ParseAction::EXIT) return result;
    settings::cuda::gpu_id = parse_gpu_id(gpu_id_text);

    if(app.count("--resume") > 0 or app.count("--resume-iter") > 0 or app.count("--resume-name") > 0) {
        tools::log->info("Resuming from iter {}", storage::file_resume_iter);
        settings::storage::file_collision_policy = FileCollisionPolicy::RESUME;
    }
    if(app.count("--replace") > 0) {
        tools::log->info("Replacing file");
        settings::storage::file_collision_policy = FileCollisionPolicy::REPLACE;
    }
    if(app.count("--revive") > 0) {
        tools::log->info("Reviving file");
        settings::storage::file_collision_policy = FileCollisionPolicy::REVIVE;
    }
    // Generate the correct output filename based on given seeds
    if(storage::output_append_seed) {
        settings::storage::output_filepath = filename_append_seed(settings::storage::output_filepath, settings::input::seed);
        settings::storage::output_filepath = filename_append_pattern(settings::storage::output_filepath, settings::state::init::initial_pattern);
    }
    return {};
}
