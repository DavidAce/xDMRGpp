//
// Created by david on 2021-10-12.
//

#include "parse.h"
#include "config/enums.h"
#include "config/loader.h"
#include "config/settings.h"
#include "tools/common/log.h"
#include <CLI/CLI.hpp>
#include <h5pp/h5pp.h>
#include "settings.h"

template<>
constexpr h5pp::LogLevel sv2enum<h5pp::LogLevel>(std::string_view item) {
    if(item == "trace") return h5pp::LogLevel::trace;
    if(item == "debug")
        return h5pp::LogLevel::debug;
    else
        return h5pp::LogLevel::info;
}

template<>
constexpr spdlog::level::level_enum sv2enum<spdlog::level::level_enum>(std::string_view item) {
    if(item == "trace") return spdlog::level::level_enum::trace;
    if(item == "debug")
        return spdlog::level::level_enum::debug;
    else
        return spdlog::level::level_enum::info;
}

// MWE: https://godbolt.org/z/jddxod53d
int settings::parse_fill(int argc, char **argv) {
    using namespace settings;
    using namespace h5pp;
    using namespace spdlog;

    auto s2e_log     = mapStr2Enum<spdlog::level::level_enum>("trace", "debug", "info");
    auto s2e_logh5pp = mapStr2Enum<h5pp::LogLevel>("trace", "debug", "info");
    auto s2e_model   = mapEnum2Str<ModelType>(ModelType::ising_tf_rf, ModelType::ising_sdual, ModelType::ising_majorana, ModelType::lbit);



    CLI::App app;
    app.description("fill: Adds missing data to hdf5 files");
    app.get_formatter()->column_width(90);
    app.option_defaults()->always_capture_default();
    app.allow_extras(false);
    /* clang-format off */
    app.add_flag("--help-preload"                      , "Print help related to preloading configuration");
    app.add_option("-p,--prefix"                       , settings::prefix               , "Path to the simulation data folder e.g. /mnt/WDB-AN1500/mbl_transition/xdmrg3-letsgo")->required();
    app.add_option("-f,--filter"                       , settings::filter               , "Path filter -- skip files whose path do not contain this string");
    app.add_option("-o,--outdir"                       , settings::outdir               , "Put new files in this directory")->required();
    app.add_option("-t,--threads"                      , threading::num_threads         , "Total number of threads (omp + std threads). Use env OMP_NUM_THREADS to control omp.");
    app.add_option("--output-dirname"                  , settings::output_dirname       , "Name of the output directory under 'prefix'");
    app.add_option("--config-dirname"                  , settings::config_dirname       , "Name of the config directory under 'prefix'");
    app.add_option("--config-default"                  , settings::config_default       , "Name of the default config file to load (in case config settings are missing in older simulations)");
    app.add_flag("--show-threads"                      , threading::show_threads        , "Show information about threading and exit immediately");
    app.add_option("-z,--compression"                  , storage::compression_level     , "Compression level of h5pp")->check(CLI::Range(0,9));
    app.add_flag  ("--inplace"                                                          , "Add data in place");
    app.add_option("-v,--log,--verbosity,--loglevel"   , console::loglevel              , "Log level of xDMRG++")->transform(CLI::CheckedTransformer(s2e_log, CLI::ignore_case))->type_name("ENUM");
    app.add_option("-V,--logh5pp"                      , console::logh5pp               , "Log level of h5pp")->transform(CLI::CheckedTransformer(s2e_logh5pp, CLI::ignore_case))->type_name("ENUM");

    /* clang-format on */

    app.parse(argc, argv);


    return 0;
}
