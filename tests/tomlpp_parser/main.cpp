#include "parse_generated_helpers.h"
#include "settings.h"
#include "simulation_preview.h"
#include <cstdio>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        settings::load(TEST_TOML_FILE);
        settings::parse(argc, argv);
        const auto runtime = test::tomlpp::make_runtime_preview();

        fmt::print("loaded {}\n", TEST_TOML_FILE);
        fmt::print("model          : {} L={}\n", enum2sv(settings::model::model_type), settings::model::model_size);
        fmt::print("algorithms     : xdmrg={} flbit={}\n", settings::xdmrg::on, settings::flbit::on);
        fmt::print("xdmrg          : algo={} ritz={} bonds=[{}, {}]\n", enum2sv(settings::xdmrg::algo), enum2sv(settings::xdmrg::ritz),
                   settings::xdmrg::bond_min, settings::xdmrg::bond_max);
        fmt::print("storage        : emid={} status={}\n", flag2str(settings::storage::mps::state_emid::policy),
                   flag2str(settings::storage::table::status::policy));
        fmt::print("lbit gates     : {} {}\n", enum2sv(settings::model::lbit::u_wkind), enum2sv(settings::model::lbit::u_mkind));
        fmt::print("demo arrays    : sites={} triplet={} bonds={} variances={}\n", settings::demo::measurement_sites, settings::model::lbit::demo_triplet,
                   settings::demo::bond_schedule, settings::demo::variance_targets);
        fmt::print("demo complex   : shift=({:.3f}, {:.3f}) krylov={}\n", runtime.spectral_shift.real(), runtime.spectral_shift.imag(),
                   settings::demo::krylov_window);
        fmt::print("runtime preview: algo={} times={} observables={}\n", runtime.active_algorithm, runtime.preview_times, settings::demo::observables);

        if(settings::demo::measurement_sites.back() >= settings::model::model_size)
            throw std::runtime_error("measurement_sites must be smaller than model_size");
        if(runtime.preview_times.empty()) throw std::runtime_error("expected a non-empty preview time grid");
        if(settings::precision::eigs_iter_min > settings::precision::eigs_iter_max) throw std::runtime_error("eigs_iter_min must not exceed eigs_iter_max");
    } catch(const test::tomlpp::CliExit &ex) { return ex.exit_code; } catch(const std::exception &ex) {
        fmt::print(stderr, "tomlpp_parser failed: {}\n", ex.what());
        return 1;
    }

    return 0;
}
