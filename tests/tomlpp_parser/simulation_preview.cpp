#include "simulation_preview.h"
#include <algorithm>
#include <cmath>

namespace test::tomlpp {
    namespace {
        std::vector<double> make_preview_times(std::size_t samples) {
            if(samples == 0) return {};
            if(samples == 1) return {settings::flbit::time_start_real};
            if(settings::flbit::time_scale == TimeScale::LINSPACED) {
                std::vector<double> times(samples, settings::flbit::time_start_real);
                auto                step = (settings::flbit::time_final_real - settings::flbit::time_start_real) / static_cast<double>(samples - 1);
                for(std::size_t idx = 0; idx < samples; ++idx) times[idx] = settings::flbit::time_start_real + step * static_cast<double>(idx);
                return times;
            }

            std::vector<double> times(samples, settings::flbit::time_start_real);
            auto                safe_start = std::max(settings::flbit::time_start_real, 1e-15);
            auto                log_step   = std::log(settings::flbit::time_final_real / safe_start) / static_cast<double>(samples - 1);
            for(std::size_t idx = 0; idx < samples; ++idx) times[idx] = safe_start * std::exp(log_step * static_cast<double>(idx));
            return times;
        }
    }

    RuntimePreview make_runtime_preview() {
        RuntimePreview preview;
        preview.active_algorithm     = settings::xdmrg::on ? "xDMRG" : (settings::flbit::on ? "fLBIT" : "none");
        preview.preview_times        = make_preview_times(std::min<std::size_t>(5, static_cast<std::size_t>(settings::flbit::time_num_steps)));
        preview.spectral_shift       = settings::demo::spectral_shift;
        preview.state_storage_policy = settings::storage::mps::state_emid::policy;
        preview.status_table_policy  = settings::storage::table::status::policy;

        preview.probe_bonds.reserve(settings::demo::bond_schedule.size());
        for(auto bond : settings::demo::bond_schedule) preview.probe_bonds.emplace_back(static_cast<long>(bond));
        return preview;
    }
}
