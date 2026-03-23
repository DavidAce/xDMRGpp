#pragma once

#include "settings.h"
#include <complex>
#include <string>
#include <vector>

namespace test::tomlpp {
    struct RuntimePreview {
        std::string          active_algorithm;
        std::vector<double>  preview_times;
        std::vector<long>    probe_bonds;
        std::complex<double> spectral_shift;
        StoragePolicy        state_storage_policy = StoragePolicy::NONE;
        StoragePolicy        status_table_policy  = StoragePolicy::NONE;
    };

    [[nodiscard]] RuntimePreview make_runtime_preview();
}
