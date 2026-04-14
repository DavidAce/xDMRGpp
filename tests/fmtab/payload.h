#pragma once

#include <array>
#include <complex>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fmtab {
    struct Payload {
        std::filesystem::path                           path;
        std::vector<std::filesystem::path>              paths;
        std::optional<std::vector<std::filesystem::path>> opt_paths;
        std::vector<size_t>                             ids;
        std::optional<std::vector<size_t>>              opt_ids;
        std::array<double, 3>                           coeffs;
        std::vector<std::complex<double>>               phases;
        std::optional<std::complex<double>>             opt_phase;
    };

    inline Payload make_payload(int unit) {
        auto suffix  = std::to_string(unit);
        auto path_a  = std::filesystem::path("/tmp/fmtab") / ("left_" + suffix + ".h5");
        auto path_b  = std::filesystem::path("/tmp/fmtab") / ("right_" + suffix + ".h5");
        auto main    = std::filesystem::path("/tmp/fmtab") / ("report_" + suffix + ".h5");
        auto ids     = std::vector<size_t>{static_cast<size_t>(unit), static_cast<size_t>(unit + 2), static_cast<size_t>(unit + 4)};
        auto coeffs  = std::array<double, 3>{0.25 + 0.01 * unit, 0.50 + 0.01 * unit, 0.75 + 0.01 * unit};
        auto phases  = std::vector<std::complex<double>>{{1.0 + 0.1 * unit, 0.5 + 0.1 * unit}, {-2.0 - 0.1 * unit, 3.0 + 0.1 * unit}};

        return {
            .path      = std::move(main),
            .paths     = {std::move(path_a), std::move(path_b)},
            .opt_paths = std::vector<std::filesystem::path>{std::filesystem::path("/tmp/fmtab") / ("opt_left_" + suffix + ".h5"),
                                                            std::filesystem::path("/tmp/fmtab") / ("opt_right_" + suffix + ".h5")},
            .ids       = std::move(ids),
            .opt_ids   = std::vector<size_t>{static_cast<size_t>(unit + 10), static_cast<size_t>(unit + 20), static_cast<size_t>(unit + 30)},
            .coeffs    = coeffs,
            .phases    = std::move(phases),
            .opt_phase = std::complex<double>{4.0 + 0.1 * unit, -1.5 - 0.1 * unit},
        };
    }
}
