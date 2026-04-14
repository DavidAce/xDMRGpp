#pragma once

#include "payload.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <string>

namespace fmtab {
    inline std::string render_std_report(int unit) {
        auto payload = make_payload(unit);
        auto lines   = std::string{};
        lines += fmt::format("unit {:02d}\n", unit);
        lines += fmt::format("path {}\n", payload.path);
        lines += fmt::format("paths {}\n", payload.paths);
        lines += fmt::format("opt_paths {}\n", payload.opt_paths);
        lines += fmt::format("ids {}\n", payload.ids);
        lines += fmt::format("opt_ids {}\n", payload.opt_ids);
        lines += fmt::format("coeffs {::+9.2f}\n", payload.coeffs);
        lines += fmt::format("phases {::+6.2f}\n", payload.phases);
        lines += fmt::format("opt_phase {:+7.3f}\n", payload.opt_phase);
        lines += fmt::format("path_list_again {}\n", payload.paths);
        lines += fmt::format("coeffs_again {::+9.2f}\n", payload.coeffs);
        lines += fmt::format("phases_again {::+6.2f}\n", payload.phases);
        return lines;
    }
}
