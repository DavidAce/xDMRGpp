#include "runner.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstdlib>

int main() {
    auto std_outputs  = fmtab::collect_std_outputs();
    auto wrap_outputs = fmtab::collect_wrap_outputs();

    if(std_outputs.size() != wrap_outputs.size()) {
        fmt::print(stderr, "size mismatch: std={} wrap={}\n", std_outputs.size(), wrap_outputs.size());
        return EXIT_FAILURE;
    }

    for(size_t idx = 0; idx < std_outputs.size(); ++idx) {
        if(std_outputs[idx] != wrap_outputs[idx]) {
            fmt::print(stderr, "mismatch at unit {}\n", idx);
            fmt::print(stderr, "--- std ---\n{}\n", std_outputs[idx]);
            fmt::print(stderr, "--- wrap ---\n{}\n", wrap_outputs[idx]);
            return EXIT_FAILURE;
        }
    }

    fmt::print("matched {} reports\n", std_outputs.size());
    return EXIT_SUCCESS;
}
