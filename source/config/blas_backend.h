#pragma once

/**
 * @file
 * Configured BLAS backend metadata exposed through a small, vendor-neutral
 * interface.
 */

#include <string>
#include <string_view>

namespace config::blas {
    /**
     * Returns the configured backend identifier, for example `"mkl"` or
     * `"flexiblas"`.
     */
    std::string_view backend_name();

    /**
     * Returns a single-line summary of the configured backend.
     */
    std::string      description();
}
