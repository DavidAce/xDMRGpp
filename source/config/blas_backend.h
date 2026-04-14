#pragma once

/**
 * @file
 * Backend-specific BLAS functionality exposed through a small, vendor-neutral
 * interface.
 *
 * The rest of the project should use this API instead of including vendor
 * headers directly. That keeps backend-specific preprocessor branching confined
 * to a single implementation unit.
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
     * Applies the requested thread count to the configured backend when the
     * backend exposes runtime thread control.
     *
     * @param num_threads Desired backend thread count.
     */
    void             set_num_threads(int num_threads);

    /**
     * Returns a single-line summary of the configured backend and its most
     * relevant runtime properties.
     */
    std::string      description();
}
