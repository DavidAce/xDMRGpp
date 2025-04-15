#pragma once
#include "config/enums.h"
#include <optional>

struct InfoPolicy {
    std::optional<double>    bits_max_error = std::nullopt; /*!< Positive for relative error = 1-bits_found/L, negative for absolute error = L-bits_found */
    std::optional<size_t>    eig_max_size   = std::nullopt; /*!< Maximum matrix size to diagonalize (skip if larger). Recommend <= 8192 */
    std::optional<double>    svd_max_size   = std::nullopt; /*!< Maximum matrix size for svd during swaps (skip if larger) Recommend <= 4096 */
    std::optional<double>    svd_trnc_lim   = std::nullopt; /*!< Maximum discarded weight in the svd during swaps. Recommend <= 1e-6 */
    std::optional<Precision> precision      = std::nullopt; /*!< Use single (fp32) or double (fp64) precision internally */
    bool                     is_compatible(const std::optional<InfoPolicy> &other);
};
