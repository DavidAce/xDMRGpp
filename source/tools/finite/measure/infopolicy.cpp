#include "infopolicy.h"

bool InfoPolicy::is_compatible(const std::optional<InfoPolicy> &other) {
    if(!other.has_value()) return false;
    bool precision_ok = precision <= other->precision;
    bool bitserror_ok = std::abs(bits_max_error.value_or(-0.5)) >= std::abs(other->bits_max_error.value_or(-0.5));
    return precision_ok and bitserror_ok;
}