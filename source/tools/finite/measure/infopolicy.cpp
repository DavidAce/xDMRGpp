#include "infopolicy.h"

bool InfoPolicy::is_compatible(const std::optional<InfoPolicy> &other) {
    if(!other.has_value()) return false;

    // Require both to be set
    if(!precision or !other->precision) return false;
    if(!bits_max_error or !other->bits_max_error) return false;

    // must be at least as accurate as requested.
    // Precision: cached precision >= requested precision (treat enum ordering as increasing accuracy).
    bool precision_ok = static_cast<int>(other->precision.value()) >= static_cast<int>(precision.value());

    // bits_max_error semantics:
    //  - positive: relative error threshold (smaller is stricter)
    //  - negative: absolute missing-bits threshold (more negative is looser)
    // So we only reuse if both use the same mode, and cached is stricter or equal.
    const double req          = bits_max_error.value();
    const double got          = other->bits_max_error.value();
    bool         bitserror_ok = false;

    if(req >= 0.0 && got >= 0.0) {
        bitserror_ok = got <= req; // cached relative error threshold is smaller or equal
    } else if(req < 0.0 && got < 0.0) {
        bitserror_ok = got >= req; // cached absolute threshold is closer to 0 (stricter)
    }

    return precision_ok and bitserror_ok;
}