#include "impl.h"

using Scalar = cx32;

template bool tools::finite::mps::normalize_state(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy);
