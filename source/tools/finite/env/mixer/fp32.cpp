#include "mixer.impl.h"

using T = fp32;

template void tools::finite::env::internal::run_expansion_term_mixer (TensorsFinite<T> &tensors, long posP, long pos0, T pad_value_env, BondExpansionConfig bcfg);