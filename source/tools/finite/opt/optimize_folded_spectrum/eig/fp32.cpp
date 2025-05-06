#include "impl.h"

using Scalar = fp32;

/* clang-format off */

template opt_mps<Scalar> optimize_folded_spectrum_eig(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<Scalar> &elog);
