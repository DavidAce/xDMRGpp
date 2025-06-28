#include "impl.h"

using Scalar = cx32;

/* clang-format off */

template opt_mps<Scalar> tools::finite::opt::internal::optimize_energy(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, OptMeta &meta, reports::eigs_log<Scalar> &elog);
