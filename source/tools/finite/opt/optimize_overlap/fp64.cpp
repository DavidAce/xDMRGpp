#include "impl.h"

using Scalar = fp64;

/* clang-format off */

template opt_mps<Scalar>  tools::finite::opt::internal::optimize_overlap(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, OptMeta &meta, reports::subs_log<Scalar> &elog);
