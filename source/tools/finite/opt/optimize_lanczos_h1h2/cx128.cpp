#include "impl.h"

using Scalar = cx128;

/* clang-format off */

template opt_mps<Scalar>  tools::finite::opt::internal::optimize_lanczos_h1h2(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<Scalar> &elog);
