#include "ed.impl.h"

using Scalar = cx128;

/* clang-format off */
template StateFinite<Scalar> tools::finite::ed::find_exact_state(const TensorsFinite<Scalar> &tensors, const AlgorithmStatus &status, tools::finite::opt::reports::eigs_log<Scalar> &elog);