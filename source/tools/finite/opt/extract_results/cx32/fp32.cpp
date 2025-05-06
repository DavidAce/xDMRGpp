#include "../impl.h"

using Scalar = cx32;
using T = fp32;

/* clang-format off */

template void tools::finite::opt::internal::extract_results<T>(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<Scalar>> &results, bool converged_only, std::optional<std::vector<long>> indices);

template void tools::finite::opt::internal::extract_results_subspace<T>(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<Scalar>> &subspace_mps, std::vector<opt_mps<Scalar>> &results);
