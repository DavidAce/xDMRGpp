#include "impl.h"

using Scalar = cx128;

/* clang-format off */
template struct tools::finite::opt::internal::Comparator<Scalar>;

template class tools::finite::opt::internal::EigIdxComparator<Scalar>;

template tools::finite::opt::opt_mps<Scalar>  tools::finite::opt::get_opt_initial_mps(const TensorsFinite<Scalar> &tensors, const OptMeta &meta);

template tools::finite::opt::opt_mps<Scalar>  tools::finite::opt::find_ground_state(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,  OptMeta &meta);

template tools::finite::opt::opt_mps<Scalar>  tools::finite::opt::get_updated_state(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,  OptMeta &meta);
