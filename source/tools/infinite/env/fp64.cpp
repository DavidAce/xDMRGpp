#include "impl.h"

using Scalar = fp64;

/* clang-format off */

template void tools::infinite::env::reset_edges(const StateInfinite<Scalar> &state, const ModelInfinite<Scalar> &model, EdgesInfinite<Scalar> &edges);

template void tools::infinite::env::enlarge_edges(const StateInfinite<Scalar> &state, const ModelInfinite<Scalar> &model, EdgesInfinite<Scalar> &edges);
