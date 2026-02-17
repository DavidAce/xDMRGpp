#include "env_x2.impl.h"

using Scalar = cx128;

/* clang-format off */

template void tools::finite::env::rebuild_edges_ene_x2(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);

template void tools::finite::env::rebuild_edges_var_x2(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);


