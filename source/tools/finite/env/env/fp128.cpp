#include "impl.h"

using Scalar = fp128;

/* clang-format off */

template void tools::finite::env::assert_edges_ene(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, const EdgesFinite<Scalar> &);

template void tools::finite::env::assert_edges_var(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, const EdgesFinite<Scalar> &);

template void tools::finite::env::assert_edges(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, const EdgesFinite<Scalar> &);

template void tools::finite::env::rebuild_edges_ene(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);

template void tools::finite::env::rebuild_edges_var(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);

template void tools::finite::env::rebuild_edges(const StateFinite<Scalar> &, const ModelFinite<Scalar> &, EdgesFinite<Scalar> &);



