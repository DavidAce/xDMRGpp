#include "standard_basis_change.impl.h"

using Scalar = fp128;

/* clang-format off */
using namespace tools::finite::opt::precond::standard;

template struct tools::finite::opt::precond::standard::BasisChange<Scalar>;