#include "matvec_sparse.impl.h"

using Scalar = cx128;
template class MatVecSparse<Scalar, false>;
template class MatVecSparse<Scalar, true>;

