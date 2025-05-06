#include "../matvec_sparse.impl.h"

using Scalar = cx32;
template class MatVecSparse<Scalar, false>;
template class MatVecSparse<Scalar, true>;
