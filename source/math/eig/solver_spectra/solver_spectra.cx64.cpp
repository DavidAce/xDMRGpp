#include "solver_spectra.impl.h"

using Scalar = cx64;

template class eig::solver_spectra<MatVecMPOS<Scalar>>;
template class eig::solver_spectra<MatVecMPO<Scalar>>;
template class eig::solver_spectra<MatVecDense<Scalar>>;
template class eig::solver_spectra<MatVecSparse<Scalar>>;
template class eig::solver_spectra<MatVecZero<Scalar>>;
