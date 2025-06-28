#include "matrix_inverse_vector_product.impl.h"
using Scalar = cx64;


template tools::common::contraction::VectorType<Scalar>
    tools::common::contraction::matrix_inverse_vector_product(MatrixLikeOperator<Scalar>                &MatrixOp, //
                                                              const Scalar                              *rhs_ptr,  //
                                                              const IterativeLinearSolverConfig<Scalar> &cfg);

