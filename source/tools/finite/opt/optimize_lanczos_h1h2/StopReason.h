#pragma once

enum class StopReason : int {
    none                  = 0,   /*! No issues on this Lanczos step: can keep running */
    converged_rNorm       = 1,   /*! The residual norm fell below the convergence threshold. */
    saturated_rNorm       = 2,   /*! The residual norm stopped decreasing */
    saturated_basis       = 4,   /*! No additional basis vectors could be found */
    saturated_relDiffTol  = 8,   /*! The relative value change fell below the threshold.  */
    saturated_absDiffTol  = 16,  /*! The absolute value change fell below the threshold.  */
    max_iterations        = 32,  /*! The maximum number of iterations was reached. */
    max_matvecs           = 64,  /*! The maximum number of matrix-vector multiplications. */
    one_valid_eigenvector = 128, /*! Only one valid eigenvector was found. */
    no_valid_eigenvector  = 256, /*! No valid eigenvector was found. */
    allow_bitops
};