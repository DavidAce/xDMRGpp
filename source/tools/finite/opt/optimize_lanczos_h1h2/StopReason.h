#pragma once

enum class StopReason : int {
    none                  = 0,   /*! No issues on this Lanczos step: can keep running */
    converged_rnormTol    = 1,   /*! The residual norm fell below a threshold. */
    saturated_basis       = 2,   /*! no additional basis vectors could be found */
    saturated_relDiffTol  = 4,   /*! The relative value change fell below a threshold.  */
    saturated_absDiffTol  = 8,   /*! The absolute value change fell below a threshold.  */
    max_iterations        = 16,  /*! The maximum number of iterations was reached. */
    max_matvecs           = 32,  /*! The maximum number of matrix-vector multiplications. */
    one_valid_eigenvector = 64,  /*! Only one valid eigenvector was found. */
    no_valid_eigenvector  = 128, /*! No valid eigenvector was found. */
    allow_bitops
};