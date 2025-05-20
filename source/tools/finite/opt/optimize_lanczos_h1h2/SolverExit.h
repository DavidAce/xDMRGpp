#pragma once

enum class SolverExit : int {
    ok                    = 0,   /*! No issues on this Lanczos step: can keep running */
    converged_rnormTol    = 1,   /*! The residual norm fell below a threshold. */
    converged_subspace    = 2,   /*! The columns in V are already exact solutions: no additional directions could be found */
    saturated_subspace    = 4,   /*! Exhausted the subspace before reaching ncv columns */
    saturated_relDiffTol  = 8,   /*! The relative value change fell below a threshold.  */
    saturated_absDiffTol  = 16,  /*! The absolute value change fell below a threshold.  */
    max_iterations        = 32,  /*! The maximum number of iterations was reached. */
    max_matvecs           = 64,  /*! The maximum number of matrix-vector multiplications. */
    one_valid_eigenvector = 128, /*! Only one valid eigenvector was found. */
    no_valid_eigenvector  = 256, /*! No valid eigenvector was found. */
    allow_bitops
};