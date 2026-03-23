#pragma once

namespace test::enumcppgen_demo {

// enumgen: bitflag
/*! \brief When to do projections to a symmetry sector. */
enum class ProjectionPolicy : int {
    NEVER     = 0,                        /*!< Never project */
    INIT      = 1,                        /*!< Project after initializing the state */
    WARMUP    = 2,                        /*!< Project during warmup */
    STUCK     = 4,                        /*!< Project when the algorithm is stuck */
    ITER      = 8,                        /*!< Project every iteration */
    CONVERGED = 16,                       /*!< Project on converged iterations */
    FINISHED  = 32,                       /*!< Project the finished state during postprocessing */
    FORCE     = 64,                       /*!< Project even if not needed */
    DEFAULT   = INIT | STUCK | CONVERGED, /*!< Default policy for multisite dmrg steps */
};

}
