#pragma once

#include "config/enum_utils.h"

/*! Task identifiers for the finite-system DMRG workflow */
enum class fdmrg_task {
    INIT_RANDOMIZE_MODEL,                /*!< Randomize model parameters */
    INIT_RANDOMIZE_INTO_PRODUCT_STATE,   /*!< Initialize a random product state */
    INIT_RANDOMIZE_INTO_ENTANGLED_STATE, /*!< Initialize a random entangled state */
    INIT_BOND_LIMITS,                    /*!< Initialize bond-dimension limits */
    INIT_TRNC_LIMITS,                    /*!< Initialize truncation-error limits */
    INIT_WRITE_MODEL,                    /*!< Write the model to file */
    INIT_CLEAR_STATUS,                   /*!< Clear the algorithm status */
    INIT_CLEAR_CONVERGENCE,              /*!< Clear convergence tracking */
    INIT_DEFAULT,                        /*!< Run the default initialization task list */
    FIND_GROUND_STATE,                   /*!< Find the ground state */
    FIND_HIGHEST_STATE,                  /*!< Find the highest-energy state */
    POST_WRITE_RESULT,                   /*!< Write the final result to file */
    POST_PRINT_RESULT,                   /*!< Print the final result */
    POST_PRINT_TIMERS,                   /*!< Print timing information */
    POST_RBDS_ANALYSIS,                  /*!< Run reverse bond-dimension scaling analysis */
    POST_RTES_ANALYSIS,                  /*!< Run reverse truncation-error scaling analysis */
    POST_DEFAULT,                        /*!< Run the default post-processing task list */
    TIMER_RESET                          /*!< Reset accumulated timers */
};

template<> std::string_view enum2sv(fdmrg_task item) noexcept;
template<> fdmrg_task       sv2enum<fdmrg_task>(std::string_view item);
