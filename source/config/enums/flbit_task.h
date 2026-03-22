#pragma once

#include "config/enum_utils.h"

/*! Task identifiers for the finite-system l-bit workflow */
enum class flbit_task {
    INIT_RANDOMIZE_MODEL,                              /*!< Randomize model parameters */
    INIT_RANDOMIZE_INTO_PRODUCT_STATE,                 /*!< Initialize a random product state */
    INIT_RANDOMIZE_INTO_PRODUCT_STATE_NEEL_SHUFFLED,   /*!< Initialize a shuffled Neel product state */
    INIT_RANDOMIZE_INTO_PRODUCT_STATE_NEEL_DISLOCATED, /*!< Initialize a dislocated Neel product state */
    INIT_RANDOMIZE_INTO_ENTANGLED_STATE,               /*!< Initialize a random entangled state */
    INIT_RANDOMIZE_INTO_PRODUCT_STATE_PATTERN,         /*!< Initialize a patterned product state */
    INIT_RANDOMIZE_INTO_MIDCHAIN_SINGLET_NEEL_STATE,   /*!< Initialize a midchain-singlet Neel state */
    INIT_BOND_LIMITS,                                  /*!< Initialize bond-dimension limits */
    INIT_TRNC_LIMITS,                                  /*!< Initialize truncation-error limits */
    INIT_WRITE_MODEL,                                  /*!< Write the model to file */
    INIT_CLEAR_STATUS,                                 /*!< Clear the algorithm status */
    INIT_CLEAR_CONVERGENCE,                            /*!< Clear convergence tracking */
    INIT_DEFAULT,                                      /*!< Run the default initialization task list */
    INIT_GATES,                                        /*!< Build the time-evolution gates */
    INIT_TIME,                                         /*!< Initialize the time grid */
    TRANSFORM_TO_LBIT,                                 /*!< Transform the state into the l-bit basis */
    TRANSFORM_TO_REAL,                                 /*!< Transform the state back to the real-spin basis */
    TIME_EVOLVE,                                       /*!< Run the time-evolution step */
    POST_WRITE_RESULT,                                 /*!< Write the final result to file */
    POST_PRINT_RESULT,                                 /*!< Print the final result */
    POST_PRINT_TIMERS,                                 /*!< Print timing information */
    POST_DEFAULT,                                      /*!< Run the default post-processing task list */
    TIMER_RESET                                        /*!< Reset accumulated timers */
};

template<> std::string_view enum2sv(flbit_task item) noexcept;
template<> flbit_task       sv2enum<flbit_task>(std::string_view item);
