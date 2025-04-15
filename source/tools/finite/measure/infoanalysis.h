#pragma once
#include "infopolicy.h"
#include <Eigen/Core>

template<typename Scalar>
struct InfoAnalysis {
    using RealScalar  = typename Eigen::NumTraits<Scalar>::Real;
    using RealArrayX  = Eigen::Array<RealScalar, Eigen::Dynamic, 1>;
    using RealArrayXX = Eigen::Array<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;

    double      bits_found    = 0.0;              /*!< The number of bits found. If all goes well this is equal to L. */
    double      icom          = 0.0;              /*!< information center of mass, aka "expected correlation length" or "xi", "<I^\ell>"  */
    double      scale_bit_one = 0.0;              /*!< The minimum length scale to measure 1 bit (the first). */
    double      scale_bit_two = 0.0;              /*!< The minimum length scale to measure 2 bits (the first two).  */
    double      scale_bit_mid = 0.0;              /*!< The minimum length scale to measure L/2 bits. */
    double      scale_bit_pen = 0.0;              /*!< The minimum length scale to measure L-1 bits (all but the last). */
    double      scale_bit_all = 0.0;              /*!< The minimum length scale to measure L bits. */
    InfoPolicy  ip            = {};               /*!< Settings used to calculate this info lattice */
    RealArrayX  info_per_scale;                   /*!< The information per scale "I^\ell" */
    RealArrayXX info_lattice;                     /*!< The information lattice "i^\ell_n" */
    RealArrayXX subsystem_entanglement_entropies; /*!< All the bipartite entanglement entropies (log2) */
};