#pragma once
#include <array>
#include <functional>
#include <optional>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
template<typename T>
struct BondExpansionResult {
    using R                                    = typename Eigen::NumTraits<T>::Real;
    bool                             ok        = false; /*!< True if the expansion took place (false on queries, e.g., BondExpansionPolicy == NONE) */
    long                             direction = 0;     /*!< 1: expansion left to right, -1: expansion right to left */
    long                             posL      = -1;    /*!< Position left of the expanded bond */
    long                             posR      = -1;    /*!< Position right of the expanded bond */
    std::array<long, 3>              dimL_old  = {};    /*!< Dimensions of the left site before the expansion */
    std::array<long, 3>              dimL_new  = {};    /*!< Dimensions of the left site after expanding and truncating */
    std::array<long, 3>              dimR_old  = {};    /*!< Dimensions of the right site before the expansion */
    std::array<long, 3>              dimR_new  = {};    /*!< Dimensions of the right site after expanding and truncating */
    std::vector<size_t>              sites;
    std::vector<std::array<long, 3>> dims_old, dims_new;
    std::vector<long>                bond_old, bond_new;
    R                                ene_old        = std::numeric_limits<R>::quiet_NaN();      /*!< The old expectation value  of energy */
    R                                ene_new        = std::numeric_limits<R>::quiet_NaN();      /*!< The old expectation value  of energy */
    R                                var_old        = std::numeric_limits<R>::quiet_NaN();      /*!< The new expectation value  of energy variance*/
    R                                var_new        = std::numeric_limits<R>::quiet_NaN();      /*!< The new expectation value  of energy variance*/
    R                                bondexp_factor = std::numeric_limits<R>::quiet_NaN();      /*!< The mixing factor used in the expansion */
    double                           alpha_mps      = std::numeric_limits<double>::quiet_NaN(); /*!< The mixing factor used in the expansion */
    double                           alpha_h1v      = std::numeric_limits<double>::quiet_NaN(); /*!< The mixing factor for the H¹-term in the expansion */
    double                           alpha_h2v      = std::numeric_limits<double>::quiet_NaN(); /*!< The mixing factor for the H²-term in the expansion */
    std::array<long, 3>              dimM           = {}; /*!< Dimensions of the mps site to be expanded (before expansion) */
    std::array<long, 3>              dimN           = {}; /*!< Dimensions of the mps site to be zero-padded (before expansion) */
    std::array<long, 3>              dimMP          = {}; /*!< Dimensions of the expanded term */
    std::array<long, 3>              dimN0          = {}; /*!< Dimensions of the zero-padded term */
    std::array<long, 3>              dimP1          = {}; /*!< Dimensions of the expansion term corresponding to H¹ */
    std::array<long, 3>              dimP2          = {}; /*!< Dimensions of the expansion term corresponding to H² */
    Eigen::Tensor<T, 3>              mixed_blk;           /*!< The mixed mps block */
    std::string                      msg;
};