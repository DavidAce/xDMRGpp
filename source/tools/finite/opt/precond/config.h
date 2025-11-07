#pragma once
enum class BasisChangeScale { NONE, MIN, AVG, MAX, SQRTMIN, SQRTAVG, SCALE };
enum class EnvWeightRegularizer { NONE, NORM, MAX, SUM, MEAN };
enum class EnvWeightType {
    ONES,
    NO_PSI_TRACE,
    WITH_PSI_TRACE,
    NO_PSI_SUM,
    WITH_PSI_SUM,
    AB_TRACE,
};

enum class EnvAggregateType {
    PLAIN,
    M1,
    M2,
    M2_inv,
    H2_inv,
    H2_zip,
};

enum class SymmetrizeAggregates { OFF, ON };

/*! How to select the diagonal D in the transform T = U.adjoint() * D * U
 *  This is used after diagonalizing the environment aggregate: env_agg = U * Y * U.adjoint()
 */
enum class TransformSpectrumType {
    EnvAggregateSpectrum, /*!< D = eigenvalues(Y) where Y = Σ_μ w_μ * env_μ,  (note that env_μ = env[:,:,μ])  */
    EnvProjectedDiagonal, /*!< D_i = Σ_μ w_μ * (U.adjoint() * env_μ * U)_{ii}  (μ-slice-wise expectation values in U-basis) */
};

struct BasisChangeConfig {
    float                 alpha = 1;
    EnvWeightType         ewt   = EnvWeightType::AB_TRACE;
    EnvWeightRegularizer  ewr   = EnvWeightRegularizer::NONE;
    EnvAggregateType      eat   = EnvAggregateType::PLAIN;
    SymmetrizeAggregates  sym   = SymmetrizeAggregates::OFF;
    TransformSpectrumType tst   = TransformSpectrumType::EnvProjectedDiagonal;

    float                  scale   = 1;
    BasisChangeScale       bcs     = BasisChangeScale::NONE;
    long                   maxreps = 1; /*!< How many times to repeat the basis change */
    size_t                 num_mv  = 0; /*!< Use to keep track of how many matvecs this config used */
    double                 time_mv = 0; /*!< Use to keep track of uch time was spent in matvecs using this config */
    template<typename T> T get_alpha() const { return static_cast<T>(alpha); }
};
