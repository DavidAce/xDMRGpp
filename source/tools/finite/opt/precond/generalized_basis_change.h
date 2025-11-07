#pragma once
#include "common.h"
#include "config.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/TensorsFinite.h"
#include "tools/finite/opt_mps.h"

namespace tools::finite::opt::precond::generalized {

    template<typename Scalar>
    struct GeneralizedBasisChange {
        using RealScalar      = decltype(std::real(std::declval<Scalar>()));
        using MatrixType      = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType      = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using MatrixReal      = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorReal      = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using MapMatType      = Eigen::Map<MatrixType>;
        using MapVecReal      = Eigen::Map<VectorReal>;
        using MapConstMatType = Eigen::Map<const MatrixType>;
        using MapConstVecReal = Eigen::Map<const VectorReal>;
        using MapConstVecType = Eigen::Map<const VectorType>;

        static constexpr RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
        std::vector<size_t>         sites;
        opt_mps<Scalar>             initial_guess;
        EnvEne<Scalar>              bc_enveL, bc_enveR; /*!< The left and right H1 environments.  */
        EnvVar<Scalar>              bc_envvL, bc_envvR; /*!< The left and right H2 environments.  */
        Eigen::Tensor<Scalar, 4>    mpo1;
        Eigen::Tensor<Scalar, 4>    mpo2;
        BasisChangeConfig           bcfg = {};

        env_pair<const EnvEne<Scalar> &> get_enve_pair() const;
        env_pair<const EnvVar<Scalar> &> get_envv_pair() const;

        std::array<Eigen::Index, 3> shape_orig;
        std::array<Eigen::Index, 3> shape_tilde;

        MatrixType   TL, TR;         // Map from original to preconditioned space  U*Y^{-1/2}*UL.adjoint() (U are eigvecs, Y  areeigvals)
        MatrixType   SL, SR;         // Map from preconditioned to original space  U*Y^{1/2}*U.adjoint() (U are eigvecs, Y are eigvals)
        MatrixType   UL, UR;         // Eigenvectors in the map from preconditioned to original space  U*Y^{1/2}*U.adjoint() (U are eigvecs, Y are eigvals)
        RealScalar   kappaL, kappaR; // The rescaling factors multiplied onto the left and right environments
        Eigen::Index pass = 0;

        static bool is_hermitian_tensor(const Eigen::TensorRef<Eigen::Tensor<Scalar, 2>> &A);
        static bool is_hermitian_matrix(const MatrixType &Am);
        static bool is_anti_hermitian_matrix(const MatrixType &Am);

        static void       print_stats(const Eigen::Tensor<Scalar, 1> &w, std::string_view tag);
        static void       symmetrize(Eigen::Tensor<Scalar, 2> &E);
        static void       regularize(Eigen::Tensor<Scalar, 1> &w, const EnvWeightRegularizer ewr, std::string_view tag);
        static MatrixType matrix_norm(const MatrixType &A);

        static std::pair<Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 1>>
            get_env_weights(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR,
                            const Eigen::Tensor<Scalar, 4> &mpo, EnvWeightType ewt, EnvWeightRegularizer ewr);

        std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 2>,
                   Eigen::Tensor<Scalar, 2>>
            get_aggregate_envs(const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR, const Eigen::Tensor<Scalar, 4> &mpo);

        struct Transform_H2_zip {
            Eigen::Tensor<Scalar, 1> w2L, w2R;
            MatrixType               P2L, P2R;
            MatrixType               T2L, T2R;
            MatrixType               S2L, S2R;
            MatrixType               U2L, U2R;
            Eigen::Tensor<Scalar, 2> env2L_agg, env2R_agg;
            Eigen::Tensor<Scalar, 3> env2L_zip, env2R_zip;
        };
        Transform_H2_zip get_generalized_transforms_H2_zip(const Eigen::Tensor<Scalar, 3> &env2L, const Eigen::Tensor<Scalar, 3> &env2R,
                                                           const Eigen::Tensor<Scalar, 4> &mpo2);

        std::tuple<MatrixType, MatrixType, MatrixType, RealScalar>
            get_generalized_transforms(const Eigen::Tensor<Scalar, 3> &env1, const Eigen::Tensor<Scalar, 3> &env2, const Eigen::Tensor<Scalar, 2> &env1_agg,
                                       const Eigen::Tensor<Scalar, 2> &env2_agg, const Eigen::Tensor<Scalar, 1> &w1, const Eigen::Tensor<Scalar, 1> &w2,
                                       const Eigen::Tensor<Scalar, 2> &P1, const Eigen::Tensor<Scalar, 2> P2);

        GeneralizedBasisChange() = default;
        GeneralizedBasisChange(const opt_mps<Scalar>                  &initial, /*!< Initial guess */
                               const Eigen::Tensor<Scalar, 4>         &mpo1,    /*!< Multisite mpo for H1 */
                               const Eigen::Tensor<Scalar, 4>         &mpo2,    /*!< Multisite mpo for H2 */
                               const env_pair<const EnvEne<Scalar> &> &env1,    /*!< Multisite env for H1 */
                               const env_pair<const EnvVar<Scalar> &> &env2,    /*!< Multisite env for H2 */
                               BasisChangeConfig                       bcfg_);

        GeneralizedBasisChange(const opt_mps<Scalar> &initial, const TensorsFinite<Scalar> &tensors, BasisChangeConfig bcfg_);

        GeneralizedBasisChange(const GeneralizedBasisChange &bc);
        GeneralizedBasisChange(const GeneralizedBasisChange &bc, BasisChangeConfig bcfg_);
    };

}
