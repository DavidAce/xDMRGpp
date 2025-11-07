#pragma once
#include "config.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/TensorsFinite.h"
#include "tools/finite/opt_mps.h"

namespace tools::finite::opt::precond::standard {
    template<typename Scalar> using Real           = decltype(std::real(std::declval<Scalar>()));
    template<typename Scalar> using Mat            = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename Scalar> using Vec            = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar, auto rank> using Ten = Eigen::Tensor<Scalar, rank>;

    template<typename Scalar>
    struct BasisChange {
        using RealScalar      = decltype(std::real(std::declval<Scalar>()));
        using MatrixType      = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType      = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using MatrixReal      = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorReal      = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using MapMatType      = Eigen::Map<MatrixType>;
        using MapVecReal      = Eigen::Map<VectorReal>;
        using MapConstMatType = Eigen::Map<const MatrixType>;
        using MapConstVecReal = Eigen::Map<const VectorReal>;

        opt_mps<Scalar> initial_guess;
        EnvEne<Scalar>  bc_enveL, bc_enveR; /*!< The left and right H1 environments.  */
        EnvVar<Scalar>  bc_envvL, bc_envvR; /*!< The left and right H2 environments.  */

        std::vector<MpoSite<Scalar>> bc_mpos;

        env_pair<const EnvEne<Scalar> &> get_enve_pair() const;
        env_pair<const EnvVar<Scalar> &> get_envv_pair() const;

        std::array<Eigen::Index, 3> shape_orig;
        std::array<Eigen::Index, 3> shape_tilde;
        RealScalar                  scale;
        RealScalar                  alpha;
        MatrixType                  TL, TR;         // Map from original to preconditioned space  U*Y^{-1/2}*UL.adjoint() (U are eigvecs, Y  areeigvals)
        MatrixType                  SL, SR;         // Map from preconditioned to original space  U*Y^{1/2}*U.adjoint() (U are eigvecs, Y are eigvals)
        RealScalar                  kappaL, kappaR; // The rescaling factors multiplied onto the left and right environments

        BasisChange(const opt_mps<Scalar> &initial, const TensorsFinite<Scalar> &tensors, BasisChangeScale bcs, RealScalar scale_, RealScalar alpha_);
    };

}