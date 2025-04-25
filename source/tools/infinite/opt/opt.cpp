#include "../opt.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpo.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateInfinite.h"
#include "tensors/TensorsInfinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"

namespace tools::infinite::opt {
    template<typename Scalar>
    Eigen::Tensor<Scalar, 3> find_ground_state(const TensorsInfinite<Scalar> &tensors, std::string_view ritzstring) {
        tools::log->trace("Starting ground state optimization");
        auto t_opt = tid::tic_scope("opt");
        if constexpr(std::is_same_v<Scalar, cx128>) { tools::log->warn("find_ground_state<cx128> is not implemented: Defaulting to cx64"); }
        eig::Ritz ritz = eig::stringToRitz(ritzstring);

        auto        shape_mps = tensors.state->dimensions();
        const auto &mpo       = tensors.model->get_2site_mpo_AB();
        const auto &env       = tensors.edges->get_env_ene_blk();

        MatVecMPO<cx64> matrix(tenx::asScalarType<cx64>(env.L), tenx::asScalarType<cx64>(env.R), tenx::asScalarType<cx64>(mpo));
        eig::solver     solver;
        solver.config.maxNev  = 1;
        solver.config.maxNcv  = settings::precision::eigs_ncv;
        solver.config.tol     = settings::precision::eigs_tol_min;
        solver.config.maxIter = 10000;
        solver.eigs(matrix, -1, -1, ritz, eig::Form::SYMM, eig::Side::R, cx64{1.0, 0.0}, eig::Shinv::OFF, eig::Vecs::ON, eig::Dephase::OFF);
        return tenx::asScalarType<Scalar>(eig::view::get_eigvec<cx64>(solver.result, shape_mps));
    }
    template Eigen::Tensor<fp32, 3>  find_ground_state(const TensorsInfinite<fp32> &tensors, std::string_view ritzstring);
    template Eigen::Tensor<fp64, 3>  find_ground_state(const TensorsInfinite<fp64> &tensors, std::string_view ritzstring);
    template Eigen::Tensor<fp128, 3> find_ground_state(const TensorsInfinite<fp128> &tensors, std::string_view ritzstring);
    template Eigen::Tensor<cx32, 3>  find_ground_state(const TensorsInfinite<cx32> &tensors, std::string_view ritzstring);
    template Eigen::Tensor<cx64, 3>  find_ground_state(const TensorsInfinite<cx64> &tensors, std::string_view ritzstring);
    template Eigen::Tensor<cx128, 3> find_ground_state(const TensorsInfinite<cx128> &tensors, std::string_view ritzstring);
    //============================================================================//
    // Do unitary evolution on an MPS
    //============================================================================//
    template<typename Scalar>
    Eigen::Tensor<Scalar, 3> time_evolve_state(const StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 2> &U)
    /*!
    @verbatim
      1--[ mps ]--2
            |
            0
                             1--[ mps ]--2
            0         --->         |
            |                      0
          [ U ]
            |
            1
    @endverbatim
    */
    {
        return U.contract(state.get_2site_mps(), tenx::idx({0}, {0}));
    }
    template Eigen::Tensor<fp32, 3>  time_evolve_state(const StateInfinite<fp32> &state, const Eigen::Tensor<fp32, 2> &U);
    template Eigen::Tensor<fp64, 3>  time_evolve_state(const StateInfinite<fp64> &state, const Eigen::Tensor<fp64, 2> &U);
    template Eigen::Tensor<fp128, 3> time_evolve_state(const StateInfinite<fp128> &state, const Eigen::Tensor<fp128, 2> &U);
    template Eigen::Tensor<cx32, 3>  time_evolve_state(const StateInfinite<cx32> &state, const Eigen::Tensor<cx32, 2> &U);
    template Eigen::Tensor<cx64, 3>  time_evolve_state(const StateInfinite<cx64> &state, const Eigen::Tensor<cx64, 2> &U);
    template Eigen::Tensor<cx128, 3> time_evolve_state(const StateInfinite<cx128> &state, const Eigen::Tensor<cx128, 2> &U);
}
