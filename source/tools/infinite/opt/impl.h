#pragma once
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

using namespace tools::infinite::opt;

template<typename Scalar>
Eigen::Tensor<Scalar, 3> tools::infinite::opt::find_ground_state(const TensorsInfinite<Scalar> &tensors, std::string_view ritzstring) {
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
    solver.config.maxNcv  = settings::precision::eigs_ncv_min;
    solver.config.tol     = settings::precision::eigs_tol_min;
    solver.config.maxIter = 10000;
    solver.eigs(matrix, -1, -1, ritz, eig::Form::SYMM, eig::Side::R, cx64{1.0, 0.0}, eig::Shinv::OFF, eig::Vecs::ON, eig::Dephase::OFF);
    return tenx::asScalarType<Scalar>(eig::view::get_eigvec<cx64>(solver.result, shape_mps));
}


//============================================================================//
// Do unitary evolution on an MPS
//============================================================================//
template<typename Scalar>
Eigen::Tensor<Scalar, 3> tools::infinite::opt::time_evolve_state(const StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 2> &U)
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
