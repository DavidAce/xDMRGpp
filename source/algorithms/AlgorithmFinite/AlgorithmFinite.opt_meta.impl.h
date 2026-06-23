#pragma once
#include "../AlgorithmFinite.h"
#include "config/enums/AlgorithmType.h"
#include "config/enums/BlockSizePolicy.h"
#include "config/enums/OptAlgo.h"
#include "config/enums/OptSolver.h"
#include "config/enums/OptType.h"
#include "config/enums/Precision.h"
#include "config/enums/ScalarType.h"
#include "config/settings.h"
#include "math/svd/config.h"
#include "tensors/TensorsFinite.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"
#include "tools/finite/multisite.h"
#include "tools/finite/opt_meta.h"
#include <algorithm>
#include <cmath>

template<typename Scalar>
typename AlgorithmFinite<Scalar>::OptMeta AlgorithmFinite<Scalar>::get_opt_meta() {
    tools::log->trace("get_opt_meta: configuring optimization step");
    OptMeta m1;

    switch(settings::precision::optScalar) {
        case ScalarType::FP32: m1.optType = OptType::FP32; break;
        case ScalarType::FP64: m1.optType = OptType::FP64; break;
        case ScalarType::FP80: m1.optType = OptType::FP80; break;
        case ScalarType::FP128: m1.optType = OptType::FP128; break;
        case ScalarType::CX32: m1.optType = OptType::CX32; break;
        case ScalarType::CX64: m1.optType = OptType::CX64; break;
        case ScalarType::CX80: m1.optType = OptType::CX80; break;
        case ScalarType::CX128: m1.optType = OptType::CX128; break;
    }

    m1.optRitz = status.opt_ritz;
    m1.optAlgo = status.algo_type == AlgorithmType::xDMRG ? settings::xdmrg::algo : OptAlgo::DMRG;
    if(status.algo_type == AlgorithmType::xDMRG) {
        if(status.algorithm_has_stuck_for > 1) {
            m1.optAlgo = settings::xdmrg::algo_stuck;
            m1.optRitz = settings::xdmrg::ritz_stuck;
        }
        if(status.iter < settings::schedule::opt::iter_max_warmup) {
            m1.optAlgo = settings::xdmrg::algo_warmup;
            m1.optRitz = settings::xdmrg::ritz_warmup;
        }
    }

    m1.svd_cfg = svd::config(status.bond_lim, status.trnc_lim);

    m1.min_sites = std::min(tensors.template get_length<size_t>(), settings::schedule::dmrg::min_blocksize);
    m1.max_sites = has_flag(settings::schedule::dmrg::blocksize_policy, BlockSizePolicy::ON_UPDATE) ? std::max(m1.min_sites, dmrg_blocksize) : m1.min_sites;

    m1.max_problem_size = settings::schedule::dmrg::max_prob_size;
    m1.chosen_sites     = tools::finite::multisite::generate_site_list(tensors.get_state(), m1.max_problem_size, m1.max_sites, m1.min_sites, m1.label);
    m1.problem_dims     = tools::finite::multisite::get_dimensions(tensors.get_state(), m1.chosen_sites);
    m1.problem_size     = tools::finite::multisite::get_problem_size(tensors.get_state(), m1.chosen_sites);

    m1.subspace_tol   = settings::solvers::eig::target_subspace_error;
    m1.primme_method  = "PRIMME_DYNAMIC";
    m1.eigs_lib       = enum2sv(settings::solvers::eig::eigslib);
    m1.eigv_target    = 0.0;
    m1.eigs_nev       = settings::solvers::eig::nev_min;
    m1.eigs_ncv       = settings::solvers::eig::ncv_min;
    m1.eigs_blk       = settings::solvers::eig::blk_min;
    if(status.algorithm_has_stuck_for > 0) {
        m1.eigs_nev = settings::solvers::eig::nev_max;
        m1.eigs_ncv = settings::solvers::eig::ncv_max;
        m1.eigs_blk = settings::solvers::eig::blk_max;
    }
    m1.eigs_iter_max = get_eigs_iter_max() * m1.eigs_nev.value();
    m1.eigs_abstol   = dmrg_eigs_abstol;
    m1.eigs_reltol   = dmrg_eigs_reltol;

    m1.eigs_jcbMaxBlockSize = settings::solvers::eig::jcb_blocksize_min;
    m1.eigs_jcbOverlapSize  = settings::solvers::eig::jcb_overlap_size;
    if(status.algorithm_saturated_for + status.algorithm_has_stuck_for > 0) {
        double jcbBlockSizeLog2_20pcnt =
            std::ceil(std::lerp(std::log2(settings::solvers::eig::jcb_blocksize_min), std::log2(settings::solvers::eig::jcb_blocksize_max), 0.20));
        double jcbBlockSizeLog2_50pcnt =
            std::ceil(std::lerp(std::log2(settings::solvers::eig::jcb_blocksize_min), std::log2(settings::solvers::eig::jcb_blocksize_max), 0.50));
        double jcbBlockSizeLog2_80pcnt =
            std::ceil(std::lerp(std::log2(settings::solvers::eig::jcb_blocksize_min), std::log2(settings::solvers::eig::jcb_blocksize_max), 0.80));
        if(status.algorithm_saturated_for > 0) { m1.eigs_jcbMaxBlockSize = static_cast<long>(std::pow(2, jcbBlockSizeLog2_20pcnt)); }
        if(status.algorithm_has_stuck_for > 0) { m1.eigs_jcbMaxBlockSize = static_cast<long>(std::pow(2, jcbBlockSizeLog2_50pcnt)); }
        if(status.algorithm_has_stuck_for > 2) { m1.eigs_jcbMaxBlockSize = static_cast<long>(std::pow(2, jcbBlockSizeLog2_80pcnt)); }
        if(status.algorithm_has_stuck_for > 4) { m1.eigs_jcbMaxBlockSize = settings::solvers::eig::jcb_blocksize_max; }
    }

    m1.optSolver = m1.problem_size <= settings::solvers::eig::max_size ? OptSolver::EIG : OptSolver::EIGS;
    m1.label     = enum2sv(m1.optAlgo);

    if(has_any_flags(m1.optType, OptType::FP80, OptType::CX80, OptType::FP128, OptType::CX128)) m1.eigs_lib = "EIGSMPO";
    m1.validate();
    return m1;
}
