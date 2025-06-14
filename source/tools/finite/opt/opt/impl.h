#pragma once
#include "algorithms/AlgorithmStatus.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "math/tenx.h"
#include "math/tenx/threads.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
#include <string>

//
#include "math/eig.h"
#include <h5pp/details/h5ppEigen.h>
#include <tensors/edges/EdgesFinite.h>

template<typename Scalar>
tools::finite::opt::opt_mps<Scalar> tools::finite::opt::get_opt_initial_mps(const TensorsFinite<Scalar> &tensors, const OptMeta &meta) {
    auto            t_init = tid::tic_scope("initial_mps", tid::level::higher);
    opt_mps<Scalar> initial_mps;
    initial_mps.set_name("initial_mps");
    initial_mps.set_sites(tensors.active_sites);
    initial_mps.set_length(tensors.template get_length<size_t>());
    initial_mps.set_tensor(tensors.template get_multisite_mps<Scalar>());
    initial_mps.set_energy(tools::finite::measure::energy(tensors));
    initial_mps.set_eshift(tools::finite::measure::energy_shift(tensors));
    initial_mps.set_hsquared(std::real(tools::finite::measure::expval_hamiltonian_squared(tensors)));
    initial_mps.set_variance(tools::finite::measure::energy_variance(tensors));
    initial_mps.set_rnorm_H1(tools::finite::measure::residual_norm_H1(tensors));
    initial_mps.set_rnorm_H2(tools::finite::measure::residual_norm_H2(tensors));
    initial_mps.set_overlap(1.0);

    switch(meta.optAlgo) {
        case OptAlgo::DMRG:
        case OptAlgo::DMRGX:
        case OptAlgo::HYBRID_DMRGX: {
            auto H1 = tools::finite::measure::expval_hamiltonian(tensors);
            initial_mps.set_eigs_eigval(std::real(H1));
            break;
        }
        case OptAlgo::XDMRG: {
            // (H-Eshift)v =  <H²> v
            auto H2 = tools::finite::measure::expval_hamiltonian_squared(tensors);
            initial_mps.set_eigs_eigval(std::real(H2));
            break;
        }
        case OptAlgo::GDMRG: {
            // (H-Eshift)v =  <H¹>/<H²> (H-Eshift)²v
            auto H1 = tools::finite::measure::expval_hamiltonian(tensors);         // <H>
            auto H2 = tools::finite::measure::expval_hamiltonian_squared(tensors); // <H²>
            initial_mps.set_eigs_eigval(std::real(H1) / std::real(H2));
            break;
        }
    }
    initial_mps.validate_initial_mps();
    tensors.clear_cache();
    tensors.clear_measurements();
    return initial_mps;
}

template<typename Scalar>
tools::finite::opt::opt_mps<Scalar> tools::finite::opt::find_ground_state(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                                                          const AlgorithmStatus &status, OptMeta &meta) {
    auto t_opt  = tid::tic_scope("opt");
    auto elog   = reports::eigs_log<Scalar>();
    auto result = internal::optimize_energy(tensors, initial_mps, status, meta, elog);
    elog.print_eigs_report();
    // Finish up
    result.set_optsolver(meta.optSolver);
    result.set_optalgo(meta.optAlgo);
    result.set_opttype(meta.optType);
    result.set_optexit(meta.optExit);
    result.validate_result();
    return result;
}

template<typename Scalar>
tools::finite::opt::opt_mps<Scalar> tools::finite::opt::get_updated_state(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                                                          const AlgorithmStatus &status, OptMeta &meta) {
    auto t_opt = tid::tic_scope("opt");
    tools::log->trace("Starting optimization: algo [{}] | solver [{}] | type [{}] | ritz [{}] | position [{}] | sites {} | shape {} = {}",
                      enum2sv(meta.optAlgo), enum2sv(meta.optSolver), enum2sv(meta.optType), enum2sv(meta.optRitz), status.position, tensors.active_sites,
                      tensors.active_problem_dims(), tensors.active_problem_size());

    using namespace opt::internal;

    if(initial_mps.get_sites() != tensors.active_sites)
        throw except::runtime_error("mismatch in active sites: initial_mps {} | active {}", initial_mps.get_sites(), tensors.active_sites);

    opt_mps<Scalar>           result;
    reports::eigs_log<Scalar> elog;
    reports::subs_log<Scalar> slog;
    // Dispatch optimization to the correct routine depending on the chosen algorithm
    if(meta.optSolver == OptSolver::H1H2) {
        result = internal::optimize_lanczos_h1h2(tensors, initial_mps, status, meta, elog);
        // auto meta2      = meta;
        // meta2.optSolver = OptSolver::EIGS;
        // meta2.optAlgo   = OptAlgo::XDMRG;
        // meta2.optRitz   = OptRitz::SM;
        // auto result2    = internal::optimize_lanczos_h1h2(tensors, initial_mps, status, meta2, elog);
        // auto meta3      = meta;
        // meta3.optSolver = meta.optSolver == OptSolver::EIGS ? OptSolver::H1H2 : OptSolver::EIGS;
        // meta3.optType   = OptType::FP64;
        // meta3.optAlgo   = OptAlgo::GDMRG;
        // meta3.optRitz   = OptRitz::LM;
        // if(meta.optSolver != meta3.optSolver or meta.optType != meta3.optType or meta.optAlgo != meta3.optAlgo or meta.optRitz != meta3.optRitz) {
        //     if(meta3.optSolver == OptSolver::EIGS) {
        //         auto result3 = internal::optimize_generalized_shift_invert(tensors, initial_mps, status, meta3, elog);
        //     } else {
        //         auto result3 = internal::optimize_lanczos_h1h2(tensors, initial_mps, status, meta3, elog);
        //     }
        // }
    } else {
        switch(meta.optAlgo) {
            case OptAlgo::DMRG: {
                result = internal::optimize_energy(tensors, initial_mps, status, meta, elog);
                break;
            }
            case OptAlgo::DMRGX: {
                result = internal::optimize_overlap(tensors, initial_mps, status, meta, slog);
                break;
            }
            case OptAlgo::HYBRID_DMRGX: {
                result = internal::optimize_subspace_variance(tensors, initial_mps, status, meta, elog);
                break;
            }
            case OptAlgo::XDMRG: {
                result = internal::optimize_folded_spectrum(tensors, initial_mps, status, meta, elog);
                break;
            }
            case OptAlgo::GDMRG: {
                result = internal::optimize_generalized_shift_invert(tensors, initial_mps, status, meta, elog);
                break;
            }
        }
    }


    slog.print_subs_report();
    elog.print_eigs_report();

    // Finish up
    result.set_optsolver(meta.optSolver);
    result.set_optalgo(meta.optAlgo);
    result.set_opttype(meta.optType);
    result.set_optexit(meta.optExit);
    result.validate_result();
    return result;
}

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

template<typename Scalar>
bool comparator::energy(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    // The eigenvalue solver on H gives results sorted in energy
    if(lhs.get_eigs_idx() != rhs.get_eigs_idx()) return lhs.get_eigs_idx() < rhs.get_eigs_idx();
    return lhs.get_energy() < rhs.get_energy();
}

template<typename Scalar>
bool comparator::energy_absolute(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    return std::abs(lhs.get_energy()) < std::abs(rhs.get_energy());
}

template<typename Scalar>
bool comparator::energy_distance(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs, RealScalar<Scalar> target) {
    if(std::isnan(target)) throw except::logic_error("Energy target for comparison is NAN");
    auto diff_energy_lhs = std::abs(lhs.get_energy() - target);
    auto diff_energy_rhs = std::abs(rhs.get_energy() - target);
    return diff_energy_lhs < diff_energy_rhs;
}

template<typename Scalar>
bool comparator::variance(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    // The eigenvalue solver on (H-E)² gives results sorted in variance
    return lhs.get_variance() < rhs.get_variance();
}

template<typename Scalar>
bool comparator::gradient(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    return lhs.get_grad_max() < rhs.get_grad_max();
}
template<typename Scalar>
bool comparator::eigval(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    auto leig     = lhs.get_eigs_eigval();
    auto reig     = rhs.get_eigs_eigval();
    auto lvar     = lhs.get_variance();
    auto rvar     = rhs.get_variance();
    auto diff_eig = std::abs(leig - reig);
    auto diff_var = std::abs(lvar - rvar);
    auto same_eig = std::abs(diff_eig) < std::numeric_limits<RealScalar<Scalar>>::epsilon() * 10;
    auto same_var = std::abs(diff_var) < std::numeric_limits<RealScalar<Scalar>>::epsilon() * 10;
    if(same_eig and same_var) {
        // There is no clear winner. We should therefore stick to comparing overlap for the sake of stability.
        auto has_overlap = lhs.get_overlap() + rhs.get_overlap() > std::sqrt(RealScalar<Scalar>{2});
        if(has_overlap) {
            // Favor comparing overlaps for stability when everything else is too similar. This requires that one or both sufficient overlap to begin with.
            tools::log->warn("comparator::eigval: degeneracy detected -- comparing overlap");
            return lhs.get_overlap() > rhs.get_overlap(); // When degenarate, follow overlap
        }
    }
    if(same_eig) {
        tools::log->warn("comparator::eigval: degeneracy detected -- comparing variance");
        // The eigvals are too close... compare rnorms instead
        return lvar < rvar;
    }
    return leig < reig;
}
template<typename Scalar>
bool comparator::eigval_absolute(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    auto leig     = std::abs(lhs.get_eigs_eigval());
    auto reig     = std::abs(rhs.get_eigs_eigval());
    auto lvar     = lhs.get_variance();
    auto rvar     = rhs.get_variance();
    auto diff_eig = std::abs(leig - reig);
    auto diff_var = std::abs(lvar - rvar);
    auto same_eig = std::abs(diff_eig) < std::numeric_limits<RealScalar<Scalar>>::epsilon() * 10;
    auto same_var = std::abs(diff_var) < std::numeric_limits<RealScalar<Scalar>>::epsilon() * 10;
    if(same_eig and same_var) {
        // There is no clear winner. We should therefore stick to comparing overlap for the sake of stability.
        auto has_overlap = lhs.get_overlap() + rhs.get_overlap() > std::sqrt(RealScalar<Scalar>{2});
        if(has_overlap) {
            // Favor comparing overlaps for stability when everything else is too similar. This requires that one or both sufficient overlap to begin with.
            tools::log->warn("comparator::eigval: degeneracy detected -- comparing overlap");
            return lhs.get_overlap() > rhs.get_overlap(); // When degenarate, follow overlap
        }
    }
    if(same_eig) {
        tools::log->warn("comparator::eigval: degeneracy detected -- comparing variance");
        // The eigvals are too close... compare rnorms instead
        return lvar < rvar;
    }
    return leig < reig;
}

template<typename Scalar>
bool comparator::overlap(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    return lhs.get_overlap() > rhs.get_overlap();
}
template<typename Scalar>
bool comparator::eigval_and_overlap(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    auto ratio = std::max(lhs.get_eigs_eigval(), rhs.get_eigs_eigval()) / std::min(lhs.get_eigs_eigval(), rhs.get_eigs_eigval());
    if(ratio < RealScalar<Scalar>{10} and lhs.get_overlap() >= std::sqrt(RealScalar<Scalar>{0.5})) return comparator::overlap(lhs, rhs);
    return comparator::eigval(lhs, rhs);
}

template<typename Scalar>
Comparator<Scalar>::Comparator(const OptMeta &meta_, RealScalar<Scalar> target_energy_) : meta(&meta_), target_energy(target_energy_) {}

template<typename Scalar>
bool Comparator<Scalar>::operator()(const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
    if(not meta) throw except::logic_error("No opt_meta given to comparator");
    switch(meta->optRitz) {
        case OptRitz::SR: return comparator::eigval(lhs, rhs);
        case OptRitz::SM: return comparator::eigval_absolute(lhs, rhs);
        case OptRitz::LR: return comparator::eigval(rhs, lhs);
        case OptRitz::LM: return comparator::eigval_absolute(rhs, lhs);
        case OptRitz::IS: return comparator::energy_distance(lhs, rhs, target_energy);
        case OptRitz::TE: return comparator::energy_distance(lhs, rhs, target_energy);
        default: return comparator::eigval(lhs, rhs);
    }
    return comparator::eigval(lhs, rhs);
}

template<typename Scalar>
EigIdxComparator<Scalar>::EigIdxComparator(OptRitz ritz_, Scalar shift_, Scalar *data_, long size_) : ritz(ritz_), shift(shift_), eigvals(data_, size_) {}
template<typename Scalar>
bool EigIdxComparator<Scalar>::operator()(long lidx, long ridx) {
    auto lhs = eigvals[lidx];
    auto rhs = eigvals[ridx];
    switch(ritz) {
        case OptRitz::NONE: throw std::logic_error("EigvalComparator: Invalid OptRitz::NONE");
        case OptRitz::SR: return std::real(lhs) < std::real(rhs);
        case OptRitz::LR: return std::real(rhs) < std::real(lhs);
        case OptRitz::SM: return std::abs(lhs) < std::abs(rhs);
        case OptRitz::LM: return std::abs(lhs) > std::abs(rhs);
        case OptRitz::IS:
        case OptRitz::TE: {
            auto diff_lhs = std::abs(lhs - shift);
            auto diff_rhs = std::abs(rhs - shift);
            return diff_lhs < diff_rhs;
        }
        default: throw std::logic_error("EigvalComparator: Invalid OptRitz");
    }
}
