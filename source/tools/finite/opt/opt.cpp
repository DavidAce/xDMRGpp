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
#include "tools/finite/measure.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
#include <string>

//
#include "math/eig.h"
#include <h5pp/details/h5ppEigen.h>
#include <tensors/edges/EdgesFinite.h>

tools::finite::opt::opt_mps tools::finite::opt::get_opt_initial_mps(const TensorsFinite &tensors, const OptMeta &meta) {
    auto    t_init = tid::tic_scope("initial_mps", tid::level::higher);
    opt_mps initial_mps;
    initial_mps.set_name("initial_mps");
    initial_mps.set_sites(tensors.active_sites);
    initial_mps.set_length(tensors.get_length<size_t>());
    initial_mps.set_tensor(tensors.get_multisite_mps<cx64>());
    initial_mps.set_energy(tools::finite::measure::energy(tensors));
    initial_mps.set_eshift(tools::finite::measure::energy_shift(tensors));
    initial_mps.set_hsquared(std::real(tools::finite::measure::expval_hamiltonian_squared<cx64>(tensors)));
    initial_mps.set_variance(tools::finite::measure::energy_variance(tensors));
    initial_mps.set_rnorm_H1(tools::finite::measure::residual_norm_H1<cx64>(tensors));
    initial_mps.set_rnorm_H2(tools::finite::measure::residual_norm_H2<cx64>(tensors));
    initial_mps.set_overlap(1.0);

    switch(meta.optAlgo) {
        case OptAlgo::DMRG:
        case OptAlgo::DMRGX:
        case OptAlgo::HYBRID_DMRGX: {
            auto H1 = tools::finite::measure::expval_hamiltonian<cx64>(tensors);
            initial_mps.set_eigs_eigval(std::real(H1));
            break;
        }
        case OptAlgo::XDMRG: {
            // (H-Eshift)v =  <H²> v
            auto H2 = tools::finite::measure::expval_hamiltonian_squared<cx64>(tensors);
            initial_mps.set_eigs_eigval(std::real(H2));
            break;
        }
        case OptAlgo::GDMRG: {
            // (H-Eshift)v =  <H¹>/<H²> (H-Eshift)²v
            auto H1 = tools::finite::measure::expval_hamiltonian<cx64>(tensors);         // <H>
            auto H2 = tools::finite::measure::expval_hamiltonian_squared<cx64>(tensors); // <H²>
            initial_mps.set_eigs_eigval(std::real(H1) / std::real(H2));
            break;
        }
    }

    initial_mps.validate_initial_mps();
    tensors.clear_cache();
    tensors.clear_measurements();
    return initial_mps;
}

tools::finite::opt::opt_mps tools::finite::opt::find_ground_state(const TensorsFinite &tensors, const opt_mps &initial_mps, const AlgorithmStatus &status,
                                                                  OptMeta &meta) {
    auto t_opt  = tid::tic_scope("opt");
    auto result = internal::optimize_energy(tensors, initial_mps, status, meta);
    tools::finite::opt::reports::print_eigs_report();
    // Finish up
    result.set_optsolver(meta.optSolver);
    result.set_optalgo(meta.optAlgo);
    result.set_optexit(meta.optExit);
    result.validate_result();
    return result;
}

tools::finite::opt::opt_mps tools::finite::opt::get_updated_state(const TensorsFinite &tensors, const opt_mps &initial_mps, const AlgorithmStatus &status,
                                                                  OptMeta &meta) {
    auto t_opt = tid::tic_scope("opt");
    tools::log->trace("Starting optimization: algo [{}] | solver [{}] | type [{}] | ritz [{}] | position [{}] | sites {} | shape {} = {}",
                      enum2sv(meta.optAlgo), enum2sv(meta.optSolver), enum2sv(meta.optType), enum2sv(meta.optRitz), status.position, tensors.active_sites,
                      tensors.active_problem_dims(), tensors.active_problem_size());

    using namespace opt::internal;

    if(initial_mps.get_sites() != tensors.active_sites)
        throw except::runtime_error("mismatch in active sites: initial_mps {} | active {}", initial_mps.get_sites(), tensors.active_sites);

    opt_mps result;
    // Dispatch optimization to the correct routine depending on the chosen algorithm
    switch(meta.optAlgo) {
        case OptAlgo::DMRG: {
            result = internal::optimize_energy(tensors, initial_mps, status, meta);
            break;
        }
        case OptAlgo::DMRGX: {
            result = internal::optimize_overlap(tensors, initial_mps, status, meta);
            break;
        }
        case OptAlgo::HYBRID_DMRGX: {
            result = internal::optimize_subspace_variance(tensors, initial_mps, status, meta);
            break;
        }
        case OptAlgo::XDMRG: {
            result = internal::optimize_folded_spectrum(tensors, initial_mps, status, meta);
            break;
        }
        case OptAlgo::GDMRG: {
            result = internal::optimize_generalized_shift_invert(tensors, initial_mps, status, meta);
            break;
        }
    }

    tools::finite::opt::reports::print_eigs_report();

    // Finish up
    result.set_optsolver(meta.optSolver);
    result.set_optalgo(meta.optAlgo);
    result.set_optexit(meta.optExit);
    result.validate_result();
    return result;
}

double tools::finite::opt::internal::windowed_func_abs(double x, double window) {
    if(std::abs(x) >= window)
        return std::abs(x) - window;
    else
        return 0;
}
double tools::finite::opt::internal::windowed_grad_abs(double x, double window) {
    if(std::abs(x) >= window)
        return num::sign(x);
    else
        return 0.0;
}

double tools::finite::opt::internal::windowed_func_pow(double x, double window) {
    if(std::abs(x) >= window)
        return x * x - window * window;
    else
        return 0.0;
}
double tools::finite::opt::internal::windowed_grad_pow(double x, double window) {
    if(std::abs(x) >= window)
        return 2.0 * x;
    else
        return 0.0;
}

std::pair<double, double> tools::finite::opt::internal::windowed_func_grad(double x, double window) {
    double func = 0;
    double grad = 0;
    if(std::abs(x) >= window) {
        if(x > 0) {
            func = (x - window) * (x - window);
            grad = 2 * (x - window);
        } else {
            func = (x + window) * (x + window);
            grad = 2 * (x + window);
        }
    }

    return std::make_pair(func, grad);
}

long tools::finite::opt::internal::get_ops(long d, long chiL, long chiR, long m) {
    if(chiR > chiL) return get_ops(d, chiR, chiL, m);
    if(d > m) {
        // d first
        long step1 = chiL * chiL * chiR * m * m * d;
        long step2 = chiL * chiR * d * m * m * m * (d * m + 1);
        long step3 = chiL * chiR * d * m * (chiR * m * m * m + m * m + 1);
        return step1 + step2 + step3;
    } else {
        // m first
        long step1 = chiL * chiL * chiR * m * d;
        long step2 = chiL * chiR * d * m * m * (d * m + 1);
        long step3 = chiL * chiR * chiR * d * m * (m + 1);
        return step1 + step2 + step3;
    }
}

namespace tools::finite::opt::internal {
    bool comparator::energy(const opt_mps &lhs, const opt_mps &rhs) {
        // The eigenvalue solver on H gives results sorted in energy
        if(lhs.get_eigs_idx() != rhs.get_eigs_idx()) return lhs.get_eigs_idx() < rhs.get_eigs_idx();
        return lhs.get_energy() < rhs.get_energy();
    }

    bool comparator::energy_absolute(const opt_mps &lhs, const opt_mps &rhs) { return std::abs(lhs.get_energy()) < std::abs(rhs.get_energy()); }

    bool comparator::energy_distance(const opt_mps &lhs, const opt_mps &rhs, double target) {
        if(std::isnan(target)) throw except::logic_error("Energy target for comparison is NAN");
        auto diff_energy_lhs = std::abs(lhs.get_energy() - target);
        auto diff_energy_rhs = std::abs(rhs.get_energy() - target);
        return diff_energy_lhs < diff_energy_rhs;
    }

    bool comparator::variance(const opt_mps &lhs, const opt_mps &rhs) {
        // The eigenvalue solver on (H-E)² gives results sorted in variance
        return lhs.get_variance() < rhs.get_variance();
    }

    bool comparator::gradient(const opt_mps &lhs, const opt_mps &rhs) { return lhs.get_grad_max() < rhs.get_grad_max(); }
    bool comparator::eigval(const opt_mps &lhs, const opt_mps &rhs) {
        auto leig     = lhs.get_eigs_eigval();
        auto reig     = rhs.get_eigs_eigval();
        auto lvar     = lhs.get_variance();
        auto rvar     = rhs.get_variance();
        auto diff_eig = std::abs(leig - reig);
        auto diff_var = std::abs(lvar - rvar);
        auto same_eig = std::abs(diff_eig) < 10 * std::numeric_limits<double>::epsilon();
        auto same_var = std::abs(diff_var) < 10 * std::numeric_limits<double>::epsilon();
        if(same_eig and same_var) {
            // There is no clear winner. We should therefore stick to comparing overlap for the sake of stability.
            auto has_overlap = lhs.get_overlap() + rhs.get_overlap() > std::sqrt(2);
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
    bool comparator::eigval_absolute(const opt_mps &lhs, const opt_mps &rhs) {
        auto leig     = std::abs(lhs.get_eigs_eigval());
        auto reig     = std::abs(rhs.get_eigs_eigval());
        auto lvar     = lhs.get_variance();
        auto rvar     = rhs.get_variance();
        auto diff_eig = std::abs(leig - reig);
        auto diff_var = std::abs(lvar - rvar);
        auto same_eig = std::abs(diff_eig) < 10 * std::numeric_limits<double>::epsilon();
        auto same_var = std::abs(diff_var) < 10 * std::numeric_limits<double>::epsilon();
        if(same_eig and same_var) {
            // There is no clear winner. We should therefore stick to comparing overlap for the sake of stability.
            auto has_overlap = lhs.get_overlap() + rhs.get_overlap() > std::sqrt(2);
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

    bool comparator::overlap(const opt_mps &lhs, const opt_mps &rhs) { return lhs.get_overlap() > rhs.get_overlap(); }

    bool comparator::eigval_and_overlap(const opt_mps &lhs, const opt_mps &rhs) {
        double ratio = std::max(lhs.get_eigs_eigval(), rhs.get_eigs_eigval()) / std::min(lhs.get_eigs_eigval(), rhs.get_eigs_eigval());
        if(ratio < 10 and lhs.get_overlap() >= std::sqrt(0.5)) return comparator::overlap(lhs, rhs);
        return comparator::eigval(lhs, rhs);
    }

    Comparator::Comparator(const OptMeta &meta_, double target_energy_) : meta(&meta_), target_energy(target_energy_) {}

    bool Comparator::operator()(const opt_mps &lhs, const opt_mps &rhs) {
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

    EigIdxComparator::EigIdxComparator(OptRitz ritz_, double shift_, double *data_, long size_) : ritz(ritz_), shift(shift_), eigvals(data_, size_) {}
    bool EigIdxComparator::operator()(long lidx, long ridx) {
        auto lhs = eigvals[lidx];
        auto rhs = eigvals[ridx];
        switch(ritz) {
            case OptRitz::NONE: throw std::logic_error("EigvalComparator: Invalid OptRitz::NONE");
            case OptRitz::SR: return lhs < rhs;
            case OptRitz::LR: return rhs < lhs;
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
}
