#include "../opt_meta.h"
#include "config/enums.h"
#include "debug/exceptions.h"
#include <fmt/ranges.h>

namespace tools::finite::opt {
    OptMeta::OptMeta()
        : optAlgo(OptAlgo::DMRG), optRitz(OptRitz::SR), optSolver(OptSolver::EIGS), optType(OptType::CX64), optWhen(OptWhen::ALWAYS), optExit(OptExit::NONE),
          bondexp_policy(BondExpansionPolicy::DEFAULT) {}

    OptMeta::OptMeta(OptAlgo algo, OptRitz ritz) : OptMeta() {
        optAlgo = algo;
        optRitz = ritz;
    }

    bool OptMeta::should_proceed(OptExit previous_exit) const {
        switch(optWhen) {
            case OptWhen::ALWAYS: return true;
            case OptWhen::NEVER: return false;
            default: {
                if(previous_exit == OptExit::SUCCESS)
                    return false;
                else
                    return have_common(optWhen, previous_exit);
            }
        }
    }
    void OptMeta::validate() const {
        //        if(optCost == OptCost::OVERLAP and optSolver == OptSolver::BFGS) throw except::runtime_error("opt: mode [OVERLAP] and solver [BFGS] are
        //        incompatible"); if(optCost == OptCost::SUBSPACE and optSolver == OptSolver::BFGS) throw except::runtime_error("opt: mode [ENERGY] and solver
        //        [BFGS] are incompatible"); if(optCost == OptCost::ENERGY and optSolver == OptSolver::BFGS) throw except::runtime_error("opt: mode [ENERGY] and
        //        solver [BFGS] are incompatible");
    }

    std::string OptMeta::string() const {
        std::string res;
        res += label;
        res += fmt::format(" | site {}", chosen_sites);
        res += fmt::format(" | dims {} = {}", problem_dims, problem_size);
        res += fmt::format(" | solv {}", enum2sv(optSolver));
        res += fmt::format(" | type {}", enum2sv(optType));
        res += fmt::format(" | ritz {}", enum2sv(optRitz));
        res += fmt::format(" | algo {}", enum2sv(optAlgo));
        res += fmt::format(" | expm {}", flag2str(bondexp_policy));
        if(eigv_target) res += fmt::format(" (tgt: {:.3e})", eigv_target.value());
        if(eigs_nev) res += fmt::format(" | nev {}", eigs_nev.value());
        if(eigs_ncv) res += fmt::format(" | ncv {}", eigs_ncv.value());
        // res += fmt::format(" | env.exp. α {:.3e}", alpha_expansion ? alpha_expansion.value() : std::numeric_limits<double>::quiet_NaN());
        if(svd_cfg) res += fmt::format(" | svd_ε {:.2e}", svd_cfg->truncation_limit.value_or(std::numeric_limits<double>::quiet_NaN()));
        return res;
    }

}
