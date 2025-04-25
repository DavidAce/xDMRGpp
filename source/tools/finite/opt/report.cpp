
#include "report.h"
#include "../opt_mps.h"
#include "general/iter.h"
#include "tid/tid.h"
#include "tools/common/log.h"

template class tools::finite::opt::reports::subs_log<fp32>;
template class tools::finite::opt::reports::subs_log<fp64>;
template class tools::finite::opt::reports::subs_log<fp128>;
template class tools::finite::opt::reports::subs_log<cx32>;
template class tools::finite::opt::reports::subs_log<cx64>;
template class tools::finite::opt::reports::subs_log<cx128>;

template class tools::finite::opt::reports::eigs_log<fp32>;
template class tools::finite::opt::reports::eigs_log<fp64>;
template class tools::finite::opt::reports::eigs_log<fp128>;
template class tools::finite::opt::reports::eigs_log<cx32>;
template class tools::finite::opt::reports::eigs_log<cx64>;
template class tools::finite::opt::reports::eigs_log<cx128>;

template<typename Scalar>
void tools::finite::opt::reports::subs_log<Scalar>::print_subs_report() {
    if(tools::log->level() > spdlog::level::debug) return;
    if(entries.empty()) return;
    /* clang-format off */
    tools::log->debug("- {:<5} {:<18} {:<18} {:<18} {:<11} {:<11} {:<11} {:<6} {:<6} {:<6}",
                       "nev",
                       "max <φ_i|ψ>",
                       "min <φ_i|ψ>",
                       "ε:(1-Σ|<φ_i|ψ>|²)",  // Special characters are counted properly in spdlog 1.7.0
                       "eig time[s]",
                       "ham time[s]",
                       "lu Time[s]",
                       "iter",
                       "mv",
                       "pc");

    for(auto &entry : entries){
        tools::log->debug("- {:<5} {:<18.16f} {:<18.16f} {:<18.2e} {:<11.2e} {:<11.2e} {:<11.2e} {:<6} {:<6} {:<6}",
                          entry.nev,
                          fp(entry.max_olap),
                          fp(entry.min_olap),
                          fp(entry.eps) ,
                          entry.eig_time,
                          entry.ham_time,
                          entry.lu_time ,
                          entry.iter,
                          entry.mv,
                          entry.pc
                          );
    }
    /* clang-format on */
    entries.clear();
}

template<typename Scalar>
void tools::finite::opt::reports::eigs_log<Scalar>::print_eigs_report(std::optional<size_t> max_entries) {
    if(entries.empty()) return;
    auto level = entries.front().level;
    if(level < tools::log->level()) {
        entries.clear();
        return;
    }
    /* clang-format off */
    tools::log->log(level, "{:<36} {:<7} {:<4} {:<4} {:<4} {:<4} {:<8} {:<22} {:<22} {:<10} {:<22} {:<18} {:<18} {:<8} {:<8} {:<9} {:<5} {:<7} {:<7} {:<10} {:<10} {:<10}",
                      "Optimization report",
                      "size",
                      "ritz",
                      "idx",
                      "nev",
                      "ncv",
                      "tol",
                      "⟨H⟩",
                      "⟨H²⟩",
                      "⟨H²⟩-⟨H⟩²", // Special characters are counted properly in fmt 1.7.0
                      "λ",
                      "overlap",
                      "norm",
                      "rnorm",
                      "|Hv-Ev|",
                      "|H²v-E²v|",
                      "iter",
                      "mv",
                      "pc",
                      "time [s]",
                      "avg [mv/s]",
                      "avg [pc/s]");

    for(const auto &[idx,entry] : iter::enumerate(entries)){
        if(max_entries and max_entries.value() <= idx) break;
        tools::log->log(level, "- {:<34} {:<7} {:<4} {:<4} {:<4} {:<4} {:<8.2e} {:<+22.15f} {:<+22.15f} {:<10.4e} {:<+22.15f} {:<18.15f} {:<18.15f} {:<8.2e} {:<8.2e} {:<9.2e} {:<5} {:<7} {:<7} {:<10.2e} {:<10.2e} {:<10.2e}",
                          entry.description,
                          entry.size, entry.ritz,entry.idx, entry.nev, entry.ncv, fp(entry.tol),
                          fp(entry.energy),
                          fp(entry.hsquared),
                          fp(entry.variance),
                          fp(entry.eigval),
                          fp(entry.overlap),fp(entry.norm), fp(entry.rnorm), fp(entry.rnorm_H1), fp(entry.rnorm_H2),
                          entry.iter, entry.mv, entry.pc,
                          entry.time,
                          static_cast<double>(entry.mv)/entry.time_mv,
                          static_cast<double>(entry.pc)/entry.time_pc
                          );
    }
    /* clang-format on */
    entries.clear();
}
