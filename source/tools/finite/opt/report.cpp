
#include "report.h"
#include "../opt_mps.h"
#include "general/iter.h"
#include "tid/tid.h"
#include "tools/common/log.h"


void tools::finite::opt::reports::print_subs_report(){
    if (tools::log->level() > spdlog::level::debug) return;
    if (subs_log.empty()) return;
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

    for(auto &entry : subs_log){
        tools::log->debug("- {:<5} {:<18.16f} {:<18.16f} {:<18.2e} {:<11.2e} {:<11.2e} {:<11.2e} {:<6} {:<6} {:<6}",
                          entry.nev,
                          entry.max_olap,
                          entry.min_olap,
                          entry.eps ,
                          entry.eig_time,
                          entry.ham_time,
                          entry.lu_time ,
                          entry.iter,
                          entry.mv,
                          entry.pc
                          );
    }
    subs_log.clear();
}




void tools::finite::opt::reports::print_eigs_report(std::optional<size_t> max_entries){
    if (eigs_log.empty()) return;
    auto level = eigs_log.front().level;
    if (level < tools::log->level()) {
        eigs_log.clear();
        return;
    }
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

    for(const auto &[idx,entry] : iter::enumerate(eigs_log)){
        if(max_entries and max_entries.value() <= idx) break;
        tools::log->log(level, "- {:<34} {:<7} {:<4} {:<4} {:<4} {:<4} {:<8.2e} {:<+22.15f} {:<+22.15f} {:<10.4e} {:<+22.15f} {:<18.15f} {:<18.15f} {:<8.2e} {:<8.2e} {:<9.2e} {:<5} {:<7} {:<7} {:<10.2e} {:<10.2e} {:<10.2e}",
                          entry.description,
                          entry.size, entry.ritz,entry.idx, entry.nev, entry.ncv, entry.tol,
                          entry.energy,
                          entry.hsquared,
                          entry.variance,
                          entry.eigval,
                          entry.overlap,entry.norm, entry.rnorm, entry.rnorm_H, entry.rnorm_H2,
                          entry.iter, entry.mv, entry.pc,
                          entry.time,
                          static_cast<double>(entry.mv)/entry.time_mv,
                          static_cast<double>(entry.pc)/entry.time_pc
                          );
    }
    eigs_log.clear();
}

/* clang-format on */
void tools::finite::opt::reports::subs_add_entry(long nev, double max_olap, double min_olap, double eps, double eig_time, double ham_time, double lu_time,
                                                 long iter, long mv, long pc) {
    if(tools::log->level() > spdlog::level::debug) return;
    subs_log.push_back({nev, max_olap, min_olap, eps, eig_time, ham_time, lu_time, iter, mv, pc});
}

void tools::finite::opt::reports::eigs_add_entry(const opt_mps &mps, spdlog::level::level_enum level) {
    if(level < tools::log->level()) return;
    std::string description = fmt::format("{:<24}", mps.get_name());
    eigs_log.push_back(eigs_entry{description, std::string(mps.get_eigs_ritz()), mps.get_tensor().size(), mps.get_eigs_idx(), mps.get_eigs_nev(),
                                  mps.get_eigs_ncv(), mps.get_energy(), mps.get_hsquared(), mps.get_variance(), mps.get_eigs_eigval(), mps.get_overlap(), mps.get_norm(),
                                  mps.get_eigs_rnorm(), mps.get_rnorm_H(),mps.get_rnorm_H2(), mps.get_eigs_tol(), mps.get_grad_max(), mps.get_iter(), mps.get_mv(), mps.get_pc(), mps.get_time(), mps.get_time_mv(), mps.get_time_pc(), level});
}
