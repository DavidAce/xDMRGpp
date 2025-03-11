#pragma once
#include "tools/common/log.h"
#include <optional>
#include <string>
#include <vector>
namespace tools::finite::opt {
    class opt_mps;
}

namespace tools::finite::opt::reports {
    struct subs_entry {
        long   nev;
        double max_olap, min_olap, eps, eig_time, ham_time, lu_time;
        long   iter, mv, pc;
    };

    struct eigs_entry {
        std::string               description;
        std::string               ritz;
        long                      size, idx, nev, ncv;
        double                    energy, hsquared, variance, eigval, overlap, norm, rnorm, rnorm_H, rnorm_H2, tol, grad;
        size_t                    iter, mv, pc;
        double                    time, time_mv, time_pc;
        spdlog::level::level_enum level = spdlog::level::debug;
    };

    inline std::vector<subs_entry> subs_log;
    inline std::vector<eigs_entry> eigs_log;

    extern void print_subs_report();
    extern void print_eigs_report(std::optional<size_t> max_entries = std::nullopt);
    extern void bfgs_add_entry(const std::string &description, long size, long space, double energy, double variance, double overlap, double norm, double rnorm,
                               double delta_f, double grad_norm, size_t iter, size_t counter, double time);
    extern void time_add_entry();
    extern void subs_add_entry(long nev, double max_olap, double min_olap, double eps, double eig_time, double ham_time, double lu_time, long iter, long mv,
                               long pc);
    extern void eigs_add_entry(const opt_mps &mps, spdlog::level::level_enum level = spdlog::level::debug);
}