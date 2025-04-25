#pragma once
#include "tools/common/log.h"
#include <optional>
#include <string>
#include <vector>
namespace tools::finite::opt {
    template<typename Scalar> class opt_mps;
}

namespace tools::finite::opt::reports {
    template<typename Scalar>
    struct subs_log {
        private:
        using Real = decltype(std::real(std::declval<Scalar>()));
        struct subs_entry {
            long   nev;
            Real   max_olap, min_olap, eps;
            double eig_time, ham_time, lu_time;
            long   iter, mv, pc;
        };

        public:
        std::vector<subs_entry> entries;

        void print_subs_report();
        template<typename T>
        void subs_add_entry(long nev, T max_olap, T min_olap, T eps, double eig_time, double ham_time, double lu_time, long iter, long mv, long pc) {
            if(tools::log->level() > spdlog::level::debug) return;
            entries.push_back({nev, static_cast<Real>(max_olap), static_cast<Real>(min_olap), eps, eig_time, ham_time, lu_time, iter, mv, pc});
        }
        void clear() { entries.clear(); }
        auto size() const { return entries.size(); }
    };

    template<typename Scalar>
    struct eigs_log {
        private:
        using Real = decltype(std::real(std::declval<Scalar>()));
        struct eigs_entry {
            std::string               description;
            std::string               ritz;
            long                      size, idx, nev, ncv;
            Real                      energy, hsquared, variance, eigval, overlap, norm, rnorm, rnorm_H1, rnorm_H2, grad;
            double                    tol;
            size_t                    iter, mv, pc;
            double                    time, time_mv, time_pc;
            spdlog::level::level_enum level = spdlog::level::debug;
        };

        public:
        std::vector<eigs_entry> entries;

        void print_eigs_report(std::optional<size_t> max_entries = std::nullopt);
        void eigs_add_entry(const opt_mps<Scalar> &mps, spdlog::level::level_enum level = spdlog::level::debug) {
            if(level < tools::log->level()) return;
            std::string description = fmt::format("{:<24}", mps.get_name());
            entries.push_back(eigs_entry{.description = description,
                                         .ritz        = std::string(mps.get_eigs_ritz()),
                                         .size        = mps.get_tensor().size(),
                                         .idx         = mps.get_eigs_idx(),
                                         .nev         = mps.get_eigs_nev(),
                                         .ncv         = mps.get_eigs_ncv(),
                                         .energy      = mps.get_energy(),
                                         .hsquared    = mps.get_hsquared(),
                                         .variance    = mps.get_variance(),
                                         .eigval      = mps.get_eigs_eigval(),
                                         .overlap     = mps.get_overlap(),
                                         .norm        = mps.get_norm(),
                                         .rnorm       = mps.get_eigs_rnorm(),
                                         .rnorm_H1    = mps.get_rnorm_H(),
                                         .rnorm_H2    = mps.get_rnorm_H2(),
                                         .grad        = mps.get_grad_max(),
                                         .tol         = mps.get_eigs_tol(),
                                         .iter        = mps.get_iter(),
                                         .mv          = mps.get_mv(),
                                         .pc          = mps.get_pc(),
                                         .time        = mps.get_time(),
                                         .time_mv     = mps.get_time_mv(),
                                         .time_pc     = mps.get_time_pc(),
                                         .level = level});
        }
        void clear() { entries.clear(); }
        auto size() { return entries.size(); }
    };

    // template<typename Scalar> inline std::vector<eigs_entry<Scalar>> eigs_log;

}