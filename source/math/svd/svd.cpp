#include "math/svd.h"
#include "tid/tid.h"
#include <debug/exceptions.h>
#include <Eigen/QR>
#include <fmt/ranges.h>

#if defined(_OPENMP)

    #include <omp.h>

#endif

svd::solver::solver() { setLogLevel(2); }

svd::config::config(long rank_max_) : rank_max(rank_max_) {}

svd::config::config(double truncation_lim_) : truncation_limit(truncation_lim_) {}

svd::config::config(long rank_max_, double truncation_lim_) : rank_max(rank_max_), truncation_limit(truncation_lim_) {}

svd::config::config(std::optional<long> rank_max_) : rank_max(rank_max_) {}

svd::config::config(std::optional<double> truncation_lim_) : truncation_limit(truncation_lim_) {}

svd::config::config(std::optional<long> rank_max_, std::optional<double> truncation_lim_) : rank_max(rank_max_), truncation_limit(truncation_lim_) {}

std::string svd::config::to_string() const {
    /* clang-format off */
    std::string msg;
    if (rank_max) msg.append(fmt::format(" | rank_max {}", rank_max.value()));
    if (rank_min) msg.append(fmt::format(" | rank_min {}", rank_min.value()));
    if (truncation_limit) msg.append(fmt::format(" | truncation_lim {:.2e}", truncation_limit.value()));
    if (switchsize_gejsv) msg.append(fmt::format(" | switchsize_gejsv {}", switchsize_gejsv.value()));
    if (switchsize_gesvd) msg.append(fmt::format(" | switchsize_gesvd {}", switchsize_gesvd.value()));
    if (switchsize_gesdd) msg.append(fmt::format(" | switchsize_gesdd {}", switchsize_gesdd.value()));
    if (svdx_select and std::holds_alternative<svdx_indices_t>(svdx_select.value())) {
        auto sel = std::get<svdx_indices_t>(svdx_select.value());
        msg.append(fmt::format(" | svdx_select {}-{}", sel.il, sel.iu));
    }
    if (svdx_select and std::holds_alternative<svdx_values_t>(svdx_select.value())) {
        auto sel = std::get<svdx_values_t>(svdx_select.value());
        msg.append(fmt::format(" | svdx_select {}-{}", sel.vl, sel.vu));
    }
    if (loglevel) msg.append(fmt::format(" | loglevel {}", loglevel.value()));
    if (svd_lib) msg.append(fmt::format(" | svd_lib {}", enum2sv(svd_lib.value())));
    if (svd_rtn) msg.append(fmt::format(" | svd_rtn {}", enum2sv(svd_rtn.value())));
    if (svd_save) msg.append(fmt::format(" | svd_save {}", enum2sv(svd_save.value())));
    if (benchmark) msg.append(fmt::format(" | benchmark {}", benchmark.value()));
    return msg.empty() ? msg : "svd settings" + msg;
    /* clang-format on */
}

void svd::solver::copy_config(const svd::config &svd_cfg) {
    if(svd_cfg.rank_max) rank_max = svd_cfg.rank_max.value();
    if(svd_cfg.rank_min) rank_min = svd_cfg.rank_min.value();
    if(svd_cfg.truncation_limit) truncation_lim = svd_cfg.truncation_limit.value();
    if(svd_cfg.switchsize_gejsv) switchsize_gejsv = svd_cfg.switchsize_gejsv.value();
    if(svd_cfg.switchsize_gesvd) switchsize_gesvd = svd_cfg.switchsize_gesvd.value();
    if(svd_cfg.switchsize_gesdd) switchsize_gesdd = svd_cfg.switchsize_gesdd.value();
    if(svd_cfg.svdx_select) svdx_select = svd_cfg.svdx_select.value();
    if(svd_cfg.loglevel) setLogLevel(svd_cfg.loglevel.value());
    if(svd_cfg.svd_lib) svd_lib = svd_cfg.svd_lib.value();
    if(svd_cfg.svd_rtn) svd_rtn = svd_cfg.svd_rtn.value();
    if(svd_cfg.svd_save) svd_save = svd_cfg.svd_save.value();
    if(svd_cfg.benchmark) benchmark = svd_cfg.benchmark.value();
}

svd::solver::solver(const svd::config &svd_cfg) : solver() { copy_config(svd_cfg); }

svd::solver::solver(std::optional<svd::config> svd_cfg) : solver() {
    if(svd_cfg) copy_config(svd_cfg.value());
}

void svd::solver::set_config(const svd::config &svd_cfg) { copy_config(svd_cfg); }

void svd::solver::set_config(std::optional<svd::config> svd_cfg) {
    if(svd_cfg) copy_config(svd_cfg.value());
}

void svd::solver::setLogLevel(int logLevel) {
    if(!log) {
        std::string name = "svd";
#if defined(_OPENMP)
        if(omp_in_parallel()) name = fmt::format("svd-{}", omp_get_thread_num());
#endif
        log = spdlog::get(name);
        if(!log) {
            log = spdlog::stdout_color_mt(name, spdlog::color_mode::always);
            log->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%n]%^[%=8l]%$ %v");
        }
    } else {
        if(logLevel != log->level()) { log->set_level(static_cast<spdlog::level::level_enum>(logLevel)); }
    }
}

double svd::solver::get_truncation_error() const { return truncation_error; }

long svd::solver::get_rank() const { return rank; }

long long svd::solver::get_count() { return count; }

