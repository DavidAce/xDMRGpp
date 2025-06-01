#pragma once
#include "math/svd/config.h"
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

enum class OptSolver;
enum class OptAlgo;
enum class OptType;
enum class OptWhen;
enum class OptExit;
enum class OptRitz;
enum class BondExpansionPolicy;

namespace tools::finite::opt {
    struct OptMeta {
        OptAlgo                    optAlgo;
        OptRitz                    optRitz;
        OptSolver                  optSolver;
        OptType                    optType;
        OptWhen                    optWhen;
        OptExit                    optExit;
        BondExpansionPolicy        bondexp_policy;
        size_t                     bondexp_maxiter   = 1ul;
        size_t                     bondexp_blocksize = 1ul;
        size_t                     bondexp_nkrylov   = 10ul;
        float                      bondexp_factor    = 1.0f;
        float                      bondexp_minalpha  = 1e-15f;
        float                      bondexp_maxalpha  = 1e-3f;
        size_t                     max_sites         = 2ul;
        size_t                     min_sites         = 1ul;
        long                       max_problem_size  = 0;
        long                       problem_size      = 0;
        std::array<long, 3>        problem_dims      = {};
        std::vector<size_t>        chosen_sites      = {};
        std::string                label;
        std::string                eigs_lib;
        std::optional<double>      subspace_tol          = std::nullopt;
        std::optional<double>      eigv_target           = std::nullopt; // AKA shift
        std::optional<double>      eigs_tol              = std::nullopt;
        std::optional<int>         eigs_nev              = std::nullopt;
        std::optional<int>         eigs_ncv              = std::nullopt;
        std::optional<int>         eigs_iter_max         = std::nullopt;
        std::optional<double>      eigs_time_max         = std::nullopt;
        std::optional<long>        eigs_jcbMaxBlockSize  = std::nullopt; // maximum  Jacobi block size (preconditioner)
        std::optional<svd::config> svd_cfg               = std::nullopt;
        std::optional<std::string> primme_method         = std::nullopt;
        std::optional<std::string> primme_projection     = std::nullopt; /*!< Choose primme_proj_<default|RR|harmonic|refined> */
        std::optional<int>         primme_minRestartSize = std::nullopt;
        std::optional<int>         primme_maxBlockSize   = std::nullopt;

        OptMeta();
        explicit OptMeta(OptAlgo algo, OptRitz ritz);
        [[nodiscard]] bool should_proceed(OptExit previous_exit) const;
        void               validate() const;
        std::string        string() const;
    };
}
