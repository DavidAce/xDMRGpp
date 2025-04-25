#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "tools/common/log.h"
#include "tools/finite/opt.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_mps.h"

/* clang-format off */
namespace tools::finite::opt::internal{
    // template<typename Scalar> using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    template<typename Scalar, auto rank> using TensorType = Eigen::Tensor<Scalar, rank>;
    template<typename Scalar, auto rank> using TensorReal = Eigen::Tensor<RealScalar<Scalar>, rank>;
    // template<typename Scalar> using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    // template<typename Scalar> using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    // template<typename Scalar> using MatrixReal = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, Eigen::Dynamic>;
    // template<typename Scalar> using VectorReal = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, 1>;


    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_energy_eig                      (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_energy                          (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_overlap                         (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::subs_log<Scalar> & slog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_subspace_variance               (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_folded_spectrum_eig             (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_folded_spectrum                 (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_generalized_shift_invert_eig    (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_generalized_shift_invert        (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar> [[nodiscard]] extern opt_mps<Scalar> optimize_lanczos_h1h2                    (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial, const AlgorithmStatus & status, OptMeta & meta, reports::eigs_log<Scalar> & elog);
    template<typename Scalar>               extern void            extract_results                          (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const OptMeta & meta, const eig::solver &solver, std::vector<opt_mps<Scalar>> &results, bool converged_only = true, std::optional<std::vector<long>> indices = std::nullopt);
    template<typename Scalar>               extern void            extract_results_subspace                 (const TensorsFinite<Scalar> & tensors, const opt_mps<Scalar> & initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<Scalar>> & subspace_mps, std::vector<opt_mps<Scalar>> &results );



    namespace comparator{
        template<typename Scalar> extern bool energy              (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
        template<typename Scalar> extern bool energy_absolute     (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
        template<typename Scalar> extern bool energy_distance     (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs, RealScalar<Scalar> target);
        template<typename Scalar> extern bool variance            (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
        template<typename Scalar> extern bool gradient            (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
        template<typename Scalar> extern bool eigval              (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
        template<typename Scalar> extern bool eigval_absolute     (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
        template<typename Scalar> extern bool overlap             (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
        template<typename Scalar> extern bool eigval_and_overlap  (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
    }
    template<typename Scalar>
    struct Comparator{
        const OptMeta * const meta = nullptr;
        RealScalar<Scalar> target_energy;
        Comparator(const OptMeta &meta_, RealScalar<Scalar> initial_energy_ = std::numeric_limits<RealScalar<Scalar>>::quiet_NaN());
        bool operator() (const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs);
    };
    template<typename Scalar>
    struct EigIdxComparator{
        OptRitz ritz = OptRitz::NONE;
        Scalar shift = 0;
        // double * data = nullptr;
        // long   size = 0;
        Eigen::Map<VectorType<Scalar>> eigvals;
        EigIdxComparator(OptRitz ritz_, Scalar shift_, Scalar * data_, long size_);
        bool operator() (long lhs, long rhs);
    };

    namespace subspace{
        extern std::vector<int> generate_nev_list(int rows);

        template<typename T, typename Scalar>
        extern std::pair<MatrixType<T>, VectorReal<T>>
        find_subspace_part(const TensorsFinite<Scalar> & tensors, RealScalar<Scalar> energy_target, const OptMeta & meta, reports::subs_log<Scalar> &slog);

        template<typename T, typename Scalar>
        extern std::pair<MatrixType<T>, VectorReal<T>>
        find_subspace_primme(const TensorsFinite<Scalar> & tensors, RealScalar<Scalar> eigval_shift, const OptMeta & meta, reports::subs_log<Scalar> &slog);


        template<typename T, typename Scalar>
        extern std::pair<MatrixType<T>, VectorReal<T>>
        find_subspace_prec(const TensorsFinite<Scalar> & tensors, RealScalar<Scalar> energy_target, const OptMeta & meta, reports::subs_log<Scalar> &slog);


        template<typename T, typename Scalar>
        extern std::pair<MatrixType<T>, VectorReal<T>>
        find_subspace_lapack(const TensorsFinite<Scalar> & tensors, reports::subs_log<Scalar> &slog);

        template<typename T, typename Scalar>
        extern std::vector<opt_mps<Scalar>> find_subspace(const TensorsFinite<Scalar> &tensors, const OptMeta & meta, reports::subs_log<Scalar> &slog);

        template<typename Scalar> extern void filter_subspace(std::vector<opt_mps<Scalar>> & subspace, size_t max_accept);

        template<typename Scalar> extern std::optional<size_t> get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<Scalar>> & eigvecs);
        template<typename Scalar> extern std::optional<size_t> get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<Scalar>> & eigvecs);
        template<typename Scalar> extern std::vector<size_t>   get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<Scalar>> & eigvecs, size_t max_eigvecs);
        template<typename Scalar> extern MatrixType<Scalar> get_eigvecs(const std::vector<opt_mps<Scalar>> & eigvecs);
        template<typename Scalar> extern VectorReal<Scalar> get_eigvals(const std::vector<opt_mps<Scalar>> & eigvecs);
        template<typename Scalar> extern VectorReal<Scalar> get_energies(const std::vector<opt_mps<Scalar>> & eigvecs);
        template<typename Scalar> extern VectorReal<Scalar> get_overlaps(const std::vector<opt_mps<Scalar>> & eigvecs);
        template<typename Scalar> extern std::vector<RealScalar<Scalar>> get_subspace_errors(const std::vector<opt_mps<Scalar>> & eigvecs);
        template<typename Scalar> extern RealScalar<Scalar> get_subspace_error(const std::vector<opt_mps<Scalar>> & eigvecs, std::optional<size_t> max_eigvecs = std::nullopt);
        template<typename Scalar> extern RealScalar<Scalar> get_subspace_error(const std::vector<RealScalar<Scalar>> &overlaps);
        template<typename Scalar> extern VectorType<Scalar> get_vector_in_subspace(const std::vector<opt_mps<Scalar>> & eigvecs, size_t idx);
        template<typename Scalar> extern VectorType<Scalar> get_vector_in_subspace(const std::vector<opt_mps<Scalar>> & eigvecs,    const VectorType<Scalar> & subspace_vector);
        template<typename Scalar> extern VectorType<Scalar> get_vector_in_fullspace(const std::vector<opt_mps<Scalar>> & eigvecs,   const VectorType<Scalar> & subspace_vector);
        template<typename Scalar> extern TensorType<Scalar,3> get_tensor_in_fullspace(const std::vector<opt_mps<Scalar>> & eigvecs, const VectorType<Scalar> & subspace_vector, const std::array<Eigen::Index,3> & dims);
        template <typename T, typename Scalar>
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> get_hamiltonian_in_subspace(const ModelFinite<Scalar> & model,
                                                                                   const EdgesFinite<Scalar> & edges,
                                                                                   const std::vector<opt_mps<Scalar>> & eigvecs);
        template <typename T, typename Scalar>
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> get_hamiltonian_squared_in_subspace(const ModelFinite<Scalar> & model,
                                                                                           const EdgesFinite<Scalar> & edges,
                                                                                           const std::vector<opt_mps<Scalar>> & eigvecs);
    }

    inline bool no_state_in_window = false;



    extern double windowed_func_abs(double x,double window);
    extern double windowed_grad_abs(double x,double window);
    extern double windowed_func_pow(double x,double window);
    extern double windowed_grad_pow(double x,double window);
    extern std::pair<double,double> windowed_func_grad(double x,double window);
    extern long get_ops(long d, long chiL, long chiR, long m);
    extern long get_ops_R(long d, long chiL, long chiR, long m);



}
