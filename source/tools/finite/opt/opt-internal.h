#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "tools/common/log.h"
#include "tools/finite/opt.h"
#include "tools/finite/opt_mps.h"

/* clang-format off */
namespace tools::finite::opt::internal{
    template<typename T>
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename T, auto rank = 3>
    using TensorType = Eigen::Tensor<T, rank>;

    [[nodiscard]] extern opt_mps optimize_energy_eig                      (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_energy                          (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_overlap                         (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_subspace_variance               (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_folded_spectrum_eig             (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_folded_spectrum                 (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_generalized_shift_invert_eig    (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_generalized_shift_invert        (const TensorsFinite & tensors, const opt_mps & initial_mps, const AlgorithmStatus & status, OptMeta & meta);
    [[nodiscard]] extern opt_mps optimize_lanczos_h1h2                    (const TensorsFinite & tensors, const opt_mps & initial, const AlgorithmStatus & status, OptMeta & meta);
    extern void extract_results                                           (const TensorsFinite & tensors, const opt_mps & initial_mps, const OptMeta & meta, const eig::solver &solver,
                                                             std::vector<opt_mps> &results, bool converged_only = true, std::optional<std::vector<long>> indices = std::nullopt);
    extern void extract_results_subspace (const TensorsFinite &tensors, const opt_mps &initial_mps, const OptMeta &meta,
                                               const eig::solver &solver, const std::vector<opt_mps> & subspace_mps, std::vector<opt_mps> &results );



    namespace comparator{
        extern bool energy              (const opt_mps &lhs, const opt_mps &rhs);
        extern bool energy_absolute     (const opt_mps &lhs, const opt_mps &rhs);
        extern bool energy_distance     (const opt_mps &lhs, const opt_mps &rhs, double target);
        extern bool variance            (const opt_mps &lhs, const opt_mps &rhs);
        extern bool gradient            (const opt_mps &lhs, const opt_mps &rhs);
        extern bool eigval              (const opt_mps &lhs, const opt_mps &rhs);
        extern bool eigval_absolute     (const opt_mps &lhs, const opt_mps &rhs);
        extern bool overlap             (const opt_mps &lhs, const opt_mps &rhs);
        extern bool eigval_and_overlap  (const opt_mps &lhs, const opt_mps &rhs);
    }
    struct Comparator{
        const OptMeta * const meta = nullptr;
        double target_energy;
        Comparator(const OptMeta &meta_, double initial_energy_ = std::numeric_limits<double>::quiet_NaN());
        bool operator() (const opt_mps &lhs, const opt_mps &rhs);
    };
    struct EigIdxComparator{
        OptRitz ritz = OptRitz::NONE;
        double shift = 0;
        // double * data = nullptr;
        // long   size = 0;
        Eigen::Map<Eigen::VectorXd> eigvals;
        EigIdxComparator(OptRitz ritz_, double shift_, double * data_, long size_);
        bool operator() (long lhs, long rhs);
    };

    namespace subspace{
        extern std::vector<int> generate_nev_list(int rows);

        template<typename Scalar>
        extern std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
        find_subspace_part(const TensorsFinite & tensors, double energy_target,
                                                          const OptMeta & meta);

        template<typename Scalar>
        extern std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
        find_subspace_primme(const TensorsFinite & tensors, double eigval_shift, const OptMeta & meta);


        template<typename Scalar>
        extern std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
        find_subspace_prec(const TensorsFinite & tensors, double energy_target, const OptMeta & meta);


        template<typename Scalar>
        extern std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
        find_subspace_lapack(const TensorsFinite & tensors);

        template<typename Scalar>
        extern std::vector<opt_mps>
        find_subspace(const TensorsFinite &tensors, const OptMeta & meta);

        extern void
        filter_subspace(std::vector<opt_mps> & subspace, size_t max_accept);

        extern std::optional<size_t> get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps> & eigvecs);
        extern std::optional<size_t> get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps> & eigvecs);
        extern std::vector<size_t>   get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps> & eigvecs, size_t max_eigvecs);

        extern Eigen::MatrixXcd get_eigvecs(const std::vector<opt_mps> & eigvecs);
        extern Eigen::VectorXd get_eigvals(const std::vector<opt_mps> & eigvecs);
        extern Eigen::VectorXd get_energies(const std::vector<opt_mps> & eigvecs);
        extern Eigen::VectorXd get_overlaps(const std::vector<opt_mps> & eigvecs);
        extern std::vector<double> get_subspace_errors(const std::vector<opt_mps> & eigvecs);
        extern double get_subspace_error(const std::vector<opt_mps> & eigvecs, std::optional<size_t> max_eigvecs = std::nullopt);
        extern double get_subspace_error(const std::vector<double> &overlaps);
        extern Eigen::VectorXcd get_vector_in_subspace(const std::vector<opt_mps> & eigvecs, size_t idx);
        extern Eigen::VectorXcd get_vector_in_subspace(const std::vector<opt_mps> & eigvecs, const Eigen::VectorXcd & subspace_vector);
        extern Eigen::VectorXcd get_vector_in_fullspace(const std::vector<opt_mps> & eigvecs, const Eigen::VectorXcd & subspace_vector);
        extern TensorType<cx64,3> get_tensor_in_fullspace(const std::vector<opt_mps> & eigvecs, const Eigen::VectorXcd & subspace_vector, const std::array<Eigen::Index,3> & dims);
        template <typename T>
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> get_hamiltonian_in_subspace(const ModelFinite & model,
                                                                                                         const EdgesFinite & edges,
                                                                                                         const std::vector<opt_mps> & eigvecs);
        template <typename T>
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> get_hamiltonian_squared_in_subspace(const ModelFinite & model,
                                                                                                         const EdgesFinite & edges,
                                                                                                         const std::vector<opt_mps> & eigvecs);
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
