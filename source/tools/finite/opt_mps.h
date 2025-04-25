#pragma once
#include "config/enums.h"
#include "math/float.h"
#include "tensors/site/mps/MpsSite.h"
#include <complex>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tools::finite::opt {
    template<typename Scalar> using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    template<typename Scalar> using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar> using VectorReal = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, 1>;
    template<typename Scalar> using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename Scalar> using MatrixReal = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename Scalar>
    class opt_mps {
        private:
        std::optional<std::string>              name           = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 3>> tensor         = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> bond           = std::nullopt;
        std::optional<std::vector<size_t>>      sites          = std::nullopt;
        std::optional<RealScalar<Scalar>>       eshift         = std::nullopt; /*!< current energy shift in the energy MPOs: eshift  */
        std::optional<RealScalar<Scalar>>       energy_shifted = std::nullopt; /*!< <H-eshift>  */
        std::optional<RealScalar<Scalar>>       energy         = std::nullopt; /*!< Energy: eigenvalue of H: (E - eshift) + eshift   */
        std::optional<RealScalar<Scalar>>       hsquared       = std::nullopt; /*!< Variance: H²-E² = <(H-eshift)²> - <H-eshift>² */
        std::optional<RealScalar<Scalar>>       variance       = std::nullopt; /*!< Variance: H²-E² = <(H-eshift)²> - <H-eshift>² */
        std::optional<RealScalar<Scalar>>       rnorm_H1       = std::nullopt; /*!< Residual norm: H|ψ⟩-E|ψ⟩ */
        std::optional<RealScalar<Scalar>>       rnorm_H2       = std::nullopt; /*!< Residual norm: H²|ψ⟩-E²|ψ⟩ */
        std::optional<RealScalar<Scalar>>       overlap        = std::nullopt;
        std::optional<double>                   alpha          = std::nullopt;
        std::optional<RealScalar<Scalar>>       norm           = std::nullopt;
        std::optional<size_t>                   length         = std::nullopt;
        std::optional<size_t>                   iter           = std::nullopt;
        std::optional<size_t>                   num_op         = std::nullopt; /*!< Number of inverse-matrix-vector products */
        std::optional<size_t>                   num_mv         = std::nullopt; /*!< Number of matrix-vector products */
        std::optional<size_t>                   num_pc         = std::nullopt; /*!< Number of preconditioner calls */
        std::optional<double>                   time           = std::nullopt;
        std::optional<double>                   time_mv        = std::nullopt;
        std::optional<double>                   time_pc        = std::nullopt;
        std::optional<double>                   delta_f        = std::nullopt;
        std::optional<RealScalar<Scalar>>       grad_tol       = std::nullopt;
        std::optional<RealScalar<Scalar>>       grad_max       = std::nullopt;
        std::optional<RealScalar<Scalar>>       relchange      = std::nullopt;
        std::optional<long>                     eigs_idx       = std::nullopt;
        std::optional<long>                     eigs_nev       = std::nullopt;
        std::optional<long>                     eigs_ncv       = std::nullopt;
        std::optional<double>                   eigs_tol       = std::nullopt;
        std::optional<RealScalar<Scalar>>       eigs_rnorm     = std::nullopt;
        std::optional<RealScalar<Scalar>>       eigs_eigval    = std::nullopt;
        std::optional<std::string>              eigs_ritz      = std::nullopt;
        std::optional<Scalar>                   eigs_shift     = std::nullopt;
        std::optional<OptAlgo>                  optAlgo        = std::nullopt;
        std::optional<OptSolver>                optSolver      = std::nullopt;
        std::optional<OptRitz>                  optRitz        = std::nullopt;
        std::optional<OptExit>                  optExit        = std::nullopt;
        std::optional<long>                     bond_lim       = std::nullopt;
        std::optional<double>                   trnc_lim       = std::nullopt;

        template<typename T> static constexpr auto nan = std::numeric_limits<RealScalar<T>>::quiet_NaN();

        template<typename T>
        decltype(auto) get_or_nan(T &&val) const {
            if constexpr(sfinae::is_std_optional_v<T>) {
                using V = typename std::remove_cvref_t<T>::value_type;
                if constexpr(sfinae::is_std_complex_v<V>) {
                    using R = typename V::value_type;
                    return val.value_or(V(nan<R>, nan<R>));

                } else {
                    return val.value_or(nan<V>);
                }
            } else
                return (val);
        }
        template<typename T>
        decltype(auto) get(T &&val, std::string_view name) const {
            if constexpr(sfinae::is_std_optional_v<T>) {
                if(val.has_value()) { return val.value(); }
                throw except::runtime_error("opt_mps<{}>: {} <{}> has not been set.", sfinae::type_name<Scalar>, name, sfinae::type_name<T>);
            } else
                return (val);
        }

        public:
        bool                         is_basis_vector = false;
        std::vector<MpsSite<Scalar>> mps_backup; // Used during subspace expansion to keep track of compatible neighbor mps

        opt_mps() = default;
        // Constructor used for initial state
        template<typename T> opt_mps(std::string_view name_, const Eigen::Tensor<T, 3> &tensor_, const std::vector<size_t> &sites_, RealScalar<T> eshift_,
                                     RealScalar<T> energy_shifted_, std::optional<RealScalar<T>> variance_, RealScalar<T> overlap_, size_t length_)
            : name(name_), tensor(tenx::asScalarType<Scalar>(tensor_)), sites(sites_), eshift(eshift_),
              energy_shifted(static_cast<RealScalar<Scalar>>(energy_shifted_)),
              energy(static_cast<RealScalar<Scalar>>(energy_shifted_) + static_cast<RealScalar<Scalar>>(eshift_)), variance(variance_),
              overlap(static_cast<RealScalar<Scalar>>(overlap_)), length(length_) {
            norm   = get_vector().norm();
            iter   = 0;
            num_mv = 0;
            time   = 0;
        }
        // Constructor used for results
        template<typename T, typename S>
        opt_mps(std::string_view name_, const Eigen::Tensor<T, 3> &tensor_, const std::vector<size_t> &sites_, S energy_, S variance_, S overlap_,
                size_t length_, size_t iter_, size_t counter_, size_t time_)
            : name(name_), tensor(tenx::asScalarType<Scalar>(tensor_)), sites(sites_), energy(static_cast<RealScalar<Scalar>>(energy_)),
              variance(static_cast<RealScalar<Scalar>>(variance_)), overlap(static_cast<RealScalar<Scalar>>(overlap_)), length(length_), iter(iter_),
              num_mv(counter_), time(time_) {
            norm = get_vector().norm();
        }

        [[nodiscard]] bool        is_initialized() const;
        [[nodiscard]] const auto &get_tensor() const { return get(tensor, "tensor"); }
        [[nodiscard]] const auto  get_vector() const { return tenx::VectorMap(get_tensor()); }
        template<typename T>
        [[nodiscard]] decltype(auto) get_tensor_as() const {
            return tenx::asScalarType<T>(get_tensor());
        }

        template<typename T>
        [[nodiscard]] auto get_vector_as() const {
            return tenx::VectorCast(get_tensor_as<T>());
        }
        [[nodiscard]] const auto &get_name() const { return get(name, "name"); }
        [[nodiscard]] const auto &get_bond() const { return get(bond, "bond"); }
        [[nodiscard]] const auto &get_sites() const { return get(sites, "sites"); }
        [[nodiscard]] auto        get_energy() const { return get(energy, "energy"); }
        [[nodiscard]] auto        get_eshift() const { return get(eshift, "eshift"); }
        [[nodiscard]] auto        get_energy_shifted() const { return get(energy_shifted, "energy_shifted"); }
        [[nodiscard]] auto        get_hsquared() const { return get(hsquared, "hsquared"); }
        [[nodiscard]] auto        get_variance() const { return get(variance, "variance"); }
        [[nodiscard]] auto        get_rnorm_H() const { return get_or_nan(rnorm_H1); }
        [[nodiscard]] auto        get_rnorm_H2() const { return get_or_nan(rnorm_H2); }
        [[nodiscard]] auto        get_overlap() const { return get(overlap, "overlap"); }
        [[nodiscard]] auto        get_alpha() const { return get_or_nan(alpha); }
        [[nodiscard]] auto        get_norm() const { return get(norm, "norm"); }
        [[nodiscard]] auto        get_length() const { return get(length, "length"); }
        [[nodiscard]] auto        get_iter() const { return iter.value_or(0); }
        [[nodiscard]] auto        get_op() const { return num_op.value_or(0); }
        [[nodiscard]] auto        get_mv() const { return num_mv.value_or(0); }
        [[nodiscard]] auto        get_pc() const { return num_pc.value_or(0); }
        [[nodiscard]] auto        get_time() const { return time.value_or(0.0); }
        [[nodiscard]] auto        get_time_mv() const { return time_mv.value_or(0.0); }
        [[nodiscard]] auto        get_time_pc() const { return time_pc.value_or(0.0); }
        [[nodiscard]] auto        get_delta_f() const { return get_or_nan(delta_f); }
        [[nodiscard]] auto        get_grad_tol() const { return get_or_nan(grad_tol); }
        [[nodiscard]] auto        get_grad_max() const { return get_or_nan(grad_max); }
        [[nodiscard]] auto        get_relchange() const { return get_or_nan(relchange); }
        [[nodiscard]] auto        get_eigs_idx() const { return eigs_idx.value_or(-1l); }
        [[nodiscard]] auto        get_eigs_nev() const { return eigs_nev.value_or(-1l); }
        [[nodiscard]] auto        get_eigs_ncv() const { return eigs_ncv.value_or(-1l); }
        [[nodiscard]] auto        get_eigs_tol() const { return get_or_nan(eigs_tol); }
        [[nodiscard]] auto        get_eigs_rnorm() const { return get_or_nan(eigs_rnorm); }
        [[nodiscard]] auto        get_eigs_eigval() const { return get_or_nan(eigs_eigval); }
        [[nodiscard]] auto        get_eigs_ritz() const { return eigs_ritz.value_or("--"); }
        [[nodiscard]] auto        get_eigs_shift() const { return get_or_nan(eigs_shift); }
        [[nodiscard]] auto        get_optsolver() const { return get(optSolver, "optSolver"); }
        [[nodiscard]] auto        get_optalgo() const { return get(optAlgo, "optAlgo"); }
        [[nodiscard]] auto        get_optexit() const { return get(optExit, "optExit"); }
        [[nodiscard]] auto        get_bond_lim() const { return get(bond_lim, "bond_lim"); }
        [[nodiscard]] auto        get_trnc_lim() const { return get(trnc_lim, "trnc_lim"); }

        void normalize();
        void set_name(std::string_view name_) { name = name_; }
        template<typename T3>
        requires tenx::sfinae::is_eigen_tensor3<T3>
        void set_tensor(T3 &&tensor_) {
            tensor = tenx::asScalarType<Scalar>(tensor_);
            norm   = get_vector().norm();
        }
        template<typename T>
        void set_tensor(const VectorType<T> &vector, const Eigen::DSizes<long, 3> &dims) {
            set_tensor(Eigen::TensorMap<const Eigen::Tensor<const T, 3>>(vector.data(), dims));
        }
        template<typename T2>
        requires tenx::sfinae::is_eigen_tensor2<T2>
        void set_bond(T2 &&bond_) {
            bond = tenx::asScalarType<Scalar>(bond_);
        }
        template<typename MatrixType>
        requires tenx::sfinae::is_eigen_matrix_v<MatrixType>
        void set_bond(MatrixType &&matrix) {
            set_bond(tenx::TensorMap(matrix));
        }
        void set_sites(const std::vector<size_t> &sites_) { sites = sites_; }
        template<typename T>
        void set_energy_shifted(T energy_shifted_) {
            energy_shifted = static_cast<RealScalar<Scalar>>(energy_shifted_);
            if(energy and not eshift) eshift = energy.value() - energy_shifted.value();
            if(eshift and not energy) energy = energy_shifted.value() + eshift.value();
        }
        template<typename T>
        void set_eshift(T energy_shift_) {
            eshift = static_cast<RealScalar<Scalar>>(energy_shift_);
            if(energy and not energy_shifted) energy_shifted = energy.value() - eshift.value();
            if(energy_shifted and not energy) energy = energy_shifted.value() + eshift.value();
        }
        template<typename T>
        void set_energy(T energy_) {
            energy = static_cast<RealScalar<Scalar>>(energy_);
            if(eshift and not energy_shifted) energy_shifted = energy.value() - eshift.value();
            if(energy_shifted and not eshift) eshift = energy.value() - energy_shifted.value();
        }
        template<typename T> void set_hsquared(T hsquared_) { hsquared = hsquared_; }
        template<typename T> void set_variance(T variance_) { variance = variance_; }
        template<typename T> void set_rnorm_H1(T rnorm_H1_) { rnorm_H1 = static_cast<RealScalar<Scalar>>(rnorm_H1_); }
        template<typename T> void set_rnorm_H2(T rnorm_H2_) { rnorm_H2 = static_cast<RealScalar<Scalar>>(rnorm_H2_); }
        template<typename T> void set_overlap(T overlap_) { overlap = static_cast<RealScalar<Scalar>>(overlap_); }
        template<typename T> void set_alpha(T alpha_) { alpha = alpha_; }
        template<typename T> void set_length(T length_) { length = length_; }
        template<typename T> void set_iter(T iter_) { iter = iter_; }
        template<typename T> void set_op(T op_) { num_op = op_; }
        template<typename T> void set_mv(T mv_) { num_mv = mv_; }
        template<typename T> void set_pc(T pc_) { num_pc = pc_; }
        template<typename T> void set_time(T time_) { time = time_; }
        template<typename T> void set_time_mv(T time_mv_) { time_mv = time_mv_; }
        template<typename T> void set_time_pc(T time_pc_) { time_pc = time_pc_; }
        template<typename T> void set_delta_f(T delta_f_) { delta_f = delta_f_; }
        template<typename T> void set_grad_tol(T grad_tol_) { grad_tol = grad_tol_; }
        template<typename T> void set_grad_max(T grad_max_) { grad_max = grad_max_; }
        template<typename T> void set_relchange(T relchange_) { relchange = relchange_; }
        template<typename T> void set_eigs_idx(T eigs_idx_) { eigs_idx = eigs_idx_; }
        template<typename T> void set_eigs_nev(T eigs_nev_) { eigs_nev = eigs_nev_; }
        template<typename T> void set_eigs_ncv(T eigs_ncv_) { eigs_ncv = eigs_ncv_; }
        template<typename T> void set_eigs_tol(T eigs_tol_) { eigs_tol = eigs_tol_; }
        template<typename T> void set_eigs_rnorm(T eigs_rnorm_) { eigs_rnorm = static_cast<RealScalar<Scalar>>(eigs_rnorm_); }
        template<typename T> void set_eigs_eigval(T eigs_eigval_) { eigs_eigval = eigs_eigval_; }
        template<typename T> void set_eigs_ritz(T eigs_ritz_) { eigs_ritz = eigs_ritz_; }
        template<typename T> void set_eigs_shift(T eigs_shift_) {
            if constexpr(sfinae::is_std_complex_v<Scalar>){eigs_shift = static_cast<Scalar>(eigs_shift_);}
            else {
                eigs_shift = static_cast<Scalar>(std::real(eigs_shift_));
            }
        }

        template<typename T> void set_optsolver(T optSolver_) { optSolver = optSolver_; }
        template<typename T> void set_optalgo(T optAlgo_) { optAlgo = optAlgo_; }
        template<typename T> void set_optexit(T optExit_) { optExit = optExit_; }
        template<typename T> void set_bond_limit(T bond_lim_) { bond_lim = bond_lim_; }
        template<typename T> void set_trnc_limit(T trnc_lim_) { trnc_lim = trnc_lim_; }

        void validate_initial_mps() const;
        void validate_basis_vector() const;
        void validate_result() const;
        bool operator<(const opt_mps &rhs) const;
        bool operator>(const opt_mps &rhs) const;
        bool has_nan() const;
    };
}
