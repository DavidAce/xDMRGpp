#pragma once

#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "svd/config.h"
#include "tools/common/log.h"
#include <optional>
namespace tid {
    class ur;
}

namespace svd {
    template<typename Scalar>
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename Scalar>
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename T>
    using RealScalar = decltype(std::real(std::declval<T>()));

    namespace internal {
        // LAPACK uses internal workspace arrays which can be reused for the duration of the program.
        // Call clear() to recover this memory space
        void clear_lapack();

        template<typename Scalar>
        struct DumpSVD {
            using Real       = decltype(std::real(std::declval<Scalar>()));
            using Cplx       = std::complex<Real>;
            using VectorReal = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
            using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
            MatrixType A, U, VT;
            VectorReal S;
            int        info             = 0;
            long       rank             = -1;
            long       rank_max         = -1;
            long       rank_min         = -1;
            double     truncation_lim   = -1.0;
            double     truncation_error = -1.0;
            size_t     switchsize_gejsv = -1ul;
            size_t     switchsize_gesvd = -1ul;
            size_t     switchsize_gesdd = -1ul;
            svd::lib   svd_lib;
            svd::rtn   svd_rtn;
            svd::save  svd_save;
            ~DumpSVD();
        };

    }

    class solver {
        private:
        tools::LoggerHandle     log              = nullptr;
        mutable double          truncation_error = 0; // Stores the last truncation error
        mutable long            rank             = 0; // Stores the last rank
        inline static long long count            = 0; // Count the number of svd invocations for this execution

        public:
        long   rank_max       = -1;                                     /*!< -1 means determine from given matrix */
        long   rank_min       = -1;                                     /*!< -1 means disabled. */
        double truncation_lim = std::numeric_limits<double>::epsilon(); /*!< Truncation error limit, discard all lambda_i for highest i satisfying
                                                                           truncation_lim < norm(lambda_i) */

        // Switch sizes for automatic algorithm promotion (only used with svd::rtn::geauto)
        size_t                       switchsize_gejsv = 1;  /*!< Default jacobi algorithm (gesjsv) when min(rows,cols) >= swtichsize_gejsv, otherwise gesvj */
        size_t                       switchsize_gesvd = 32; /*!< Default preconditioned QR bidiagonalization (gesvd) when min(rows,cols) >= maxsize_gesvd */
        size_t                       switchsize_gesdd = 64; /*!< Default bidiagonal divide and conquer when  min(rows,cols) >= switchsize_gesdd */
        svd::lib                     svd_lib          = svd::lib::lapacke;
        svd::rtn                     svd_rtn          = svd::rtn::geauto;
        svd::save                    svd_save         = svd::save::NONE;
        std::optional<svdx_select_t> svdx_select      = std::nullopt;
        bool                         benchmark        = false;

        private:
        template<typename Scalar>
        std::tuple<MatrixType<Scalar>, VectorType<Scalar>, MatrixType<Scalar>> do_svd_lapacke(const Scalar *mat_ptr, long rows, long cols) const;

        template<typename Scalar>
        std::tuple<MatrixType<Scalar>, VectorType<Scalar>, MatrixType<Scalar>> do_svd_eigen(const Scalar *mat_ptr, long rows, long cols) const;

        template<typename Scalar>
        std::tuple<MatrixType<Scalar>, VectorType<Scalar>, MatrixType<Scalar>> do_svd_rsvd(const Scalar *mat_ptr, long rows, long cols) const;

        void copy_config(const svd::config &svd_cfg);

        // template<typename T>
        // [[nodiscard]] std::pair<long, fp64> get_rank_from_truncation_error(const T &S) const {
        //     VectorType<fp64> truncation_errors(S.size() + 1);
        //     for(long s = 0; s <= S.size(); s++) {
        //         truncation_errors[s] = static_cast<fp64>(S.tail(S.size() - s).norm());
        //     } // Last one should be zero, i.e. no truncation
        //     auto rank_    = (truncation_errors.array() >= truncation_lim).count();
        //     auto rank_lim = S.size();
        //     if(rank_max > 0) rank_lim = std::min(S.size(), rank_max);
        //     rank_ = std::min(rank_, rank_lim);
        //     if(rank_min > 0) rank_ = std::max(rank_, std::min(S.size(),
        //                                                       rank_min)); // Make sure we don't overtruncate in some cases (e.g. when stashing)
        //     if(rank_ <= 0) {
        //         if(log)
        //             log->error("Size {} | Rank {} | Rank limit {} | truncation error limit {:8.2e} | error {:8.2e}", S.size(), rank_, rank_lim,
        //             truncation_lim,
        //                        truncation_errors[rank_]);
        //         throw std::logic_error("rank <= 0");
        //     }
        //     // log->info("Truncation error: {:.5e} limit {:.5e} | Smallest kept singular value: {:.5e} ({} in total)",
        //     // fp(truncation_errors[rank_]),fp(truncation_lim), fp(S[rank_-1]), S.size());
        //     return {rank_, truncation_errors[rank_]};
        // }

        template<typename Vec>
        [[nodiscard]] std::pair<long, fp64> get_rank_from_truncation_error(const Vec &S) const {
            using Scalar = typename Vec::Scalar;
            using Real   = Eigen::NumTraits<Scalar>::Real;
            const long n = safe_cast<long>(S.size());
            if(n <= 0) throw std::logic_error("get_rank_from_truncation_error: S.size() <= 0");

            if(!(truncation_lim >= 0.0 && truncation_lim < 1.0))
                throw std::logic_error("get_rank_from_truncation_error: truncation_lim must satisfy 0 <= truncation_lim < 1");

            const fp64 SnormSq = static_cast<fp64>(S.squaredNorm());
            if(!(SnormSq > 0.0)) throw std::logic_error("get_rank_from_truncation_error: ||S|| == 0");

            // Strictly-positive prefix length (S is sorted descending)
            long pos_lim = 0;
            while(pos_lim < n && std::real(S(pos_lim)) > 0) ++pos_lim;
            if(pos_lim == 0) throw std::logic_error("get_rank_from_truncation_error: no positive singular values");

            long rank_lim = pos_lim;
            if(rank_max > 0) rank_lim = std::min(rank_lim, safe_cast<long>(rank_max));
            if(rank_lim <= 0) throw std::logic_error("get_rank_from_truncation_error: rank_lim <= 0");

            const fp64 thrSq = static_cast<fp64>(truncation_lim) * static_cast<fp64>(truncation_lim) * SnormSq;

            // Choose smallest rank_ in [1, rank_lim] with ||S.tail(n-rank_)||^2 < thrSq
            fp64 tail_norm_sq = 0.0;
            long rank_        = rank_lim;

            for(long i = n - 1; i >= 1; --i) {
                const fp64 si  = static_cast<fp64>(std::real(S(i)));
                tail_norm_sq  += si * si;

                if(i <= rank_lim && tail_norm_sq < thrSq) rank_ = i;
            }

            if(rank_min > 0) rank_ = std::max(rank_, std::min(rank_lim, safe_cast<long>(rank_min)));

            if(std::real(S(rank_ - 1)) <= Real{0}) throw std::logic_error("get_rank_from_truncation_error: rank selection would keep a zero singular value");

            // Return relative truncation error (dimensionless)
            fp64 tail_norm_sq_final = 0.0;
            for(long i = rank_; i < n; ++i) {
                const fp64 si       = static_cast<fp64>(std::real(S(i)));
                tail_norm_sq_final += si * si;
            }
            const fp64 trunc_err_rel = std::sqrt(tail_norm_sq_final / SnormSq);

            return {rank_, trunc_err_rel};
        }

        template<typename Scalar>
        void print_matrix(const Scalar *mat_ptr, long rows, long cols, std::string_view tag, long dec = 8) const;
        template<typename Scalar>
        void print_vector(const Scalar *vec_ptr, long size, std::string_view tag, long dec = 8) const;

        public:
        solver();
        solver(const svd::config &svd_cfg);
        solver(std::optional<svd::config> svd_cfg);

        void set_config(const svd::config &svd_cfg);
        void set_config(std::optional<svd::config> svd_cfg);

        void                           setLogLevel(int logLevel);
        [[nodiscard]] double           get_truncation_error() const;
        [[nodiscard]] long             get_rank() const;
        [[nodiscard]] static long long get_count();

        template<typename Scalar>
        std::tuple<MatrixType<Scalar>, VectorType<Scalar>, MatrixType<Scalar>> do_svd_ptr(const Scalar *mat_ptr, long rows, long cols,
                                                                                          const svd::config &settings);

        template<typename Derived>
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> pseudo_inverse(const Eigen::EigenBase<Derived> &matrix,
                                                                                               const svd::config               &svd_cfg = svd::config()) {
            assert(matrix.derived().rows() > 0);
            assert(matrix.derived().cols() > 0);
            // Eigen::Map<const MatrixType<Scalar>> mat(tensor.data(), tensor.dimension(0), tensor.dimension(1));
            // return tenx::TensorCast(mat.completeOrthogonalDecomposition().pseudoInverse());

            auto [U, S, VT] = do_svd_ptr(matrix.derived().data(), matrix.derived().rows(), matrix.derived().cols(), svd_cfg);
            return VT.adjoint() * S.cwiseInverse().asDiagonal() * U.adjoint();
        }

        template<typename Scalar>
        Eigen::Tensor<Scalar, 2> pseudo_inverse(const Eigen::Tensor<Scalar, 2> &tensor, const svd::config &svd_cfg = svd::config()) {
            return tenx::TensorMap(pseudo_inverse(tenx::MatrixMap(tensor), svd_cfg));
        }

        template<typename Derived>
        auto do_svd(const Eigen::DenseBase<Derived> &mat, const svd::config &svd_cfg = svd::config()) {
            return do_svd_ptr(mat.derived().data(), mat.rows(), mat.cols(), svd_cfg);
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 2>> decompose(const Eigen::Tensor<Scalar, 2> &tensor,
                                                                                                           const svd::config &svd_cfg = svd::config()) {
            auto [U, S, VT] = do_svd_ptr(tensor.data(), tensor.dimension(0), tensor.dimension(1), svd_cfg);
            return std::make_tuple(tenx::TensorMap(U), tenx::TensorMap(S.template cast<Scalar>()), tenx::TensorMap(VT));
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 2>>
            decompose(const Eigen::Tensor<Scalar, 3> &tensor, const long rows, const long cols, const svd::config &svd_cfg = svd::config()) {
            if(rows * cols != tensor.size()) throw std::runtime_error("rows * cols  != tensor.size()");
            auto [U, S, VT] = do_svd_ptr(tensor.data(), rows, cols, svd_cfg);
            return std::make_tuple(tenx::TensorMap(U), tenx::TensorMap(S.template cast<Scalar>()), tenx::TensorMap(VT));
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 3>> decompose(const Eigen::Tensor<Scalar, 4> &tensor,
                                                                                                           const svd::config &svd_cfg = svd::config()) {
            long dL   = tensor.dimension(0);
            long chiL = tensor.dimension(1);
            long dR   = tensor.dimension(2);
            long chiR = tensor.dimension(3);
            if(dL * chiL * dR * chiR != tensor.size()) throw std::range_error("schmidt error: tensor size does not match given dimensions.");
            auto [U, S, VT] = do_svd_ptr(tensor.data(), dL * chiL, dR * chiR, svd_cfg);
            return std::make_tuple(tenx::TensorMap(U, dL, chiL, S.size()), tenx::TensorMap(S.template cast<Scalar>(), S.size()),
                                   tenx::TensorMap(VT, S.size(), dR, chiR).shuffle(tenx::array3{1, 0, 2}));
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 3>>
            decompose_multisite(const Eigen::Tensor<Scalar, 3> &tensor, long dL, long dR, long chiL, long chiR, const svd::config &svd_cfg = svd::config()) {
            /* This function assumes that the tensor is given with indices in multisite standard form, i.e., with order
             *
             * 1) physical indices (any number of them, contracted from left to right)
             * 2) left bond index
             * 4) right bond index
             *
             * (1)chiL ---[tensor]--- (2)chiR
             *               |
             *          (0)d*d*d...
             *
             * we start by transposing the tensor into "left-right" order suitable for a decomposition
             *
             * (1)chiL---[  tensor  ]---(3)chiR
             *            |        |
             *        (0)dL      (2)dR
             *
             * The function call argument order dL, dR, chiL,chiR is meant as a hint for how to use this function.
             *
             * NOTE: This function does not normalize the singular values!
             *
             */
            if(dR == 1) { // probably a single-site right-moving split. No need for a shuffle on VT!
                auto [U, S, VT] = do_svd_ptr(tensor.data(), dL * chiL, dR * chiR, svd_cfg);
                return std::make_tuple(tenx::TensorMap(U, dL, chiL, S.size()), tenx::TensorMap(S, S.size()), tenx::TensorMap(VT, 1, S.size(), chiR));

            } else {
                Eigen::Tensor<Scalar, 4> tensor4 = tensor.reshape(tenx::array4{dL, dR, chiL, chiR}).shuffle(tenx::array4{0, 2, 1, 3});
                return decompose(tensor4, svd_cfg);
            }
        }

        template<typename Derived>
        std::tuple<MatrixType<typename Derived::Scalar>, VectorType<typename Derived::Scalar>, MatrixType<typename Derived::Scalar>>
            decompose(const Eigen::DenseBase<Derived> &matrix, const svd::config &svd_cfg = svd::config()) {
            auto [U, S, VT] = do_svd_ptr(matrix.derived().data(), matrix.rows(), matrix.cols(), svd_cfg);
            return std::make_tuple(U, S.template cast<typename Derived::Scalar>(), VT);
        }

        template<typename Scalar, auto N>
        std::tuple<Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 3>>
            schmidt(const Eigen::Tensor<Scalar, N> &tensor, long dL, long chiL, long dR, long chiR, const svd::config &svd_cfg = svd::config()) {
            /*
             * When using this function, it is important that the index order is correct
             * The order is
             * 1) left physical index
             * 2) left bond index
             * 3) right physical index
             * 4) right bond index
             *
             * It is assumed that this order is built into tensor already. Hard to debug errors will occurr
             * if this is not the case!
             * The function call argument order dL, chiL, dR,chiR is meant as a hint for how to use this function.
             */
            if(dL * chiL * dR * chiR != tensor.size()) throw std::range_error("schmidt error: tensor size does not match given dimensions.");
            auto [U, S, VT] = do_svd_ptr(tensor.data(), dL * chiL, dR * chiR, svd_cfg);
            return std::make_tuple(tenx::TensorMap(U, dL, chiL, S.size()), tenx::TensorMap(S.normalized().template cast<Scalar>(), S.size()),
                                   tenx::TensorMap(VT, S.size(), dR, chiR).shuffle(tenx::array3{1, 0, 2}));
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 3>> schmidt(const Eigen::Tensor<Scalar, 4> &tensor,
                                                                                                         const svd::config &svd_cfg = svd::config()) {
            long dL   = tensor.dimension(0);
            long chiL = tensor.dimension(1);
            long dR   = tensor.dimension(2);
            long chiR = tensor.dimension(3);
            if(dL * chiL * dR * chiR != tensor.size()) throw std::range_error("schmidt error: tensor size does not match given dimensions.");
            auto [U, S, VT] = do_svd_ptr(tensor.data(), dL * chiL, dR * chiR, svd_cfg);
            return std::make_tuple(tenx::TensorMap(U, dL, chiL, S.size()), tenx::TensorMap(S.normalized().template cast<Scalar>(), S.size()),
                                   tenx::TensorMap(VT, S.size(), dR, chiR).shuffle(tenx::array3{1, 0, 2}));
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 3>>
            schmidt_multisite(const Eigen::Tensor<Scalar, 3> &tensor, long dL, long dR, long chiL, long chiR, const svd::config &svd_cfg = svd::config()) {
            /* This function assumes that the tensor is given with indices in multisite standard form, i.e., with order
             *
             * 1) physical indices (any number of them, contracted from left to right)
             * 2) left bond index
             * 4) right bond index
             *
             * (1)chiL ---[tensor]--- (2)chiR
             *               |
             *          (0)d*d*d...
             *
             * we start by transposing the tensor into "left-right" order suitable for schmidt decomposition
             *
             * (1)chiL---[  tensor  ]---(3)chiR
             *            |        |
             *        (0)dL      (2)dR
             *
             * The function call argument order dL, dR, chiL,chiR is meant as a hint for how to use this function.
             *
             */
            if(dR == 1) { // probably a single-site right-moving split. No need for a shuffle on VT!
                auto [U, S, VT] = do_svd_ptr(tensor.data(), dL * chiL, dR * chiR, svd_cfg);
                return std::make_tuple(tenx::TensorMap(U, dL, chiL, S.size()), tenx::TensorMap(S.normalized().template cast<Scalar>(), S.size()),
                                       tenx::TensorMap(VT, 1, S.size(), chiR));

            } else {
                Eigen::Tensor<Scalar, 4> tensor_for_schmidt = tensor.reshape(tenx::array4{dL, dR, chiL, chiR}).shuffle(tenx::array4{0, 2, 1, 3});
                return schmidt(tensor_for_schmidt, dL, chiL, dR, chiR, svd_cfg);
            }
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 3>>
            schmidt_into_right_normalized(const Eigen::Tensor<Scalar, 3> &tensor, long dR, const svd::config &svd_cfg = svd::config()) {
            /* This schmidt decomposition is used to pull a site out of rank-3 tensor from the right
             * We obtain USV matrices from a tensor containing N sites with spin dim = d in two steps:
             *
             * (1)chiL ---[mps]--- (2)chiR
             *             |
             *        (0) dL*dR
             *
             * where the index order is in parentheses. This is is reinterpreted as
             *
             *
             * (2)chiL---[    mps    ]---(3)chiR
             *            |         |
             *  (0)dL=d^(N-1)   (1)dR=d
             *
             * and reshuffled into
             *
             * (1)chiL---[    mps    ]---(3)chiR
             *            |         |
             *  (0)dL=d^(N-1)   (2)dR=d
             *
             * which then is split up into
             *
             * chiL ---[U]----chi chi---[S]---chi chi---[V]---chiR
             *          |                                |
             *     d^(N-1)=dL                           d=dR
             *
             * Here V is an "M" matrix of type B
             *
             */
            if(tensor.dimension(0) % dR != 0) throw std::runtime_error("Tensor dim 0 is not divisible by the given spin dimension " + std::to_string(dR));

            long dLdR = tensor.dimension(0);
            long dL   = dLdR / dR;
            long chiL = tensor.dimension(1);
            long chiR = tensor.dimension(2);
            return schmidt_multisite(tensor, dL, dR, chiL, chiR, svd_cfg);
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 3>>
            schmidt_into_left_normalized(const Eigen::Tensor<Scalar, 3> &tensor, long dL, const svd::config &svd_cfg = svd::config()) {
            /* This schmidt decomposition is used to pull a site out of rank-3 tensor from the left
             * We obtain USV matrices from a tensor containing N sites with spin dim = d in two steps:
             *
             * (1)chiL ---[mps]--- (2)chiR
             *             |
             *        (0) dL*dR
             *
             * where the index order is in parentheses. This is reinterpreted as
             *
             *
             * (2)chiL---[    mps    ]---(3)chiR
             *            |         |
             *        (0)dL=d  (1)dR=d^(N-1)
             *
             * and reshuffled into
             *
             * (1)chiL---[    mps    ]---(3)chiR
             *            |         |
             *        (0)dL=d   (2)dR=d^(N-1)
             *
             * which then is split up into
             *
             * chiL ---[U]----chi chi---[S]---chi chi---[V]---chiR
             *          |                                |
             *        dL=d                            dR=d^(N-1)
             *
             * Here U is an "M" matrix of type A
             * Here V is an "M" matrix of type B
             *
             */
            if(tensor.dimension(0) % dL != 0) throw std::runtime_error("Tensor dim 0 is not divisible by the given spin dimension " + std::to_string(dL));

            long dLdR = tensor.dimension(0);
            long dR   = dLdR / dL;
            long chiL = tensor.dimension(1);
            long chiR = tensor.dimension(2);
            return schmidt_multisite(tensor, dL, dR, chiL, chiR, svd_cfg);
        }

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 2>> split_mpo_l2r(const Eigen::Tensor<Scalar, 4> &mpo,
                                                                                     const svd::config              &svd_cfg = svd::config());
        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 4>> split_mpo_r2l(const Eigen::Tensor<Scalar, 4> &mpo,
                                                                                     const svd::config              &svd_cfg = svd::config());

        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 2>> invert_mpo_l2r(const Eigen::Tensor<Scalar, 4> &mpo,
                                                                                      const svd::config              &svd_cfg = svd::config());
        template<typename Scalar>
        std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 4>> invert_mpo_r2l(const Eigen::Tensor<Scalar, 4> &mpo,
                                                                                      const svd::config              &svd_cfg = svd::config());

        template<typename Scalar>
        std::pair<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 4>> split_mpo_pair(const Eigen::Tensor<Scalar, 6> &mpo,
                                                                                     const svd::config              &svd_cfg = svd::config()) {
            /*
             * Splits a pair of merged MPO's back into two MPO's.
             *
             *
             *         (1)dL     (4)dR                             (1)dL                                      (2)dR
             *            |        |                                 |                                          |
             *   (0)mL---[  mpoLR   ]---(3)mR     --->    (0)mL---[ mpoL ]---mC(3)  (0)---[S]---(1)  mC(0)---[ mpoR ]---(1)mR
             *            |        |                                 |                                          |
             *         (2)dL     (5)dR                            (2)dL                                       (3)dR
             *
             *
             * The square root of S can then be multiplied into both left and right MPO's, on the mC index.
             * The left mpo can be shuffled back to standard form with
             *   mpoL: shuffle(0,3,1,2)
             *
             */
            auto rows = mpo.dimension(0) * mpo.dimension(1) * mpo.dimension(2);
            auto cols = mpo.dimension(3) * mpo.dimension(4) * mpo.dimension(5);

            auto mL         = mpo.dimension(0);
            auto mR         = mpo.dimension(3);
            auto dL         = mpo.dimension(1);
            auto dR         = mpo.dimension(4);
            auto [U, S, VT] = do_svd_ptr(mpo.data(), rows, cols, svd_cfg);
            auto mC         = S.size();
            S               = S.cwiseSqrt();
            U               = U * S.asDiagonal();
            VT              = S.asDiagonal() * VT;

            return std::make_pair(tenx::TensorMap(U).reshape(tenx::array4{mL, dL, dL, mC}).shuffle(tenx::array4{0, 3, 1, 2}),
                                  tenx::TensorMap(VT).reshape(tenx::array4{mC, mR, dR, dR}));
        }

        template<typename Scalar>
        std::pair<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>> split_mpo_gate(const Eigen::Tensor<Scalar, 5> &gate,
                                                                                     const svd::config              &svd_cfg = svd::config()) {
            /*
             * Splits an MPO out from a 2-site gate.
             *
             *
             *         (2)dL    (4)dR                             (2)dL                                      (2)dR
             *            |       |                                 |                                          |
             *   (0)mL---[  gate   ]          --->       (0)mL---[ mpoL ]---mC(3)  (0)---[S]---(1)  mC(0)---[ gate ]
             *            |       |                                 |                                          |
             *         (1)dL    (3)dR                            (1)dL                                       (1)dR
             *
             * On the left side, the index 0 is either a dummy or a trailing index of the gate -> mpo process.
             * The square root of S can then be multiplied both left and right, on the mC index.
             * The left mpo can be shuffled back to standard form with
             *   mpoL: shuffle(0,3,2,1)
             *
             */
            auto mL         = gate.dimension(0);
            auto dLdn       = gate.dimension(1);
            auto dLup       = gate.dimension(2);
            auto dRdn       = gate.dimension(3);
            auto dRup       = gate.dimension(4);
            auto rows       = mL * dLdn * dLup;
            auto cols       = dRdn * dRup;
            auto [U, S, VT] = do_svd_ptr(gate.data(), rows, cols, svd_cfg);
            auto mC         = S.size();

            S  = S.cwiseSqrt();
            U  = U * S.asDiagonal();
            VT = S.asDiagonal() * VT;

            return std::make_pair(tenx::TensorMap(U).reshape(tenx::array4{mL, dLup, dLdn, mC}).shuffle(tenx::array4{0, 3, 2, 1}),
                                  tenx::TensorMap(VT).reshape(tenx::array3{mC, dRup, dRdn}));
        }
    };
}
