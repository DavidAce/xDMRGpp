#pragma once
#include "math/svd.h"
#include "tid/tid.h"
#include <debug/exceptions.h>
#include <Eigen/QR>
#include <fmt/ranges.h>

#if defined(_OPENMP)

    #include <omp.h>

#endif


/*! \brief Performs SVD on a matrix
 *  This function is defined in cpp to avoid long compilation times when having Eigen::BDCSVD included everywhere in headers.
 *  Performs rigorous checks to ensure stability of DMRG.
 *  In some cases Eigen::BCDSVD/JacobiSVD will fail with segfault. Here we use a patched version of Eigen that throws an error
 *  instead so we get a chance to catch it and use lapack svd instead.
 *   \param mat_ptr Pointer to the matrix. Supported are double * and std::complex<double> *
 *   \param rows Rows of the matrix
 *   \param cols Columns of the matrix
 *   \param svd_cfg Optional overrides to default svd configuration
 *   \return The U, S, and V matrices (with S as a vector) extracted from the Eigen::BCDSVD SVD object.
 */
template<typename Scalar>
std::tuple<svd::MatrixType<Scalar>, svd::VectorType<Scalar>, svd::MatrixType<Scalar>> svd::solver::do_svd_ptr(const Scalar *mat_ptr, long rows, long cols,
                                                                                                              const svd::config &svd_cfg) {
    //    auto t_svd = tid::tic_scope("svd", tid::level::highest);

    copy_config(svd_cfg);
    auto sizeS    = std::min(rows, cols);
    long rank_lim = rank_max > 0 ? std::min(sizeS, rank_max) : sizeS;

    // Resolve geauto
    if(svd_cfg.svd_rtn == svd::rtn::geauto or svd_rtn == svd::rtn::geauto) {
        svd_rtn = svd::rtn::gesvj;
        if(switchsize_gejsv != -1ul and std::cmp_greater_equal(sizeS, switchsize_gejsv)) svd_rtn = svd::rtn::gejsv;
        if(switchsize_gesvd != -1ul and std::cmp_greater_equal(sizeS, switchsize_gesvd)) svd_rtn = svd::rtn::gesvd;
        if(switchsize_gesdd != -1ul and std::cmp_greater_equal(sizeS, switchsize_gesdd)) svd_rtn = svd::rtn::gesdd;

        if(svd_rtn != rtn::gesdd and rows * cols >= 256) svd_rtn = rtn::gesdd; // If it's a large problem, do gesdd anyway.
        if(svd_rtn == rtn::gesdd) {
            bool is_rank_low   = std::cmp_greater_equal(sizeS,
                                                        rank_lim * 8); // Will keep at least 25% of the singular values
            bool is_rank_lower = std::cmp_greater_equal(sizeS,
                                                        rank_lim * 16); // Will keep at least 10% of the singular values
            if(is_rank_low) { svd_rtn = svd::rtn::gesvdx; }
            if(is_rank_lower) { svd_rtn = svd::rtn::gersvd; }
        }
        //        log->info("sizeS = {} | lim {} | {} {} {} | {} ", sizeS, rank_lim, switchsize_gejsv, switchsize_gesvd, switchsize_gesdd,
        //        enum2sv(svd_rtn));
    }
    if constexpr(!sfinae::is_any_v<Scalar, fp32, fp64, cx32, cx64>) svd_lib = svd::lib::eigen; // Eigen handles long double and fp128
#pragma omp atomic
    count++;
    switch(svd_lib) {
        case svd::lib::lapacke: {
            try {
                if(svd_rtn == svd::rtn::gersvd)
                    return do_svd_rsvd(mat_ptr, rows, cols);
                else
                    return do_svd_lapacke(mat_ptr, rows, cols);
            } catch(const std::exception &ex) {
                log->warn("{} {} failed to perform SVD: {} | Trying Lapacke gejsv", enum2sv(svd_lib), enum2sv(svd_rtn), std::string_view(ex.what()));
                try {
                    auto svd_rtn_backup = svd_rtn; // Restore after
                    auto svd_log_level  = log->level();
                    log->set_level(spdlog::level::trace);
                    svd_rtn         = rtn::gejsv;
                    auto [U, S, VT] = do_svd_lapacke(mat_ptr, rows, cols);
                    svd_rtn         = svd_rtn_backup;
                    log->set_level(svd_log_level);
                    return {U, S, VT};
                } catch(const std::exception &ex2) {
                    log->warn("{} {} failed to perform SVD: {} | Trying Eigen JacobiSVD", enum2sv(svd_lib), enum2sv(svd_rtn), std::string_view(ex2.what()));
                    auto svd_rtn_backup = svd_rtn; // Restore after
                    auto svd_log_level  = log->level();
                    log->set_level(spdlog::level::trace);
                    svd_rtn         = rtn::gejsv;
                    auto [U, S, VT] = do_svd_eigen(mat_ptr, rows, cols);
                    svd_rtn         = svd_rtn_backup;
                    log->set_level(svd_log_level);
                    return {U, S, VT};
                }
            }
            break;
        }
        case svd::lib::eigen: {
            try {
                if(svd_rtn == svd::rtn::gersvd)
                    return do_svd_rsvd(mat_ptr, rows, cols);
                else
                    return do_svd_eigen(mat_ptr, rows, cols);
            } catch(const std::exception &ex) {
                if constexpr(tenx::sfinae::is_quadruple_prec_v<Scalar>) {
                    throw except::runtime_error("{} {} failed to perform SVD: {} | No other libraries to try with quadruple precision", enum2sv(svd_lib),
                                                enum2sv(svd_rtn), std::string_view(ex.what()));
                } else {
                    log->warn("{} {} failed to perform SVD: {} | Trying Lapack", enum2sv(svd_lib), enum2sv(svd_rtn), std::string_view(ex.what()));
                    return do_svd_lapacke(mat_ptr, rows, cols);
                }
            }
            break;
        }
        default: throw std::logic_error("Unrecognized svd library");
    }
    throw std::logic_error("Unrecognized svd library");
}


template<typename Scalar>
void svd::solver::print_matrix([[maybe_unused]] const Scalar *mat_ptr, [[maybe_unused]] long rows, [[maybe_unused]] long cols,
                               [[maybe_unused]] std::string_view tag, [[maybe_unused]] long dec) const {
#if !defined(NDEBUG)
    auto A = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(mat_ptr, rows, cols);
    log->warn("Matrix [{}] with dimensions {}x{}\n", tag, rows, cols);
    for(long r = 0; r < A.rows(); r++) {
        for(long c = 0; c < A.cols(); c++) fmt::print("({1:.{0}f}) ", dec, fp(A(r, c)));
        fmt::print("\n");
    }
#endif
}


template<typename Scalar>
void svd::solver::print_vector([[maybe_unused]] const Scalar *vec_ptr, [[maybe_unused]] long size, [[maybe_unused]] std::string_view tag,
                               [[maybe_unused]] long dec) const {
#if !defined(NDEBUG)
    auto V = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(vec_ptr, size);
    log->warn("Vector [{}] with size {}\n", tag, size);
    for(long i = 0; i < V.size(); i++) fmt::print("{1:.{0}f}\n", dec, fp(V[i]));
#endif
}

// template<typename Scalar>
// std::pair<long, fp64> svd::solver::get_rank_from_truncation_error(const VectorType<Scalar> &S) const {
//     VectorType<fp64> truncation_errors(S.size() + 1);
//     for(long s = 0; s <= S.size(); s++) { truncation_errors[s] = S.bottomRows(S.size() - s).norm(); } // Last one should be zero, i.e. no truncation
//     auto rank_    = (truncation_errors.array() >= truncation_lim).count();
//     auto rank_lim = S.size();
//     if(rank_max > 0) rank_lim = std::min(S.size(), rank_max);
//     rank_ = std::min(rank_, rank_lim);
//     if(rank_min > 0) rank_ = std::max(rank_, std::min(S.size(),
//                                                       rank_min)); // Make sure we don't overtruncate in some cases (e.g. when stashing)
//
//     //    tools::log->info("Size {} | Rank {} | Rank limit {} | truncation error limit {:8.5e} | error {:8.5e}", S.size(), rank_, rank_lim,
//     //                     truncation_lim, truncation_errors[rank_]);
//     //    tools::log->info("Size {} | Rank {} | Rank limit {} | truncation error limit {:8.5e} | error {:8.5e} truncation errors: {:8.5e}", S.size(), rank_,
//     //    rank_lim,
//     //                     truncation_lim, truncation_errors[rank_], fmt::join(truncation_errors, ", "));
//     if(rank_ <= 0) {
//         if(log)
//             log->error("Size {} | Rank {} | Rank limit {} | truncation error limit {:8.2e} | error {:8.2e} truncation errors: {:8.2e}", S.size(), rank_,
//                        rank_lim, truncation_lim, truncation_errors[rank_], fmt::join(truncation_errors, ", "));
//         throw std::logic_error("rank <= 0");
//     }
//     return {rank_, truncation_errors[rank_]};
// }

// template std::pair<long, fp64> svd::solver::get_rank_from_truncation_error(const VectorType<fp32> &S) const;
// template std::pair<long, fp64> svd::solver::get_rank_from_truncation_error(const VectorType<fp64> &S) const;
// template std::pair<long, fp64> svd::solver::get_rank_from_truncation_error(const VectorType<cx32> &S) const;
// template std::pair<long, fp64> svd::solver::get_rank_from_truncation_error(const VectorType<cx64> &S) const;

template<typename Scalar>
auto dropfilter(svd::MatrixType<Scalar> &U, svd::VectorType<Scalar> &S, svd::MatrixType<Scalar> &V, double threshold, double min_log_drop_size) {
    long   drop_rank = 0;
    double drop_size = 0;
    for(long idx = 0; idx < S.size(); ++idx) {
        if(idx + 1 < S.size()) drop_size = std::abs(std::abs(std::log10(std::real(S[idx]))) - std::abs(std::log10(std::real(S[idx + 1]))));
        if(std::real(S[idx]) >= threshold and drop_size <= min_log_drop_size) {
            drop_rank = idx + 1; // Keep 0,1,2...idx+1 singular values
        } else
            break;
    }
    // fmt::print("drop_rank: {} | drop_size {}\n", drop_rank, drop_size);
    if(drop_rank > 0) {
        U = U.leftCols(drop_rank).eval();
        S = S.head(drop_rank).eval();
        V = V.topRows(drop_rank).eval();
    }
    return S.size();
}

template<typename Scalar>
std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 2>> svd::solver::split_mpo_l2r(const Eigen::Tensor<Scalar, 4> &mpo, const svd::config &svd_cfg) {
    /*
     * Compress an MPO left to right using SVD as described in https://journals.aps.org/prb/pdf/10.1103/PhysRevB.95.035129
     *
     *          (2) d
     *             |
     *  (0) m ---[mpo]--- (1) m
     *             |
     *         (3) d
     *
     * is shuffled into
     *
     *          (0) d
     *             |
     *  (2) m ---[mpo]--- (3) m
     *             |
     *         (1) d
     *
     * and reshaped like
     *
     * ddm (012) ---[mpo]--- (3) m
     *
     * and subjected to the typical SVD so mpo = USV. This is then reshaped back into
     *
     *            d
     *            |
     *     m ---[mpo]---  m'   m'---[S]---'m m'---[V]---m
     *            |
     *            d
     *
     * where hopefully m' < m and the transfer matrix T = SV is multiplied onto the mpo on the right later.
     *
     * To stablize the compression, it is useful to insert avgS *  1/avgS, where one factor is put into U and the other into S.
     *
     *
     */
    using Real                         = decltype(std::real(std::declval<Scalar>()));
    auto                     dim0      = mpo.dimension(2);
    auto                     dim1      = mpo.dimension(3);
    auto                     dim2      = mpo.dimension(0);
    auto                     dim3      = mpo.dimension(1);
    auto                     dim_ddm   = dim0 * dim1 * dim2;
    Eigen::Tensor<Scalar, 2> mpo_rank2 = mpo.shuffle(tenx::array4{2, 3, 0, 1}).reshape(tenx::array2{dim_ddm, dim3});
    auto [U, S, VT]                    = do_svd_ptr(mpo_rank2.data(), mpo_rank2.dimension(0), mpo_rank2.dimension(1), svd_cfg);
    auto Smin                          = S.real().minCoeff();
    auto Smax                          = S.real().maxCoeff();
    // Stabilize by inserting avgS *  1/avgS
    auto avgS = num::next_power_of_two<Real>(S.head(S.nonZeros()).real().mean()); // Nearest power of two larger than S.mean()
    if(avgS > 1) {
        S /= avgS;
        // std::tie(rank, truncation_error) = get_rank_from_truncation_error(S.head(S.nonZeros()));
        std::tie(rank, truncation_error) = get_rank_from_truncation_error(S);
        U                                = U.leftCols(rank).eval();
        S                                = S.head(rank).eval();
        VT                               = VT.topRows(rank).eval();
        U *= avgS;
    }
    // rank = dropfilter(U, S, V, svd_cfg.truncation_limit.value_or(1e-16), 8);
    VT = S.asDiagonal() * VT; // Rescaled singular values
    fmt::print("S l2r min {:.5e} | max {:.5e} -->  min {:.5e} | max {:.5e} | size {}\n", fp(Smin), fp(Smax), fp(S.real().minCoeff()), fp(S.real().maxCoeff()),
               S.size());
    /* clang-format off */
    return std::make_tuple(
            tenx::TensorMap(U).reshape(tenx::array4{dim0, dim1, dim2, rank}).shuffle(
                    tenx::array4{2, 3, 0, 1}).template cast<Scalar>(),
            tenx::TensorMap(VT).template cast<Scalar>());
    /* clang-format on */
}


template<typename Scalar>
std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 4>> svd::solver::split_mpo_r2l(const Eigen::Tensor<Scalar, 4> &mpo, const svd::config &svd_cfg) {
    /*
     * Splits an MPO right to left using SVD as described in https://journals.aps.org/prb/pdf/10.1103/PhysRevB.95.035129
     *
     *          (2) d
     *             |
     *  (0) m ---[mpo]--- (1) m
     *            |
     *        (3) d
     *
     * is shuffled into
     *
     *          (1) d
     *             |
     *  (0) m ---[mpo]--- (3) m
     *            |
     *        (2) d
     *
     * and reshaped like
     *
     * d (0) ---[mpo]--- (123) ddm
     *
     * and subjected to the typical SVD so mpo = USV. This is then reshaped back into
     *
     *                                          d
     *                                          |
     *   m---[U]---m'   m'---[S]---'m   m' ---[mpo]---  m
     *                                          |
     *                                          d
     *
     * where hopefully m' < m and the transfer matrix T = US is multiplied onto the mpo on the left later.
     *
     * To stablize the compression, it is useful to insert avgS *  1/avgS, where one factor is put into V and the other into S.
     *
     *
     */

    auto dim0    = mpo.dimension(0);
    auto dim1    = mpo.dimension(2);
    auto dim2    = mpo.dimension(3);
    auto dim3    = mpo.dimension(1);
    auto dim_ddm = dim1 * dim2 * dim3;

    Eigen::Tensor<Scalar, 2> mpo_rank2 = mpo.shuffle(tenx::array4{0, 2, 3, 1}).reshape(tenx::array2{dim0, dim_ddm});
    auto [U, S, VT]                    = do_svd_ptr(mpo_rank2.data(), mpo_rank2.dimension(0), mpo_rank2.dimension(1), svd_cfg);

    auto Smin = S.real().minCoeff();
    auto Smax = S.real().maxCoeff();

    // Stabilize by inserting avgS *  1/avgS
    using R   = RealScalar<Scalar>;
    auto avgS = num::next_power_of_two<R>(S.head(S.nonZeros()).real().mean()); // Nearest power of two larger than S.mean()
    if(avgS > 1) {
        S /= avgS;
        std::tie(rank, truncation_error) = get_rank_from_truncation_error(S);
        U                                = U.leftCols(rank).eval();
        S                                = S.head(rank).eval();
        VT                               = VT.topRows(rank).eval();
        VT *= avgS;
    }
    // TRY PRINTING S before rescaling
    fmt::print("S r2l min {:.5e} | max {:.5e} -->  min {:.5e} | max {:.5e} | size {}\n", fp(Smin), fp(Smax), fp(S.real().minCoeff()), fp(S.real().maxCoeff()),
               S.size());
    // rank = dropfilter(U, S, V, svd_cfg.truncation_limit.value_or(1e-16), 8);
    U = U * S.asDiagonal();

    /* clang-format off */
    return std::make_tuple(
            tenx::TensorMap(U).template cast<Scalar>(),
            tenx::TensorMap(VT).reshape(tenx::array4{rank, dim1, dim2, dim3}).shuffle(
                    tenx::array4{0, 3, 1, 2}).template cast<Scalar>());
    /* clang-format on */
}
