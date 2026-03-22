#pragma once
#include "../svd.h"
#include "lapack_wrappers/lapack_wrappers.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "tid/tid.h"
#include <complex>
#include <csignal>
#include <Eigen/Core>
#include <fmt/ranges.h>

namespace svd {
#if defined(NDEBUG)
    static constexpr bool ndebug = true;
#else
    static constexpr bool ndebug = false;
#endif
}

// namespace svd {
//
//     namespace internal {
//         // These are workspace arrays used by LAPACK which can be reused for the duration of the program.
//         // Call clear() to recover the memory space
//         std::vector<int>                  iwork;
//         std::vector<std::complex<double>> cwork;
//         std::vector<double>               rwork;
//         void                              clear_lapack() {
//             iwork = std::vector<int>();
//             cwork = std::vector<std::complex<double>>();
//             rwork = std::vector<double>();
//             iwork.shrink_to_fit();
//             cwork.shrink_to_fit();
//             rwork.shrink_to_fit();
//         }
//     }
// }

template<typename T>
std::vector<long> get_valid_rows(const Eigen::MatrixBase<T> &m) {
    std::vector<long> v;
    v.reserve(static_cast<size_t>(m.rows()));
    for(long col = 0; col < 1; ++col) { // Cheat by checking the first column only
        for(long row = 0; row < m.rows(); ++row) {
            auto val = m(row, col);
            if constexpr(std::is_arithmetic_v<typename T::Scalar>) {
                if(!std::isnan(val) && !std::isinf(val)) v.emplace_back(row);
            } else {
                if(!std::isnan(std::real(val)) && !std::isinf(std::real(val)) && !std::isnan(std::imag(val)) && !std::isinf(std::imag(val)))
                    v.emplace_back(row);
            }
        }
    }
    return v;
}
template<typename T>
std::vector<long> get_valid_cols(const Eigen::MatrixBase<T> &m) {
    std::vector<long> v;
    v.reserve(static_cast<size_t>(m.cols()));
    for(long col = 0; col < m.cols(); ++col) {
        if(m.col(col).allFinite()) { v.emplace_back(col); }
    }
    return v;
}

template<typename Scalar>
std::tuple<svd::MatrixType<Scalar>, svd::VectorType<Scalar>, svd::MatrixType<Scalar>> svd::solver::do_svd_lapacke(const Scalar *mat_ptr, long rows,
                                                                                                                  long cols) const {
    static_assert(svd::internal::lapack_wrappers::lapacke_scalar<Scalar>,
                  "svd::solver::do_svd_lapacke requires a LAPACKE-backed scalar type");

    // Setup useful sizes
    int rowsA = safe_cast<int>(rows);
    int colsA = safe_cast<int>(cols);
    int sizeS = std::min(rowsA, colsA);
    // Sanity checks
    assert(rows > 0);
    assert(cols > 0);
    assert(sizeS > 0);

    if(rows < cols and (svd_rtn == rtn::gejsv or svd_rtn == rtn::gesvj)) {
        // The jacobi routines needs a tall matrix
        //        auto t_adj = tid::tic_token("adjoint", tid::highest);
        log->trace("Transposing {}x{} into a tall matrix {}x{}", rows, cols, cols, rows);
        // MatrixType<Scalar> A = Eigen::Map<const MatrixType<Scalar>>(mat_ptr, rows, cols);
        // A.adjointInPlace(); // Adjoint directly on a map seems to give a bug?
        auto               Amap = Eigen::Map<const MatrixType<Scalar>>(mat_ptr, rows, cols);
        MatrixType<Scalar> A    = MatrixType<Scalar>(Amap).adjoint().eval();
        // Sanity checks
        assert(A.rows() > 0);
        assert(A.cols() > 0);

        //        t_adj.toc();
        auto [U, S, VT] = do_svd_lapacke(A.data(), A.rows(), A.cols());
        assert(U.rows() == A.rows());
        assert(VT.cols() == A.cols());
        return std::make_tuple(VT.adjoint(), S, U.adjoint());
    }
    //    auto t_lpk = tid::tic_scope("lapacke", tid::highest);

    MatrixType<Scalar> A    = Eigen::Map<const MatrixType<Scalar>>(mat_ptr, rows, cols); // gets destroyed in some routines
    auto               dump = internal::DumpSVD<Scalar>();
    dump.svd_save           = svd_save;
    if(dump.svd_save != svd::save::NONE) dump.A = A;
    //    saveMetaData.svd_is_running = true; // TODO: REMOVE THIS LINE! We don't really want to save it every time!!
    //    saveMetaData.svd_save = save::ALL;  // TODO: REMOVE THIS LINE! We don't really want to save it every time!!
    //    saveMetaData.A = A; // TODO: REMOVE THIS LINE! We don't really want to save it every time!!
    // Initialize containers
    MatrixType<Scalar>             U;
    VectorType<RealScalar<Scalar>> S;
    MatrixType<Scalar>             V;
    MatrixType<Scalar>             VT;
    log->trace("Starting SVD with lapacke | rows {} | cols {}", rows, cols);

    int info   = 0;
    int rowsU  = rowsA;
    int colsU  = std::min(rowsA, colsA);
    int rowsVT = std::min(rowsA, colsA);
    int colsVT = colsA;
    int rowsV  = colsA;
    int colsV  = std::min(rowsA, colsA);
    int lda    = rowsA;
    int ldu    = rowsU;
    int ldvt   = rowsVT;
    int ldv    = rowsV;

    int         mx = std::max(rowsA, colsA);
    int         mn = std::min(rowsA, colsA);
    std::string errmsg;
    //    auto t_pre = tid::tic_scope("preamble", tid::highest);
    long nonzeros = 0;
    try {
        // Sanity checks
        if constexpr(!ndebug) { // We usually get a negative "info" value if there are nans anyway.
            if(A.isZero()) log->warn("Lapacke SVD: A is a zero matrix");
            if(not A.allFinite()) {
                print_matrix(A.data(), A.rows(), A.cols(), "A");
                throw std::runtime_error("A has inf's or nan's");
            }
        }

        using namespace svd::internal;
        using namespace svd::internal::lapack_wrappers;

        thread_local Workspace<Scalar> workspace;
        Context<Scalar>   ctx(A, U, S, V, VT, workspace, rowsA, colsA, sizeS, rowsU, colsU, rowsVT, colsVT, rowsV, colsV, lda, ldu, ldvt, ldv, mx, mn,
                            rank_max, truncation_lim, svdx_select);

        if constexpr(!ndebug)
            log->debug("Running Lapacke {}{} | truncation limit {:.4e} | switchsize bdc {} | size {}", type_prefix<Scalar>(), enum2sv(svd_rtn),
                       truncation_lim, switchsize_gesdd, sizeS);

        if constexpr(std::is_same_v<Scalar, fp32>) {
            switch(svd_rtn) {
                case rtn::gesvd: info = sgesvd(ctx); break;
                case rtn::gesvj: info = sgesvj(ctx); break;
                case rtn::gejsv: info = sgejsv(ctx); break;
                case rtn::gesdd: info = sgesdd(ctx); break;
                case rtn::gesvdx: info = sgesvdx(ctx); break;
                default: throw std::logic_error("invalid case for enum svd::rtn");
            }
        } else if constexpr(std::is_same_v<Scalar, fp64>) {
            switch(svd_rtn) {
                case rtn::gesvd: info = dgesvd(ctx); break;
                case rtn::gesvj: info = dgesvj(ctx); break;
                case rtn::gejsv: info = dgejsv(ctx); break;
                case rtn::gesdd: info = dgesdd(ctx); break;
                case rtn::gesvdx: info = dgesvdx(ctx); break;
                default: throw std::logic_error("invalid case for enum svd::rtn");
            }
        } else if constexpr(std::is_same_v<Scalar, cx32>) {
            switch(svd_rtn) {
                case rtn::gesvd: info = cgesvd(ctx); break;
                case rtn::gesvj: info = cgesvj(ctx); break;
                case rtn::gejsv: info = cgejsv(ctx); break;
                case rtn::gesdd: info = cgesdd(ctx); break;
                case rtn::gesvdx: info = cgesvdx(ctx); break;
                default: throw std::logic_error("invalid case for enum svd::rtn");
            }
        } else if constexpr(std::is_same_v<Scalar, cx64>) {
            switch(svd_rtn) {
                case rtn::gesvd: info = zgesvd(ctx); break;
                case rtn::gesvj: info = zgesvj(ctx); break;
                case rtn::gejsv: info = zgejsv(ctx); break;
                case rtn::gesdd: info = zgesdd(ctx); break;
                case rtn::gesvdx: info = zgesvdx(ctx); break;
                default: throw std::logic_error("invalid case for enum svd::rtn");
            }
        }

        if(info < 0) throw except::runtime_error("{}{} error: parameter {} is invalid", error_prefix<Scalar>(), enum2sv(svd_rtn), info);
        if(info > 0) throw except::runtime_error("{}{} error: could not converge: info {}", error_prefix<Scalar>(), enum2sv(svd_rtn), info);

        log->trace("Truncating singular values");
        std::tie(rank, truncation_error) = get_rank_from_truncation_error(S);
        // Do the truncation

        U.conservativeResize(Eigen::NoChange, rank);
        S.conservativeResize(rank);
        VT.conservativeResize(rank, Eigen::NoChange);

        // We may need to prune further if there are nan row/cols. This is an easy way to recover from silent non-convergence that would
        // inject nans otherwise.
        // We do this because often the Eigen implementation will crash with segmentation fault on this type of error
        auto valid_rows = get_valid_rows(VT); // It's usually sufficient to check just one column in VT, since the same cols of U would also be affected
        if(static_cast<size_t>(rank) != valid_rows.size()) {
            // We have a choice. If the problem size is huge, Jacobi is too expensive. It may just be better to salvage this result.
            // If the problem size is small, we can let the Jacobi solvers try instead.
            if(sizeS <= 1024) {
                throw except::runtime_error("Detected non-finite rows/cols in U, S or VT. The problem size is small ({}). Try another solver.", sizeS);
            }
            log->warn("Pruning non-finite rows/cols from the results! rank {} -> {}", rank, valid_rows.size());
            log->debug("valid rows: {}", valid_rows);
            U    = U(Eigen::placeholders::all, valid_rows);
            S    = S(valid_rows);
            VT   = VT(valid_rows, Eigen::placeholders::all);
            rank = S.size();
        }

        // Sanity checks
        if constexpr(!ndebug) {
            bool uerr = !U.allFinite();
            bool serr = !S.allFinite() or (S.array() <= 0).any();
            bool verr = !VT.allFinite();
            if(uerr or serr or verr) { // Will only print if !ndebug
                print_matrix(U.data(), U.rows(), U.cols(), "U");
                print_matrix(VT.data(), VT.rows(), VT.cols(), "VT");
                print_vector(S.data(), S.size(), "S", 16);
            }
            if(uerr) errmsg += "U has inf's or nan's";
            if(serr) errmsg += "S has inf's or nan's or non-positive values";
            if(verr) errmsg += "VT has inf's or nan's";
            if(!errmsg.empty()) throw except::runtime_error(errmsg);
        }
    } catch(const except::runtime_error &ex) {
        if(dump.svd_save == svd::save::FAIL) {
            dump.U                = U;
            dump.S                = S;
            dump.VT               = VT;
            dump.rank             = rank;
            dump.truncation_error = truncation_error;
            dump.info             = info;
        }
        throw except::runtime_error("Lapacke SVD error \n"
                                    "  Singular values  = {::.5e}\n"
                                    "  Truncation Error = {:.4e}\n"
                                    "  Rank             = {}\n"
                                    "  Dims             = ({}, {})\n"
                                    "  Lapacke info     : {}\n"
                                    "  Error message    : {}\n",
                                    fv(S), truncation_error, rank, rows, cols, info, ex.what());
    }
    if(dump.svd_save == svd::save::ALL or dump.svd_save == svd::save::LAST) {
        dump.U                = U;
        dump.S                = S;
        dump.VT               = VT;
        dump.rank             = rank;
        dump.truncation_error = truncation_error;
        dump.info             = info;
    }
    if(log->level() == spdlog::level::trace)
        log->trace("SVD with Lapacke finished successfully | truncation limit {:<8.2e} | rank {:<4} | nonzeros {:<4} | rank_max {:<4} | {:>4} x {:<4} | trunc "
                   "{:8.2e}, time {:8.2e}",
                   truncation_lim, rank, nonzeros, rank_max, rows, cols, truncation_error, 0.0
                   //            t_lpk->get_last_interval()
        );
    return std::make_tuple(U, S, VT);
}
