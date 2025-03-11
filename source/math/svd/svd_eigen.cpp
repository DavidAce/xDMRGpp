#include <complex.h>
#undef I

#include "../cast.h"
#include "../svd.h"
#include "debug/exceptions.h"
#include "tid/tid.h"
#include <Eigen/QR>
#include <Eigen/SVD>

namespace svd {
#if defined(NDEBUG)
    static constexpr bool ndebug = true;
#else
    static constexpr bool ndebug = false;
#endif
}

/*! \brief Performs SVD on a matrix
 *  This function is defined in cpp to avoid long compilation times when having Eigen::BDCSVD included everywhere in headers.
 *  Performs rigorous checks to ensure stability of DMRG.
 *  In some cases Eigen::BCDSVD/JacobiSVD will fail with segfault. Here we use a patched version of Eigen that throws an error
 *  instead so we get a chance to catch it and use lapack svd instead.
 *   \param mat_ptr Pointer to the matrix. Supported are double * and std::complex<double> *
 *   \param rows Rows of the matrix
 *   \param cols Columns of the matrix
 *   \return The U, S, and V matrices (with S as a vector) extracted from the Eigen::BCDSVD SVD object.
 */
template<typename Scalar>
std::tuple<svd::MatrixType<Scalar>, svd::VectorType<Scalar>, svd::MatrixType<Scalar>> svd::solver::do_svd_eigen(const Scalar *mat_ptr, long rows,
                                                                                                                long cols) const {
    //    auto t_eigen = tid::tic_scope("eigen", tid::highest);
    log->trace("Starting SVD with Eigen");
    auto                                 minRC = std::min(rows, cols);
    Eigen::Map<const MatrixType<Scalar>> mat(mat_ptr, rows, cols);

    if(rows <= 0) throw except::runtime_error("SVD error: rows = {}", rows);
    if(cols <= 0) throw except::runtime_error("SVD error: cols = {}", cols);

    if constexpr(!ndebug) {
        // These are more expensive debugging operations
        if(not mat.allFinite()) throw std::runtime_error("SVD error: matrix has inf's or nan's");
        if(mat.isZero(0)) throw std::runtime_error("SVD error: matrix is all zeros");
        if(mat.isZero()) log->warn("Lapacke SVD Warning\n\t Given matrix elements are all close to zero");
    }
    if(svd_save != svd::save::NONE) save_svd(MatrixType<Scalar>(mat));
    if(svd_save != svd::save::FAIL) saveMetaData.A = MatrixType<Scalar>(mat);

    Eigen::BDCSVD<MatrixType<Scalar>> SVD;

    // Set up the SVD solver
    if(switchsize_gesdd == -1ul) {
        SVD.setSwitchSize(safe_cast<int>(minRC));
    } else {
        SVD.setSwitchSize(safe_cast<int>(switchsize_gesdd));
    }

    // Add suffix for more detailed breakdown of matrix sizes
    auto t_suffix = benchmark ? fmt::format("{}", num::next_multiple<long>(minRC, 5l)) : "";
    auto svd_info =
        fmt::format("| {} x {} | rank_max {} | truncation limit {:.4e} | switchsize bdc {}", rows, cols, rank_max, truncation_lim, switchsize_gesdd);
    bool use_jacobi = minRC < safe_cast<long>(switchsize_gesdd);
    if(use_jacobi or svd_rtn == rtn::gejsv or svd_rtn == svd::rtn::gesvj) {
        // We only use Jacobi for precision. So we use all the precision we can get.
        log->debug("Running Eigen::JacobiSVD {}", svd_info);
        // Run the svd
        //        auto t_jcb = tid::tic_token(fmt::format("jcb{}", t_suffix), tid::highest);
        SVD.compute(mat, Eigen::ComputeFullU | Eigen::ComputeFullV | Eigen::FullPivHouseholderQRPreconditioner);
    } else {
        log->debug("Running Eigen::BDCSVD {}", svd_info);
        // Run the svd
        //        auto t_bdc = tid::tic_token(fmt::format("bdc{}", t_suffix), tid::highest);
        SVD.compute(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }

    long rank_lim   = rank_max > 0 ? std::min(minRC, rank_max) : minRC;
    rank            = SVD.nonzeroSingularValues();
    long max_size   = std::min(rank, rank_lim);
    bool U_finite   = SVD.matrixU().leftCols(max_size).allFinite();
    bool S_finite   = SVD.singularValues().head(max_size).allFinite();
    bool V_finite   = SVD.matrixV().leftCols(max_size).allFinite();
    bool S_positive = (SVD.singularValues().head(max_size).array() >= 0).all();

    if(SVD.rank() <= 0 or rank == 0 or not U_finite or not S_finite or not S_positive or not V_finite) {
        if(svd_save == svd::save::FAIL) {
            saveMetaData.U                = SVD.matrixU();
            saveMetaData.S                = SVD.singularValues();
            saveMetaData.VT               = MatrixType<Scalar>(SVD.matrixV().adjoint());
            saveMetaData.rank             = rank;
            saveMetaData.truncation_error = truncation_error;
            saveMetaData.info             = -1;
            save_svd();
        }
        if(not mat.allFinite()) {
            print_matrix(mat.data(), mat.rows(), mat.cols(), "A");
            log->critical("Eigen SVD error: matrix has inf's or nan's");
        }
        if(mat.isZero()) {
            print_matrix(mat.data(), mat.rows(), mat.cols(), "A", 16);
            log->critical("Eigen SVD error: matrix is all zeros");
        }
        if(not S_positive) {
            print_vector(SVD.singularValues().head(rank).data(), rank, "S", 16);
            log->critical("Eigen SVD error: S is not positive");
        }
        // #if !defined(NDEBUG)
        save_svd<Scalar>(SVD.matrixU(), SVD.singularValues(), SVD.matrixV().adjoint(), -1);

        // #endif

        throw except::runtime_error("Eigen SVD error \n"
                                    "  Rank max         = {}\n"
                                    "  Dimensions       = ({}, {})\n"
                                    "  A all finite     : {}\n"
                                    "  U all finite     : {}\n"
                                    "  S all finite     : {}\n"
                                    "  S all positive   : {}\n"
                                    "  V all finite     : {}\n",
                                    rank_max, rows, cols, mat.allFinite(), U_finite, S_finite, S_positive, V_finite);
    }

    // Truncation error needs normalized singular values
    std::tie(rank, truncation_error) = get_rank_from_truncation_error(SVD.singularValues().head(max_size).normalized());
    if(svd_save == save::ALL or svd_save == save::LAST)
        save_svd<Scalar>(SVD.matrixU().leftCols(rank), SVD.singularValues().head(rank), SVD.matrixV().leftCols(rank).adjoint(), 0);

    log->trace("SVD with Eigen finished successfully");
    saveMetaData = svd::internal::SaveMetaData{}; // Clear
    // Not all calls to do_svd need normalized S, so we do not normalize here!
    return std::make_tuple(SVD.matrixU().leftCols(rank), SVD.singularValues().head(rank), SVD.matrixV().leftCols(rank).adjoint());
}

template std::tuple<svd::MatrixType<fp32>, svd::VectorType<fp32>, svd::MatrixType<fp32>> svd::solver::do_svd_eigen(const fp32 *, long, long) const;
template std::tuple<svd::MatrixType<fp64>, svd::VectorType<fp64>, svd::MatrixType<fp64>> svd::solver::do_svd_eigen(const fp64 *, long, long) const;
template std::tuple<svd::MatrixType<cx32>, svd::VectorType<cx32>, svd::MatrixType<cx32>> svd::solver::do_svd_eigen(const cx32 *, long, long) const;
template std::tuple<svd::MatrixType<cx64>, svd::VectorType<cx64>, svd::MatrixType<cx64>> svd::solver::do_svd_eigen(const cx64 *, long, long) const;