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
 *  instead, so we get a chance to catch it and use lapack svd.
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
        if(not mat.allFinite()) {
            print_matrix(mat.data(), mat.rows(), mat.cols(), "A");
            throw std::runtime_error("SVD error: matrix has inf's or nan's");
        }
        if(mat.isZero(0)) throw std::runtime_error("SVD error: matrix is all zeros");
        if(mat.isZero()) log->warn("Lapacke SVD Warning\n\t Given matrix elements are all close to zero");
    }

    auto dump     = internal::DumpSVD<Scalar>();
    dump.svd_save = svd_save;
    if(dump.svd_save != svd::save::NONE) dump.A = mat;

    auto extract_svd = [&dump, this](const auto &SVD, long rank_lim) -> std::tuple<MatrixType<Scalar>, VectorType<Scalar>, MatrixType<Scalar>> {
        long max_size = std::min(SVD.nonzeroSingularValues(), rank_lim);
        // Truncation error needs normalized singular values
        std::tie(rank, truncation_error) = get_rank_from_truncation_error(SVD.singularValues().normalized());

        auto U = SVD.matrixU().leftCols(rank);
        auto S = SVD.singularValues().head(rank);
        auto V = SVD.matrixV().leftCols(rank);

        bool success = SVD.info() == Eigen::ComputationInfo::Success;
        if constexpr(!ndebug) {
            bool U_finite   = U.allFinite();
            bool S_finite   = S.allFinite();
            bool V_finite   = V.allFinite();
            bool S_positive = (SVD.singularValues().head(max_size).array() >= 0).all();
            success         = success and (SVD.rank() > 0 and max_size > 0 and U_finite and S_finite and S_positive and V_finite);
            if(!U_finite) {
                print_matrix(SVD.matrixU().data(), SVD.matrixU().rows(), SVD.matrixU().cols(), "U", 16);
                log->critical("Eigen SVD error: U is not finite");
            }
            if(!S_finite) {
                print_vector(SVD.singularValues().head(rank).data(), rank, "S", 16);
                log->critical("Eigen SVD error: S is not finite");
            }
            if(not S_positive) {
                print_vector(SVD.singularValues().head(rank).data(), rank, "S", 16);
                log->critical("Eigen SVD error: S is not positive");
            }
            if(!V_finite) {
                print_matrix(SVD.matrixV().data(), SVD.matrixV().rows(), SVD.matrixV().cols(), "V", 16);
                log->critical("Eigen SVD error: V is not finite");
            }
        }
        bool do_dump = (!success and dump.svd_save == svd::save::FAIL) or dump.svd_save == svd::save::ALL or dump.svd_save == svd::save::LAST;
        if(do_dump) {
            dump.U                = SVD.matrixU();
            dump.S                = SVD.singularValues();
            dump.VT               = SVD.matrixV().adjoint();
            dump.rank             = rank;
            dump.truncation_error = truncation_error;
            dump.info             = SVD.info();
        }

        if(!success) {
            throw except::runtime_error("Eigen SVD error \n"
                                        "  Rank             = {}\n"
                                        "  Rank max         = {}\n"
                                        "  Dimensions       = ({}, {})\n",
                                        rank, rank_max, SVD.rows(), SVD.cols());
        }
        log->trace("SVD with Eigen finished successfully");

        return std::make_tuple(U, S, V.adjoint());
    };

    // Add suffix for a more detailed breakdown of matrix sizes
    auto t_suffix = benchmark ? fmt::format("{}", num::next_multiple<long>(minRC, 5l)) : "";
    auto svd_info =
        fmt::format("| {} x {} | rank_max {} | truncation limit {:.4e} | switchsize bdc {}", rows, cols, rank_max, truncation_lim, switchsize_gesdd);
    const long switch_eff = switchsize_gesdd == -1ul ? minRC : safe_cast<long>(switchsize_gesdd);
    const bool use_jacobi = minRC < switch_eff;
    const long rank_lim   = rank_max > 0 ? std::min(minRC, rank_max) : minRC;

    if(use_jacobi or svd_rtn == rtn::gejsv or svd_rtn == svd::rtn::gesvj) {
        // We only use Jacobi for precision. So we use all the precision we can get.
        log->debug("Running Eigen::JacobiSVD {}", svd_info);
        auto t_jcb = tid::tic_token(fmt::format("jcb{}", t_suffix), tid::highest);
        auto SVD   = Eigen::JacobiSVD<MatrixType<Scalar>, Eigen::ComputeThinU | Eigen::ComputeThinV | Eigen::ColPivHouseholderQRPreconditioner>();
        // Run the svd
        SVD.compute(mat);
        return extract_svd(SVD, rank_lim);
    } else {
        log->debug("Running Eigen::BDCSVD {}", svd_info);
        auto t_bdc = tid::tic_token(fmt::format("bdc{}", t_suffix), tid::highest);

        Eigen::BDCSVD<MatrixType<Scalar>, Eigen::ComputeThinU | Eigen::ComputeThinV> SVD;

        // Set up
        if(switchsize_gesdd == -1ul) {
            SVD.setSwitchSize(safe_cast<int>(minRC));
        } else {
            SVD.setSwitchSize(safe_cast<int>(switchsize_gesdd));
        }
        // Run the svd
        SVD.compute(mat);
        return extract_svd(SVD, rank_lim);
    }
}
