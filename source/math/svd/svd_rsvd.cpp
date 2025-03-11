#include "math/svd.h"
#include "debug/exceptions.h"
#include "math/rnd.h"
#include "rsvd/Constants.hpp"
#include "rsvd/ErrorEstimators.hpp"
#include "rsvd/RandomizedSvd.hpp"
#include "tid/tid.h"
#include <Eigen/Dense>
#include <general/sfinae.h>

/*! \brief Performs randomized SVD on a matrix
 */
template<typename Scalar>
std::tuple<svd::MatrixType<Scalar>, svd::VectorType<Scalar>, svd::MatrixType<Scalar>>
    svd::solver::do_svd_rsvd([[maybe_unused]] const Scalar *mat_ptr, [[maybe_unused]] long rows, [[maybe_unused]] long cols) const {
    auto t_rsvd   = tid::tic_scope("rsvd");
    long rank_lim = rank_max > 0 ? std::min(std::min(rows, cols), rank_max) : std::min(rows, cols);
    if(rank_lim <= 0) throw std::logic_error("rank_lim <= 0");
    if(rows <= 0) throw except::runtime_error("SVD error: rows = {}", rows);
    if(cols <= 0) throw except::runtime_error("SVD error: cols = {}", cols);
    auto mat = Eigen::Map<const MatrixType<Scalar>>(mat_ptr, rows, cols);
    if constexpr(tenx::sfinae::is_std_complex_v<Scalar>) {
        if(tenx::isReal(mat)) {
            svd::MatrixType<RealType<Scalar>> matreal = mat.real();
            auto [U, S, V]                            = do_svd_rsvd(matreal.data(), rows, cols);
            return std::make_tuple(U.template cast<Scalar>(), S.template cast<Scalar>(), V.template cast<Scalar>());
        }
    }

#if !defined(NDEBUG)
    // These are more expensive debugging operations
    if(not mat.allFinite()) throw std::runtime_error("SVD error: matrix has inf's or nan's");
    if(mat.isZero(0)) throw std::runtime_error("SVD error: matrix is all zeros");
    if(mat.isZero()) log->warn("Lapacke SVD Warning\n\t Given matrix elements are all close to zero (prec 1e-12)");
#endif

    // Randomized SVD
    //    std::mt19937_64 randomEngine{};
    //    randomEngine.seed(777);
    //    rnd::

    Rsvd::RandomizedSvd<MatrixType<Scalar>, pcg64, Rsvd::SubspaceIterationConditioner::None> SVD(rnd::internal::rng64);
    //    Rsvd::RandomizedSvd<MatrixType<Scalar>, pcg64, Rsvd::SubspaceIterationConditioner::Mgs> SVD(rnd::internal::rng);

    log->debug("Running RSVD | {} x {} | truncation limit {:.4e} | rank_lim {}", rows, cols, truncation_lim, rank_lim);
    // Run the svd
    SVD.compute(mat, rank_lim, 2, 2);

    // Truncation error needs normalized singular values
    std::tie(rank, truncation_error) = get_rank_from_truncation_error(SVD.singularValues().col(0));

    if(rank == 0 or not SVD.matrixU().leftCols(rank).allFinite() or not SVD.singularValues().topRows(rank).allFinite() or
       not SVD.matrixV().leftCols(rank).allFinite()) {
        throw except::runtime_error("RSVD SVD error \n"
                                    "  Truncation Error = {:.4e}\n"
                                    "  Rank             = {}\n"
                                    "  Dims             = ({}, {})\n"
                                    "  A all finite     : {}\n"
                                    "  U all finite     : {}\n"
                                    "  S all finite     : {}\n"
                                    "  V all finite     : {}\n",
                                    truncation_error, rank, rows, cols, mat.allFinite(), SVD.matrixU().leftCols(rank).allFinite(),
                                    SVD.singularValues().topRows(rank).allFinite(), SVD.matrixV().leftCols(rank).allFinite());
    }
    log->trace("SVD with RND SVD finished successfully | rank {:<4} | rank_lim {:<4} | {:>4} x {:<4} | trunc {:8.2e}, time {:8.2e}", rank, rank_lim, rows, cols,
               truncation_error, t_rsvd->get_last_interval());
    // Not all calls to do_svd need normalized S, so we do not normalize here!
    return std::make_tuple(SVD.matrixU().leftCols(rank), SVD.singularValues().topRows(rank), SVD.matrixV().leftCols(rank).adjoint());
}

template std::tuple<svd::MatrixType<fp32>, svd::VectorType<fp32>, svd::MatrixType<fp32>> svd::solver::do_svd_rsvd(const fp32 *, long, long) const;
template std::tuple<svd::MatrixType<fp64>, svd::VectorType<fp64>, svd::MatrixType<fp64>> svd::solver::do_svd_rsvd(const fp64 *, long, long) const;
template std::tuple<svd::MatrixType<cx32>, svd::VectorType<cx32>, svd::MatrixType<cx32>> svd::solver::do_svd_rsvd(const cx32 *, long, long) const;
template std::tuple<svd::MatrixType<cx64>, svd::VectorType<cx64>, svd::MatrixType<cx64>> svd::solver::do_svd_rsvd(const cx64 *, long, long) const;