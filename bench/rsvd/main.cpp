
#include "math/fit.h"
#include "math/linalg/matrix.h"
#include "math/rnd.h"
#include "math/stat.h"
#include "math/svd.h"
#include "math/svd/rsvd/ErrorEstimators.hpp"
#include "math/svd/rsvd/RandomizedRangeFinder.hpp"
#include "math/tenx.h"
#include "tid/tid.h"
#include <h5pp/h5pp.h>
#include <thread>

void bench() {
    svd::config svd_settings;
    svd_settings.svd_save = svd::save::NONE;
    svd_settings.loglevel = 2;
    auto filename         = fmt::format("{}/svd-save.h5", BENCH_DATA_DIR);
    if(h5pp::fs::exists(filename)) {
        h5pp::File h5file(filename, h5pp::FilePermission::READONLY, 2);

        for(const auto &idx : num::range(0, 50)) {
            auto A_dims   = h5file.getDatasetDimensions(fmt::format("svd_{}/A_cplx", idx));
            auto rows     = A_dims.at(0);
            auto cols     = A_dims.at(1);
            auto rank_max = std::min({rows, cols});
            if(rank_max < 1000) continue;
            //            tools::log->info("A_dims: {}", A_dims);
            auto A_file = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("svd_{}/A_cplx", idx));
            //            auto U_file  = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("svd_{}/U_cplx", idx));
            auto S_file = h5file.readDataset<Eigen::VectorXd>(fmt::format("svd_{}/S", idx));
            //            auto VT_file = h5file.readDataset<Eigen::MatrixXcd>(fmt::format("svd_{}/VT_cplx", idx));

            svd_settings.rank_max         = rank_max;
            svd_settings.truncation_limit = 1e-5;

            long   gesddRank  = 0;
            long   gersvdRank = 0;
            double gesddErr   = 0;
            double gesddTrc   = 0;
            double gesddNorm  = 0;
            double gersvdTrc  = 0;
            double gersvdErr  = 0;
            double gersvdNorm = 0;
            {
                svd_settings.svd_lib = svd::lib::lapacke;
                svd_settings.svd_rtn = svd::rtn::gesdd;
                omp_set_num_threads(16);
                svd::solver svd(svd_settings);
                auto        t_lapack = tid::tic_scope("lapack");
                auto [U, S, VT]      = svd.do_svd(A_file);
                t_lapack.toc();
                auto A_gesdd = Eigen::MatrixXcd{U * S.asDiagonal() * VT};
                gesddNorm    = S.norm();
                gesddRank    = svd.get_rank();
                gesddTrc     = svd.get_truncation_error();
                gesddErr     = Rsvd::relativeFrobeniusNormError(A_file, A_gesdd);
            }
            {
                omp_set_num_threads(16);
                svd_settings.svd_lib  = svd::lib::lapacke;
                svd_settings.svd_rtn  = svd::rtn::gersvd;
                svd_settings.rank_max = static_cast<long>(static_cast<double>(rank_max) * 0.10); // TODO: TESTING!
                auto        t_rsvd    = tid::tic_scope("rsvd");
                svd::solver svd(svd_settings);
                auto [U_, S_, VT_]    = svd.do_svd(A_file);
                auto   Y              = Eigen::VectorXd(S_.real().array().log());
                auto   X              = num::range<double>(0, Y.size());
                auto   fit            = fit::log_stretched(X, Y, {1.0, 0.1, 1.0});
                auto   C              = fit.coeffs[0];
                auto   k              = fit.coeffs[1];
                auto   b              = fit.coeffs[2];
                double rank_predict   = std::pow(std::log(C / 1e-5), 1. / b) * k;
                svd_settings.rank_max = static_cast<long>(rank_predict); // TODO: TESTING!
                svd::solver svd2(svd_settings);
                auto [U, S, VT] = svd2.do_svd(A_file);
                t_rsvd.toc();

                //                Eigen::VectorXd S_diff = S_log10.tail(S_log10.size()-1) - S_log10.head(S_log10.size()-1);
                //                double mean_log10_diff = S_diff.array().mean();
                //                auto   linfit        = stat::linearFit(), S_log10, static_cast<long>(S_log10.size() / 2));
                //                double rank_predict  = 3 * epsilon_log10 / linfit.slope;
                tools::log->info("rank_predict: {:.1f} | C {:.3e} | k {:.3e} | b {:.3e}", rank_predict, C, k, b);
                gersvdNorm  = S.norm();
                gersvdRank  = svd2.get_rank();
                gersvdTrc   = svd2.get_truncation_error();
                auto A_rsvd = Eigen::MatrixXcd{U * S.asDiagonal() * VT};
                gersvdErr   = Rsvd::relativeFrobeniusNormError(A_file, A_rsvd);
            }
            //            {
            //                auto t_rsvd2 = tid::tic_scope("rsvd2");
            //
            //                auto                   oversamples    = 2U;
            //                auto                   numIter        = 5U;
            //                auto                   rank           = gesddRank;
            //                auto                   m_randomEngine = rnd::internal::rng;
            //                const Eigen::Index     matrixShortSize{std::min(A_file.rows(), A_file.cols())};
            //                const Eigen::Index     rangeApproximationDim{std::min(matrixShortSize, rank + oversamples)};
            //                const Eigen::MatrixXcd q =
            //                    (numIter == 0U) ? Rsvd::Internal::singleShot<Eigen::MatrixXcd, pcg64>(A_file, rangeApproximationDim, m_randomEngine)
            //                                    : Rsvd::Internal::RandomizedSubspaceIterations<Eigen::MatrixXcd, pcg64,
            //                                    Rsvd::SubspaceIterationConditioner::None>::compute(
            //                                          A_file, rangeApproximationDim, numIter, m_randomEngine);
            //
            //                const auto b = q.adjoint() * A_file;
            //                //    Eigen::JacobiSVD<MatrixType> svd(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
            //                Eigen::BDCSVD<Eigen::MatrixXcd> eigen_svd(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
            //                Eigen::MatrixXcd                U, VT;
            //                Eigen::VectorXd                 S;
            //                U.noalias() = q * eigen_svd.matrixU().leftCols(rank);
            //                S           = eigen_svd.singularValues().head(rank);
            //                VT          = eigen_svd.matrixV().leftCols(rank);
            //            }

            fmt::print("A {:>4} x {:>4} | rank_max {:4} "
                       "| eigen {:8.2e} s "
                       "| lapack  {:8.2e} s rank {:>4} (trc {:.2e} err {:.2e} norm {:8.4f}) "
                       "| rsvd {:8.2e} s rank {:>4} (trc {:.2e} err {:.2e} norm {:<6.4f}) speedup {:.2f}"
                       "| rsvd2 {:8.2e}\n",
                       rows, cols, rank_max,                                                               //
                       tid::get("eigen").get_last_interval(),                                              //
                       tid::get("lapack").get_last_interval(), gesddRank, gesddTrc, gesddErr, gesddNorm,   //
                       tid::get("rsvd").get_last_interval(), gersvdRank, gersvdTrc, gersvdErr, gersvdNorm, //
                       tid::get("lapack").get_last_interval() / tid::get("rsvd").get_last_interval(),      //
                       tid::get("rsvd2").get_last_interval());
        }
        fmt::print("Total time | eigen {:8.2e} s | lapack {:8.2e} s | rsvd {:8.2e}\n", tid::get("eigen").get_time(), tid::get("lapack").get_time(),
                   tid::get("rsvd").get_time());
    }
}

int main() {
    tools::Logger::setLogger(tools::log, "rsvd", 0, true);
    auto filename = fmt::format("{}/rsvd-matrices.h5", BENCH_DATA_DIR);

    if(not h5pp::fs::exists(filename)) {
        tools::log->error("File does not exist: {}", filename);
        exit(0);
    }

    bench();
    tools::log->info("Success");
    return 0;
}