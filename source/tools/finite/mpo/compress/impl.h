#pragma once
#include "../../mpo.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "io/fmt_custom.h"
#include "math/float.h"
#include "math/linalg/matrix/to_string.h"
#include "math/svd.h"
#include "tid/tid.h"
#include <complex>

namespace settings {
    inline constexpr bool debug_compression = false;
}

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_svdcompressed_mpos(std::vector<Eigen::Tensor<Scalar, 4>> mpos) {
    tools::log->trace("Compressing MPOs: {} sites", mpos.size());
    // Setup SVD
    // Here we need a lot of precision:
    //  - Use very low svd threshold
    //  - Force the use of JacobiSVD
    //  - Force the use of Lapacke -- it is more precise than Eigen (I don't know why)
    // Eigen Jacobi becomes ?gesvd (i.e. using QR) with the BLAS backend.
    // See here: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1732
    auto svd_cfg             = svd::config();
    svd_cfg.svd_lib          = svd::lib::lapacke;
    svd_cfg.svd_rtn          = svd::rtn::gejsv;
    svd_cfg.truncation_limit = std::numeric_limits<double>::epsilon();

    svd::solver svd(svd_cfg);

    // Print the results
    std::vector<std::string> report;
    // if(tools::log->level() == spdlog::level::trace)
    for(const auto &mpo : mpos) report.emplace_back(fmt::format("{}", mpo.dimensions()));

    for(size_t iter = 0; iter < 4; iter++) {
        // Next compress from left to right
        Eigen::Tensor<Scalar, 2> T_l2r; // Transfer matrix
        Eigen::Tensor<Scalar, 4> T_mpo;
        for(auto &&[idx, mpo] : iter::enumerate(mpos)) {
            auto mpo_dim_old = mpo.dimensions();
            if(T_l2r.size() == 0)
                T_mpo = mpo; // First iter
            else
                T_mpo = T_l2r.contract(mpo, tenx::idx({1}, {0})); // Subsequent iters

            if(idx + 1 == mpos.size()) {
                mpo = T_mpo;
            } else {
                std::tie(mpo, T_l2r) = svd.split_mpo_l2r(T_mpo);
            }
            if constexpr(settings::debug_compression) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }

        // Now we have done left to right. Next we do right to left
        Eigen::Tensor<Scalar, 2> T_r2l; // Transfer matrix
        Eigen::Tensor<Scalar, 4> mpo_T; // Absorbs transfer matrix
        for(auto &&[idx, mpo] : iter::enumerate_reverse(mpos)) {
            auto mpo_dim_old = mpo.dimensions();
            if(T_r2l.size() == 0)
                mpo_T = mpo;
            else
                mpo_T = mpo.contract(T_r2l, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
            if(idx == 0) {
                mpo = mpo_T;
            } else {
                std::tie(T_r2l, mpo) = svd.split_mpo_r2l(mpo_T);
            }
            if constexpr(settings::debug_compression) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }
    }

    // Print the results
    if constexpr(settings::debug_compression) {
        for(const auto &[idx, msg] : iter::enumerate(report)) tools::log->debug("mpo {}: {} -> {}", idx, msg, mpos[idx].dimensions());
    }
    return mpos;
}

template<typename Vec, typename Real>
requires(std::floating_point<Real>)
std::pair<bool, typename Vec::Scalar> is_parallel_least_squares(const Vec &a, const Vec &b, Real tol_rel, Real tol_abs) {
    using Scalar     = typename Vec::Scalar;
    const Real anorm = a.norm();
    const Real bnorm = b.norm();
    if(anorm == Real{0} or bnorm == Real{0}) return {false, Scalar{0}};

    const Scalar denom = a.dot(a); // a†a (real if a is complex, but keep Scalar)
    if(std::abs(denom) == Real{0}) return {false, Scalar{0}};

    const Scalar factor = a.dot(b) / denom; // (a†b)/(a†a)
    const Real   rnorm  = (b - factor * a).norm();

    const bool ok = rnorm <= (tol_abs + tol_rel * bnorm);
    return {ok, factor};
}

template<typename Scalar>
std::pair<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 2>> deparallelize_mpo_l2r(const Eigen::Tensor<Scalar, 4> &mpo) {
    // Collect index 0,2,3 (left, top, bottom) for rows and leave index 1 for columns.

    /*
     *
     *          (2) d
     *             |
     *  (0) m ---[mpo]--- (1) m
     *             |
     *         (3) d
     *
     * is shuffled {2, 3, 0, 1} into
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
     * Then find parallel columns in this matrix.
     *
     * Finally shuffle back with  {2, 3, 0, 1}

     */

    auto dim0                     = mpo.dimension(2);
    auto dim1                     = mpo.dimension(3);
    auto dim2                     = mpo.dimension(0);
    auto dim3                     = mpo.dimension(1);
    auto dim_ddm                  = dim0 * dim1 * dim2;
    auto mpo_rank2                = Eigen::Tensor<Scalar, 2>(mpo.shuffle(tenx::array4{2, 3, 0, 1}).reshape(tenx::array2{dim_ddm, dim3}));
    auto mpo_map                  = tenx::MatrixMap(mpo_rank2);
    using RealScalar              = decltype(std::real(std::declval<Scalar>()));
    using MatrixType              = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    static constexpr auto eps     = std::numeric_limits<RealScalar>::epsilon();
    static constexpr auto tol_rel = eps * 10;
    static constexpr auto tol_abs = RealScalar{0};

    auto cols     = mpo_map.cols();
    auto col_keep = std::vector<long>{};
    auto mat_xfer = MatrixType(cols, cols);
    mat_xfer.setZero();
    for(long jdx = 0; jdx < mpo_map.cols(); ++jdx) { // checked col index
        if(mpo_map.col(jdx).norm() == RealScalar{0}) {
            // Leave mat_xfer(:, jdx) all zeros, do not keep jdx
            continue;
        }
        auto kmax     = static_cast<long>(col_keep.size()); // col_keep.size() increases inside the for loop
        bool keep_jdx = true;
        for(long kdx = 0; kdx < kmax; ++kdx) { // Kept col index
            // Check if the previous col(idx) is parallel to the current col(jdx)
            auto col_jdx                  = mpo_map.col(jdx);                                // A new column in mpo_map
            auto col_kdx                  = mpo_map.col(col_keep[static_cast<size_t>(kdx)]); // A kept column from cols_keep
            auto [is_parallel, prefactor] = is_parallel_least_squares(col_kdx, col_jdx, tol_rel, tol_abs);
            if(is_parallel) { // can be discarded
                mat_xfer(kdx, jdx) = prefactor;
                keep_jdx           = false;
                break; // Got to next jdx
            }
        }
        if(keep_jdx) {
            // We should keep column jdx if it isn't parallel to any of the kept columns
            col_keep.emplace_back(jdx); // must be added before setting xfer
            mat_xfer(static_cast<long>(col_keep.size() - 1ul), jdx) = Scalar{1};
        }
    }

    // Resize the transfer matrix. It should have size col_keep.size() x cols
    mat_xfer.conservativeResize(static_cast<long>(col_keep.size()), Eigen::NoChange);

    // Create the deparallelized mpo by shuffling the indices back into position
    auto matrix_dep = MatrixType(mpo_map(Eigen::placeholders::all, col_keep)); // Deparallelized matrix
    auto tensor_dep = tenx::TensorMap(matrix_dep, std::array<long, 4>{dim0, dim1, dim2, matrix_dep.cols()});
    auto mpo_dep    = Eigen::Tensor<Scalar, 4>(tensor_dep.shuffle(std::array<long, 4>{2, 3, 0, 1}));

    if constexpr(settings::debug_compression) {
        // Sanity check
        auto mpo_old = mpo_map;
        auto mpo_new = MatrixType(matrix_dep * mat_xfer);
        if(not mpo_old.isApprox(mpo_new)) {
            tools::log->info("mpo_xfer:\n{}", linalg::matrix::to_string(mat_xfer.real(), 6));
            tools::log->info("mpo_old:\n{}", linalg::matrix::to_string(mpo_old.real(), 6));
            tools::log->info("mpo_new:\n{}", linalg::matrix::to_string(mpo_new.real(), 6));
            throw except::logic_error("mpo l2r changed during deparallelization");
        }
    }
    return {mpo_dep, tenx::TensorMap(mat_xfer)};
}

template<typename Scalar>
std::pair<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 4>> deparallelize_mpo_r2l(const Eigen::Tensor<Scalar, 4> &mpo) {
    // Collect index 1,2,3 (right, top, bottom) for rows and leave index 0 for rows.

    /*
     *
     *          (2) d
     *             |
     *  (0) m ---[mpo]--- (1) m
     *            |
     *        (3) d
     *
     * is shuffled into {0, 2, 3, 1}
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
     * Then discard parallel rows in this matrix.
     *
     * Finally shuffle back with  {0, 3, 1, 2}
     */

    auto dim0                     = mpo.dimension(0);
    auto dim1                     = mpo.dimension(2);
    auto dim2                     = mpo.dimension(3);
    auto dim3                     = mpo.dimension(1);
    auto dim_ddm                  = dim1 * dim2 * dim3;
    auto mpo_rank2                = Eigen::Tensor<Scalar, 2>(mpo.shuffle(tenx::array4{0, 2, 3, 1}).reshape(tenx::array2{dim0, dim_ddm}));
    auto mpo_map                  = tenx::MatrixMap(mpo_rank2);
    using RealScalar              = decltype(std::real(std::declval<Scalar>()));
    using MatrixType              = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    static constexpr auto eps     = std::numeric_limits<RealScalar>::epsilon();
    static constexpr auto tol_rel = eps * 10;
    static constexpr auto tol_abs = RealScalar{0};
    auto                  rows    = mpo_map.rows();

    auto row_keep = std::vector<long>{};
    auto mat_xfer = MatrixType(rows, rows);
    mat_xfer.setZero();
    for(long idx = 0; idx < mpo_map.rows(); ++idx) { // checked row index
        if(mpo_map.row(idx).norm() == RealScalar{0}) { continue; }
        auto kmax     = safe_cast<long>(row_keep.size()); // row_keep.size() increases inside the for loop
        bool keep_idx = true;
        for(long kdx = 0; kdx < kmax; ++kdx) { // Kept row index
            // Check if the previous row(idx) is parallel to the current row(idx)
            auto row_idx = mpo_map.row(idx);                                // A new row in mpo_map
            auto row_kdx = mpo_map.row(row_keep[static_cast<size_t>(kdx)]); // A kept row from cols_keep

            auto [is_parallel, prefactor] = is_parallel_least_squares(row_kdx, row_idx, tol_rel, tol_abs);
            if(is_parallel) { // can be discarded
                mat_xfer(idx, kdx) = prefactor;
                keep_idx           = false;
                break; // Got to next jdx
            }
        }
        if(keep_idx) {
            // We should keep row idx if it isn't parallel to any of the kept rows
            row_keep.emplace_back(idx); // must be added before setting xfer
            mat_xfer(idx, static_cast<long>(row_keep.size() - 1ul)) = Scalar{1};
        }
    }
    // Resize the transfer matrix. It should have size rows x row_keep.size()
    mat_xfer.conservativeResize(Eigen::NoChange, static_cast<long>(row_keep.size()));

    // Create the deparallelized mpo by shuffling the indices back into position
    auto matrix_dep = MatrixType(mpo_map(row_keep, Eigen::placeholders::all)); // Deparallelized matrix
    auto tensor_dep = tenx::TensorMap(matrix_dep, std::array<long, 4>{matrix_dep.rows(), dim1, dim2, dim3});
    auto mpo_dep    = Eigen::Tensor<Scalar, 4>(tensor_dep.shuffle(std::array<long, 4>{0, 3, 1, 2}));

    if constexpr(settings::debug_compression) {
        // Sanity check
        auto mpo_old = mpo_map;
        auto mpo_new = MatrixType(mat_xfer * matrix_dep);
        if(not mpo_old.isApprox(mpo_new)) {
            tools::log->info("mpo_xfer:\n{}", linalg::matrix::to_string(mat_xfer.real(), 6));
            tools::log->info("mpo_old:\n{}", linalg::matrix::to_string(mpo_old.real(), 6));
            tools::log->info("mpo_new:\n{}", linalg::matrix::to_string(mpo_new.real(), 6));
            throw except::logic_error("mpo r2l changed during deparallelization");
        }
    }

    return {tenx::TensorMap(mat_xfer), mpo_dep};
}

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_deparallelized_mpos(std::vector<Eigen::Tensor<Scalar, 4>> mpos) {
    tools::log->trace("Deparallelizing MPOs: {} sites", mpos.size());

    // Print the results
    std::vector<std::string> report;
    // if(tools::log->level() == spdlog::level::trace)
    for(const auto &mpo : mpos) report.emplace_back(fmt::format("{}", mpo.dimensions()));
    const Eigen::Index maxiter = 2;
    for(size_t iter = 0; iter < maxiter; iter++) {
        // Start by deparallelizing left to right compress from left to right
        Eigen::Tensor<Scalar, 2> T_l2r; // Transfer matrix
        Eigen::Tensor<Scalar, 4> T_mpo;
        for(auto &&[idx, mpo] : iter::enumerate(mpos)) {
            auto mpo_dim_old = mpo.dimensions();
            if(T_l2r.size() == 0)
                T_mpo = mpo; // First iter
            else
                T_mpo = T_l2r.contract(mpo, tenx::idx({1}, {0})); // Subsequent iters

            if(idx + 1 == mpos.size()) {
                mpo = T_mpo;
            } else {
                std::tie(mpo, T_l2r) = deparallelize_mpo_l2r(T_mpo);
            }
            if constexpr(settings::debug_compression) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }

        // Now we have done left to right. Next we do right to left
        Eigen::Tensor<Scalar, 2> T_r2l; // Transfer matrix
        Eigen::Tensor<Scalar, 4> mpo_T; // Absorbs transfer matrix
        for(auto &&[idx, mpo] : iter::enumerate_reverse(mpos)) {
            auto mpo_dim_old = mpo.dimensions();
            if(T_r2l.size() == 0)
                mpo_T = mpo;
            else
                mpo_T = mpo.contract(T_r2l, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
            if(idx == 0) {
                mpo = mpo_T;
            } else {
                std::tie(T_r2l, mpo) = deparallelize_mpo_r2l(mpo_T);
            }
            if constexpr(settings::debug_compression) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }
    }

    // Print the results
    if constexpr(settings::debug_compression) {
        for(const auto &[idx, msg] : iter::enumerate(report)) tools::log->debug("mpo {}: {} -> {}", idx, msg, mpos[idx].dimensions());
    }
    return mpos;
}
