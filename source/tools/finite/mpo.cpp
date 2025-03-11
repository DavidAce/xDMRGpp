//
#include <complex>
#ifndef lapack_complex_float
    #define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
    #define lapack_complex_double std::complex<double>
#endif

// complex must be included before lapacke!
#if defined(MKL_AVAILABLE)
    #include <mkl_lapacke.h>
#elif defined(OPENBLAS_AVAILABLE)
    #include <openblas/lapacke.h>
#elif defined(FLEXIBLAS_AVAILABLE)
    #include <flexiblas/lapacke.h>
#else
    #include <lapacke.h>
#endif

//

#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "io/fmt_custom.h"
#include "math/float.h"
#include "math/svd.h"
#include "mpo.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tid/tid.h"

//
// #include <Eigen/Cholesky>
#include "general/sfinae.h"
#include "math/linalg/matrix.h"
#include "math/linalg/tensor.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <Eigen/QR>

std::pair<Eigen::Tensor<cx64, 4>, Eigen::Tensor<cx64, 4>> tools::finite::mpo::swap_mpo(const Eigen::Tensor<cx64, 4> &mpoL, const Eigen::Tensor<cx64, 4> &mpoR) {
    /* The swap operation takes two neighboring sites
     *
     *            (2)dL                           (2)dR
     *              |                               |
     *   (0)mL---[ mpsL ]---mC(1)       mC(0)---[ mpsR ]---(1)mR
     *              |                               |
     *           (3)dL                            (3)dR
     *
     *
     * and joins them into
     *
     *          (1)dL (4)dR
     *             |    |
     *   (0)mL ---[tensor]--- (3)mR
     *             |    |
     *          (2)dL (5)dR
     *
     *
     * Then we swap the physical indices
     *
     *
     *         (4)dR     (1)dL
     *            |        |
     *             \      /
     *               \  /
     *                /\      <---- Swap
     *              /    \
     *            /        \
     *            |        |
     *   (0)mL---[  tensor  ]---(3)mR
     *            |        |
     *             \      /
     *               \  /
     *                /\      <---- Swap
     *              /    \
     *            /        \
     *            |        |
     *         (5)dR     (2)dL
     *
     *
     * This is accomplished by the shuffle:
     *       shuffle(0, 4, 5, 3, 1, 2);
     *
     *
     * We can now split the tensor back into two MPO's, by first reshaping the tensor into a matrix with merged indices (012) x (345)
     *
     *         (4)dR     (1)dL                             (1)dR                                      (1)dL
     *            |        |                                 |                                          |
     *   (0)mL---[  tensor  ]---(3)mR     --->    (0)mL---[ mpoL ]---mC(3)  (0)---[S]---(1)  mC(0)---[ mpoR ]---(1)mR
     *            |        |                                 |                                          |
     *         (5)dR     (2)dL                            (2)dR                                       (2)dL
     *
     *
     * The square root of S can then be multiplied into both left and right MPO's, on the mC index.
     * The left mpo can be shuffled back to standard form with
     *   mpoL: shuffle(0,3,1,2)
     *
     */

    Eigen::Tensor<cx64, 6> swapped_mpo = mpoL.contract(mpoR, tenx::idx({1}, {0})).shuffle(tenx::array6{0, 4, 5, 3, 1, 2}); // swap
    auto                   svd_cfg     = svd::config();
    svd_cfg.truncation_limit           = 1e-16;
    svd_cfg.svd_lib                    = svd::lib::lapacke;
    svd_cfg.svd_rtn                    = svd::rtn::gejsv;

    auto svd = svd::solver(svd_cfg);
    return svd.split_mpo_pair(swapped_mpo, svd_cfg);
}

void tools::finite::mpo::swap_sites(ModelFinite &model, size_t posL, size_t posR, std::vector<size_t> &sites) {
    auto t_swap = tid::tic_scope("swap");
    if(posR != posL + 1) throw except::logic_error("Expected posR == posL+1. Got posL {} and posR {}", posL, posR);
    if(posR != std::clamp(posR, 0ul, model.get_length() - 1ul)) throw except::logic_error("Expected posR in [0,{}]. Got {}", model.get_length() - 1, posR);
    if(posL != std::clamp(posL, 0ul, model.get_length() - 1ul)) throw except::logic_error("Expected posL in [0,{}]. Got {}", model.get_length() - 1, posL);

    auto &mpo_posL = model.get_mpo(posL);
    auto &mpo_posR = model.get_mpo(posR);

    auto [mpoL, mpoR]   = swap_mpo(mpo_posL.MPO(), mpo_posR.MPO());
    auto [mpoL2, mpoR2] = swap_mpo(mpo_posL.MPO2(), mpo_posR.MPO2());
    tools::log->debug("mpo({}) : {} -> {}", mpo_posL.get_position(), mpo_posL.MPO().dimensions(), mpoL.dimensions());
    tools::log->debug("mpo({}) : {} -> {}", mpo_posR.get_position(), mpo_posR.MPO().dimensions(), mpoR.dimensions());
    tools::log->debug("mpo²({}): {} -> {}", mpo_posL.get_position(), mpo_posL.MPO2().dimensions(), mpoL2.dimensions());
    tools::log->debug("mpo²({}): {} -> {}", mpo_posR.get_position(), mpo_posR.MPO2().dimensions(), mpoR2.dimensions());
    mpo_posL.set_mpo(mpoL);
    mpo_posR.set_mpo(mpoR);
    mpo_posL.set_mpo_squared(mpoL2);
    mpo_posR.set_mpo_squared(mpoR2);

    std::swap(sites[posL], sites[posR]);
}

std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_mpos_with_edges(const std::vector<Eigen::Tensor<cx64, 4>> &mpos,
                                                                            const Eigen::Tensor<cx64, 1> &Ledge, const Eigen::Tensor<cx64, 1> &Redge) {
    auto  mpos_with_edge = mpos;
    auto &threads        = tenx::threads::get();

    /* We can prepend edgeL to the first mpo to reduce the size of subsequent operations.
     * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
     *                        2               2
     *                        |               |
     *    0---[L]---(1)(0)---[M]---1 =  0---[LM]---1
     *                        |               |
     *                        3               3
     */
    const auto &mpoL_src = mpos.front();
    auto       &mpoL_tgt = mpos_with_edge.front();
    mpoL_tgt.resize(tenx::array4{1, mpoL_src.dimension(1), mpoL_src.dimension(2), mpoL_src.dimension(3)});
    mpoL_tgt.device(*threads->dev) = Ledge.reshape(tenx::array2{1, Ledge.size()}).contract(mpoL_src, tenx::idx({1}, {0}));

    /* We can append edgeR to the last mpo to reduce the size of subsequent operations.
     * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
     *         2                              1                       2
     *         |                              |                       |
     *    0---[M]---1  0---[R]---1   =  0---[MR]---3  [shuffle]  0---[MR]---1
     *         |                              |                       |
     *         3                              2                       3
     */
    const auto &mpoR_src = mpos.back();
    auto       &mpoR_tgt = mpos_with_edge.back();
    mpoR_tgt.resize(tenx::array4{mpoR_src.dimension(0), 1, mpoR_src.dimension(2), mpoR_src.dimension(3)});
    mpoR_tgt.device(*threads->dev) = mpoR_src.contract(Redge.reshape(tenx::array2{Redge.size(), 1}), tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
    return mpos_with_edge;
}

std::vector<Eigen::Tensor<cx128, 4>> tools::finite::mpo::get_mpos_with_edges_t(const std::vector<Eigen::Tensor<cx128, 4>> &mpos,
                                                                               const Eigen::Tensor<cx128, 1> &Ledge, const Eigen::Tensor<cx128, 1> &Redge) {
    auto  mpos_with_edge = mpos;
    auto &threads        = tenx::threads::get();

    /* We can prepend edgeL to the first mpo to reduce the size of subsequent operations.
     * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
     *                        2               2
     *                        |               |
     *    0---[L]---(1)(0)---[M]---1 =  0---[LM]---1
     *                        |               |
     *                        3               3
     */
    const auto &mpoL_src = mpos.front();
    auto       &mpoL_tgt = mpos_with_edge.front();
    mpoL_tgt.resize(tenx::array4{1, mpoL_src.dimension(1), mpoL_src.dimension(2), mpoL_src.dimension(3)});
    mpoL_tgt.device(*threads->dev) = Ledge.reshape(tenx::array2{1, Ledge.size()}).contract(mpoL_src, tenx::idx({1}, {0}));

    /* We can append edgeR to the last mpo to reduce the size of subsequent operations.
     * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
     *         2                              1                       2
     *         |                              |                       |
     *    0---[M]---1  0---[R]---1   =  0---[MR]---3  [shuffle]  0---[MR]---1
     *         |                              |                       |
     *         3                              2                       3
     */
    const auto &mpoR_src = mpos.back();
    auto       &mpoR_tgt = mpos_with_edge.back();
    mpoR_tgt.resize(tenx::array4{mpoR_src.dimension(0), 1, mpoR_src.dimension(2), mpoR_src.dimension(3)});
    mpoR_tgt.device(*threads->dev) = mpoR_src.contract(Redge.reshape(tenx::array2{Redge.size(), 1}), tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
    return mpos_with_edge;
}

std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_compressed_mpos(std::vector<Eigen::Tensor<cx64, 4>> mpos, MpoCompress mpoComp) {
    switch(mpoComp) {
        case MpoCompress::NONE: return mpos;
        case MpoCompress::SVD: return get_svdcompressed_mpos(mpos);
        case MpoCompress::DPL: return get_deparallelized_mpos(mpos);
        default: return mpos;
    }
}
std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_compressed_mpos(const std::vector<Eigen::Tensor<cx64, 4>> &mpos,
                                                                            const Eigen::Tensor<cx64, 1> &Ledge, const Eigen::Tensor<cx64, 1> &Redge,
                                                                            MpoCompress mpoComp) {
    switch(mpoComp) {
        case MpoCompress::NONE: return mpos;
        case MpoCompress::SVD: return get_svdcompressed_mpos(mpos, Ledge, Redge);
        case MpoCompress::DPL: return get_deparallelized_mpos(mpos, Ledge, Redge);
        default: return mpos;
    }
}

std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_svdcompressed_mpos(std::vector<Eigen::Tensor<cx64, 4>> mpos) {
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
        Eigen::Tensor<cx64, 2> T_l2r; // Transfer matrix
        Eigen::Tensor<cx64, 4> T_mpo;
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
            if constexpr(settings::debug) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }

        // Now we have done left to right. Next we do right to left
        Eigen::Tensor<cx64, 2> T_r2l; // Transfer matrix
        Eigen::Tensor<cx64, 4> mpo_T; // Absorbs transfer matrix
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
            if constexpr(settings::debug) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }
    }

    // Print the results
    if(tools::log->level() == spdlog::level::debug)
        for(const auto &[idx, msg] : iter::enumerate(report)) tools::log->debug("mpo {}: {} -> {}", idx, msg, mpos[idx].dimensions());

    return mpos;
}
std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_svdcompressed_mpos(const std::vector<Eigen::Tensor<cx64, 4>> &mpos,
                                                                               const Eigen::Tensor<cx64, 1> &Ledge, const Eigen::Tensor<cx64, 1> &Redge) {
    return get_svdcompressed_mpos(get_mpos_with_edges(mpos, Ledge, Redge));
}

std::pair<Eigen::Tensor<cx64, 4>, Eigen::Tensor<cx64, 2>> deparallelize_mpo_l2r(const Eigen::Tensor<cx64, 4> &mpo) {
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

    auto dim0      = mpo.dimension(2);
    auto dim1      = mpo.dimension(3);
    auto dim2      = mpo.dimension(0);
    auto dim3      = mpo.dimension(1);
    auto dim_ddm   = dim0 * dim1 * dim2;
    auto mpo_rank2 = Eigen::Tensor<cx64, 2>(mpo.shuffle(tenx::array4{2, 3, 0, 1}).reshape(tenx::array2{dim_ddm, dim3}));
    auto mpo_map   = tenx::MatrixMap(mpo_rank2);

    // auto rows     = mpo_map.rows();
    auto cols     = mpo_map.cols();
    auto col_keep = std::vector<long>{};
    auto mat_xfer = Eigen::MatrixXcd(cols, cols);
    mat_xfer.setZero();
    for(long jdx = 0; jdx < mpo_map.cols(); ++jdx) { // checked col index
        if(col_keep.size() == 0 and mpo_map.col(jdx).norm() != 0.0) {
            // Keep if none are already kept
            mat_xfer(0l, jdx) = cx64(1.0, 0.0);
            col_keep.emplace_back(jdx);
            continue;
        }
        auto kmax     = static_cast<long>(col_keep.size()); // col_keep.size() increases inside the for loop
        bool keep_jdx = true;
        for(long kdx = 0; kdx < kmax; ++kdx) { // Kept col index
            // Check if the previous col(idx) is parallel to the current col(jdx)
            auto col_jdx = mpo_map.col(jdx);                                // A new column in mpo_map
            auto col_kdx = mpo_map.col(col_keep[static_cast<size_t>(kdx)]); // A kept column from cols_keep

            // Find the row index with the first nonzero element in both col_kdx and col_jdx
            auto prefactor = cx64(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // Factor between the two columns
            for(long rdx = 0; rdx < std::min(col_kdx.size(), col_jdx.size()); ++rdx) {                                 // row index
                if(std::abs(col_kdx[rdx]) != 0.0 and std::abs(col_jdx[rdx]) != 0.0) {
                    prefactor = col_jdx[rdx] / col_kdx[rdx];
                    break;
                }
            }
            if(std::isnan(prefactor.real()) or std::isnan(prefactor.imag()))
                continue; // The factor could not be set. This can happen if the columns are orthogonal.
            bool is_parallel = true;
            // Check that all nonzero elements agree on this prefactor
            for(long rdx = 0; rdx < std::min(col_kdx.size(), col_jdx.size()); ++rdx) { // row index
                auto diff = col_kdx[rdx] * prefactor - col_jdx[rdx];
                if(std::abs(diff) > std::numeric_limits<double>::epsilon()) {
                    is_parallel = false;
                    break;
                }
            }

            if(is_parallel) { // can be discarded
                mat_xfer(kdx, jdx) = prefactor;
                keep_jdx           = false;
                break; // Got to next jdx
            }
        }
        if(keep_jdx) {
            // We should keep column jdx if it isn't parallel to any of the kept columns
            col_keep.emplace_back(jdx); // must be added before setting xfer
            mat_xfer(static_cast<long>(col_keep.size() - 1ul), jdx) = cx64(1.0, 0.0);
        }
    }

    // Resize the transfer matrix. It should have size col_keep.size() x cols
    mat_xfer.conservativeResize(static_cast<long>(col_keep.size()), Eigen::NoChange);

    // Create the deparallelized mpo by shuffling the indices back into position
    auto matrix_dep = Eigen::MatrixXcd(mpo_map(Eigen::all, col_keep)); // Deparallelized matrix
    auto tensor_dep = tenx::TensorMap(matrix_dep, std::array<long, 4>{dim0, dim1, dim2, matrix_dep.cols()});
    auto mpo_dep    = Eigen::Tensor<cx64, 4>(tensor_dep.shuffle(std::array<long, 4>{2, 3, 0, 1}));

    if constexpr(settings::debug) {
        // Sanity check
        auto mpo_old = mpo_map;
        auto mpo_new = Eigen::MatrixXcd(matrix_dep * mat_xfer);
        if(not mpo_old.isApprox(mpo_new)) {
            tools::log->info("mpo_xfer:\n{}", linalg::matrix::to_string(mat_xfer.real(), 6));
            tools::log->info("mpo_old:\n{}", linalg::matrix::to_string(mpo_old.real(), 6));
            tools::log->info("mpo_new:\n{}", linalg::matrix::to_string(mpo_new.real(), 6));
            throw except::logic_error("mpo l2r changed during deparallelization");
        }
    }
    return {mpo_dep, tenx::TensorMap(mat_xfer)};
}

std::pair<Eigen::Tensor<cx64, 2>, Eigen::Tensor<cx64, 4>> deparallelize_mpo_r2l(const Eigen::Tensor<cx64, 4> &mpo) {
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

    auto dim0      = mpo.dimension(0);
    auto dim1      = mpo.dimension(2);
    auto dim2      = mpo.dimension(3);
    auto dim3      = mpo.dimension(1);
    auto dim_ddm   = dim1 * dim2 * dim3;
    auto mpo_rank2 = Eigen::Tensor<cx64, 2>(mpo.shuffle(tenx::array4{0, 2, 3, 1}).reshape(tenx::array2{dim0, dim_ddm}));
    auto mpo_map   = tenx::MatrixMap(mpo_rank2);

    auto rows = mpo_map.rows();
    // auto cols     = mpo_map.cols();
    auto row_keep = std::vector<long>{};
    auto mat_xfer = Eigen::MatrixXcd(rows, rows);
    mat_xfer.setZero();
    for(long idx = 0; idx < mpo_map.rows(); ++idx) { // checked row index
        if(row_keep.size() == 0 and mpo_map.row(idx).norm() != 0.0) {
            // Keep if none are already kept
            mat_xfer(idx, 0l) = cx64(1.0, 0.0);
            row_keep.emplace_back(idx);
            continue;
        }
        auto kmax     = safe_cast<long>(row_keep.size()); // row_keep.size() increases inside the for loop
        bool keep_idx = true;
        for(long kdx = 0; kdx < kmax; ++kdx) { // Kept row index
            // Check if the previous row(idx) is parallel to the current row(idx)
            auto row_idx = mpo_map.row(idx);                                // A new row in mpo_map
            auto row_kdx = mpo_map.row(row_keep[static_cast<size_t>(kdx)]); // A kept row from cols_keep

            // Find the col index with the first nonzero element in both row_kdx and row_jdx
            auto prefactor = cx64(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // Factor between the two columns
            for(long cdx = 0; cdx < std::min(row_kdx.size(), row_idx.size()); ++cdx) {                                 // col index
                if(std::abs(row_kdx[cdx]) != 0.0 and std::abs(row_idx[cdx]) != 0.0) {
                    prefactor = row_idx[cdx] / row_kdx[cdx];
                    break;
                }
            }
            if(std::isnan(prefactor.real()) or std::isnan(prefactor.imag()))
                continue; // The factor could not be set. This can happen if the rows are orthogonal.
            bool is_parallel = true;
            // Check that all nonzero elements agree on this prefactor
            for(long cdx = 0; cdx < std::min(row_kdx.size(), row_idx.size()); ++cdx) { // row index
                auto diff = row_kdx[cdx] * prefactor - row_idx[cdx];
                if(std::abs(diff) > std::numeric_limits<double>::epsilon()) {
                    is_parallel = false;
                    break;
                }
            }

            if(is_parallel) { // can be discarded
                mat_xfer(idx, kdx) = prefactor;
                keep_idx           = false;
                break; // Got to next jdx
            }
        }
        if(keep_idx) {
            // We should keep row idx if it isn't parallel to any of the kept rows
            row_keep.emplace_back(idx); // must be added before setting xfer
            mat_xfer(idx, static_cast<long>(row_keep.size() - 1ul)) = cx64(1.0, 0.0);
        }
    }
    // Resize the transfer matrix. It should have size rows x row_keep.size()
    mat_xfer.conservativeResize(Eigen::NoChange, static_cast<long>(row_keep.size()));

    // Create the deparallelized mpo by shuffling the indices back into position
    auto matrix_dep = Eigen::MatrixXcd(mpo_map(row_keep, Eigen::all)); // Deparallelized matrix
    auto tensor_dep = tenx::TensorMap(matrix_dep, std::array<long, 4>{matrix_dep.rows(), dim1, dim2, dim3});
    auto mpo_dep    = Eigen::Tensor<cx64, 4>(tensor_dep.shuffle(std::array<long, 4>{0, 3, 1, 2}));

    if constexpr(settings::debug) {
        // Sanity check
        auto mpo_old = mpo_map;
        auto mpo_new = Eigen::MatrixXcd(mat_xfer * matrix_dep);
        if(not mpo_old.isApprox(mpo_new)) {
            tools::log->info("mpo_xfer:\n{}", linalg::matrix::to_string(mat_xfer.real(), 6));
            tools::log->info("mpo_old:\n{}", linalg::matrix::to_string(mpo_old.real(), 6));
            tools::log->info("mpo_new:\n{}", linalg::matrix::to_string(mpo_new.real(), 6));
            throw except::logic_error("mpo r2l changed during deparallelization");
        }
    }

    return {tenx::TensorMap(mat_xfer), mpo_dep};
}

std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_deparallelized_mpos(std::vector<Eigen::Tensor<cx64, 4>> mpos) {
    tools::log->trace("Deparallelizing MPOs: {} sites", mpos.size());

    // Print the results
    std::vector<std::string> report;
    // if(tools::log->level() == spdlog::level::trace)
    for(const auto &mpo : mpos) report.emplace_back(fmt::format("{}", mpo.dimensions()));

    for(size_t iter = 0; iter < 2; iter++) {
        // Start by deparallelizing left to right compress from left to right
        Eigen::Tensor<cx64, 2> T_l2r; // Transfer matrix
        Eigen::Tensor<cx64, 4> T_mpo;
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
            if constexpr(settings::debug) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }

        // Now we have done left to right. Next we do right to left
        Eigen::Tensor<cx64, 2> T_r2l; // Transfer matrix
        Eigen::Tensor<cx64, 4> mpo_T; // Absorbs transfer matrix
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
            if constexpr(settings::debug) tools::log->trace("iter {} | idx {} | dim {} -> {}", iter, idx, mpo_dim_old, mpo.dimensions());
        }
    }

    // Print the results
    if(tools::log->level() == spdlog::level::debug)
        for(const auto &[idx, msg] : iter::enumerate(report)) tools::log->debug("mpo {}: {} -> {}", idx, msg, mpos[idx].dimensions());

    return mpos;
}

std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_deparallelized_mpos(const std::vector<Eigen::Tensor<cx64, 4>> &mpos,
                                                                                const Eigen::Tensor<cx64, 1> &Ledge, const Eigen::Tensor<cx64, 1> &Redge) {
    return get_deparallelized_mpos(get_mpos_with_edges(mpos, Ledge, Redge));
}