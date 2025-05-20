#pragma once
#include "../common.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include "to_string.h"
#include <Eigen/Core>

namespace linalg::matrix {
#if defined(NDEBUG)
    inline constexpr bool debug_mgs = false;
#else
    inline constexpr bool debug_mgs = false;
#endif

    template<typename Scalar>
    struct MGS_Result {
        using IdxT       = Eigen::Index;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using VectorIdxT = Eigen::Matrix<IdxT, Eigen::Dynamic, 1>;
        MatrixType        Q;
        MatrixType        R;
        IdxT              nCols = 0;
        std::vector<IdxT> nonZeroCols;
        VectorReal        initColNorms;
        RealScalar        orthError     = RealScalar{0};
        RealScalar        maxOrthError  = std::numeric_limits<RealScalar>::quiet_NaN();
        bool              isOrthonormal = true;
    };

    template<typename MatrixT>
    MGS_Result<typename MatrixT::Scalar> modified_gram_schmidt_dgks(const MatrixT &A) {
        // Orthonormalize with Modified Gram Schmidt
        using IdxT       = Eigen::Index;
        using Scalar     = typename MatrixT::Scalar;
        using MatrixType = typename MGS_Result<Scalar>::MatrixType;
        using RealScalar = typename MGS_Result<Scalar>::RealScalar;
        // using VectorReal = typename MGS_Result<Scalar>::VectorReal;
        // using VectorIdxT = typename MGSResult<Scalar>::VectorIdxT;
        auto m               = MGS_Result<Scalar>();
        m.Q                  = A;
        m.R                  = MatrixType::Zero(A.cols(), A.cols());
        m.nCols              = A.cols();
        m.initColNorms       = A.colwise().norm();
        RealScalar AsqrtSize = std::abs(std::sqrt<RealScalar>(A.cols()));
        RealScalar eps       = std::numeric_limits<RealScalar>::epsilon();

        m.nonZeroCols.reserve(A.cols());

        for(long i = 0; i < m.Q.cols(); ++i) {
            const auto colNorm = m.Q.col(i).norm();
            m.R(i, i)          = colNorm;

            // Tolerance for zero detection
            auto colNormTol = eps * AsqrtSize * m.initColNorms(i);
            // Avoid normalizing a near-zero vector.
            if(colNorm < colNormTol or m.initColNorms(i) < eps * AsqrtSize or !std::isfinite(colNorm)) {
                m.Q.col(i).setZero(); // Set to zero explicitly
                continue;
            }
            m.nonZeroCols.emplace_back(i);

            // DGKS re-orthogonalization on Q.col(i)
            // First CGS‐style sweep: project out previous q_j
            for(IdxT j = 0; j < i; ++j) {
                Scalar alpha = m.Q.col(j).dot(m.Q.col(i));
                m.Q.col(i) -= alpha * m.Q.col(j);
                m.R(j, i) = alpha;
            }
            // Second sweep: mop up the rounding residues
            for(IdxT j = 0; j < i; ++j) {
                Scalar beta = m.Q.col(j).dot(m.Q.col(i));
                m.Q.col(i) -= beta * m.Q.col(j);
                m.R(j, i) += beta; // now R(j,i)=α_j+β_j
            }

            // Update the norm
            m.R(i, i) = m.Q.col(i).norm();
            m.Q.col(i) /= m.R(i, i); // Renormalize

            // Forward projection onto the remaining columns
            for(long j = i + 1; j < m.Q.cols(); ++j) {
                m.R(i, j) = m.Q.col(i).dot(m.Q.col(j));
                m.Q.col(j) -= m.R(i, j) * m.Q.col(i);
            }
        }
        m.nonZeroCols.shrink_to_fit();
        if constexpr(debug_mgs) {
            if(m.nonZeroCols.size() >= 2) {
                // Orthogonality check
                MatrixType Qnnz    = m.Q(Eigen::all, m.nonZeroCols);
                auto       nnzCols = Qnnz.cols();
                m.orthError        = (Qnnz.adjoint() * Qnnz - MatrixType::Identity(nnzCols, nnzCols)).cwiseAbs().maxCoeff();
                m.maxOrthError     = eps * RealScalar{1e4};
                m.isOrthonormal    = m.orthError <= m.maxOrthError;
            }
        }
        return m;
    }

    template<typename MatrixT>
    MGS_Result<typename MatrixT::Scalar> modified_gram_schmidt(const MatrixT &A) {
        // Orthonormalize with Modified Gram Schmidt
        using Scalar     = typename MatrixT::Scalar;
        using MatrixType = typename MGS_Result<Scalar>::MatrixType;
        using RealScalar = typename MGS_Result<Scalar>::RealScalar;
        // using VectorReal = typename MGS_Result<Scalar>::VectorReal;
        // using VectorIdxT = typename MGSResult<Scalar>::VectorIdxT;
        auto m               = MGS_Result<Scalar>();
        m.Q                  = A;
        m.R                  = MatrixType::Zero(A.cols(), A.cols());
        m.nCols              = A.cols();
        m.initColNorms       = A.colwise().norm();
        RealScalar AsqrtSize = std::abs(std::sqrt<RealScalar>(A.cols()));
        RealScalar eps       = std::numeric_limits<RealScalar>::epsilon();
        m.nonZeroCols.reserve(A.cols());

        for(long i = 0; i < m.Q.cols(); ++i) {
            const auto colNorm = m.Q.col(i).norm();
            m.R(i, i)          = colNorm;

            // Tolerance for zero detection
            RealScalar colNormTol = eps * AsqrtSize * m.initColNorms(i);

            // Avoid normalizing a near-zero vector.
            if(colNorm < colNormTol or m.initColNorms(i) < eps * AsqrtSize or !std::isfinite(colNorm)) {
                m.Q.col(i).setZero(); // Set to zero explicitly
                continue;
            }
            m.nonZeroCols.emplace_back(i);
            m.Q.col(i) /= m.R(i, i);
            for(long j = i + 1; j < m.Q.cols(); ++j) {
                m.R(i, j) = m.Q.col(i).dot(m.Q.col(j));
                m.Q.col(j) -= m.R(i, j) * m.Q.col(i);
            }
        }
        m.nonZeroCols.shrink_to_fit();
        if constexpr(debug_mgs) {
            // Orthogonality check
            MatrixType Qnnz    = m.Q(Eigen::all, m.nonZeroCols);
            auto       nnzCols = Qnnz.cols();
            m.orthError        = (Qnnz.adjoint() * Qnnz - MatrixType::Identity(nnzCols, nnzCols)).cwiseAbs().maxCoeff();
            m.maxOrthError     = eps * AsqrtSize * 10000;
            m.isOrthonormal    = m.orthError <= m.maxOrthError;
        }
        return m;
    }

    /*!
     * Performs  column-pivoted Modified Gram-Schmidt factorization with ncfix leftmost columns fixed:
     *    A * P = Q * R
     * where:
     *  - The first ncfix columns of A are copied unchanged into Q (their direction and norm
     *    remain exactly as in A) but are used as orthogonalizing vectors for later columns.
     *  - The remaining columns are pivoted in order of descending residual norm and then
     *    orthonormalized.
     *
     * Arguments:
     *   A     An m×n input matrix [M | P], where M (first ncfix columns) is already
     *         orthogonal (not normalized) and must stay fixed in Q.
     *   ncfix Number of leading columns to preserve exactly (0 ≤ ncfix ≤ n).
     *
     * Returns:
     *   MGSResult containing:
     *     Q           The m×n output matrix. Q.col(i) for i<ncfix equals A.col(i) exactly.
     *                 For i≥ncfix, Q.col(i) is a unit vector from the pivoted MGS.
     *     Rdiag       Vector of length n of R(i,i) = ||column_i|| before normalization.
     *                 Columns with Rdiag(i) ≤ tol are left unnormalized.
     *     permutation The pivot indices applied to A to produce Q.
     *     ncfix       Echo of the input ncfix.
     *
     */

    template<typename Scalar>
    struct MGS_ColPiv_Result {
        using IdxT       = Eigen::Index;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using VectorIdxT = Eigen::Matrix<IdxT, Eigen::Dynamic, 1>;
        MatrixType        Q;
        MatrixType        R;
        IdxT              nCols = 0;
        VectorReal        Rdiag;
        VectorReal        initColNorms;
        VectorIdxT        permutation; // A permutation vector that keeps track of column pivoting
        std::vector<IdxT> nonOrthCols;
        std::vector<IdxT> nonZeroCols;
        RealScalar        threshold = std::numeric_limits<RealScalar>::quiet_NaN();
        Eigen::Index      ncfix     = 0;
    };

    template<typename MatrixT>
    MGS_ColPiv_Result<typename MatrixT::Scalar> modified_gram_schmidt_colpiv(const MatrixT &A, long ncfix = 0) {
        using Scalar = typename MatrixT::Scalar;
        // using MatrixType = typename MGS_ColPiv_Result<Scalar>::MatrixType;
        using IdxT       = typename MGS_ColPiv_Result<Scalar>::IdxT;
        using RealScalar = typename MGS_ColPiv_Result<Scalar>::RealScalar;
        using VectorReal = typename MGS_ColPiv_Result<Scalar>::VectorReal;
        using VectorIdxT = typename MGS_ColPiv_Result<Scalar>::VectorIdxT;

        const IdxT ncols = A.cols();

        MGS_ColPiv_Result<Scalar> m;
        m.Q            = A;
        m.nCols        = ncols;
        m.Rdiag        = VectorReal::Zero(ncols);
        m.initColNorms = A.colwise().norm();
        m.ncfix        = ncfix;

        // Initialize a permutation vector that keeps track of column pivoting
        m.permutation = VectorIdxT::LinSpaced(ncols, 0, ncols - 1);

        RealScalar AsqrtSize = std::abs(std::sqrt<RealScalar>(A.cols()));
        RealScalar eps       = std::numeric_limits<RealScalar>::epsilon();
        // Perform the Modified Gram-Schmidt process with column pivoting
        // The first ncfix columns (from M) are fixed/locked, while the rest are mutable
        for(IdxT i = 0; i < ncols; ++i) {
            // Pivot among mutable columns (i >= ncfix)
            if(i >= ncfix) {
                // Determine the index of the column (from i to end) with the maximal norm.
                IdxT       pivot   = i;
                RealScalar maxNorm = m.Q.col(i).norm();
                for(IdxT j = i + 1; j < ncols; ++j) {
                    RealScalar norm_j = m.Q.col(j).norm();
                    if(norm_j > maxNorm) {
                        maxNorm = norm_j;
                        pivot   = j;
                    }
                }
                // Swap the current column with the column having maximum residual norm.
                if(pivot != i) {
                    m.Q.col(i).swap(m.Q.col(pivot));
                    std::swap(m.permutation(i), m.permutation(pivot));
                    std::swap(m.initColNorms(i), m.initColNorms(pivot));
                }
            }

            const auto colNorm = m.Q.col(i).norm();
            m.Rdiag(i)         = colNorm;

            // Tolerance for zero detection
            RealScalar colNormTol = eps * AsqrtSize * m.initColNorms(i);

            // Avoid normalizing a near-zero vector.
            if(std::abs(colNorm) < colNormTol or m.initColNorms(i) < eps * AsqrtSize or !std::isfinite(colNorm)) {
                m.Q.col(i).setZero(); // Set to zero explicitly
                continue;
            }
            m.nonZeroCols.emplace_back(i);

            // Normalize only mutable columns
            if(i >= ncfix) m.Q.col(i) /= colNorm;

            // Project the subsequent mutable columns
            for(long j = i + 1; j < ncols; ++j) {
                if(j < ncfix) {
                    continue; // Skip fixed columns
                }
                // Subtract projection onto Q.col(i)
                /* clang-format off */
            if(i >= ncfix)  m.Q.col(j) -= m.Q.col(i).dot(m.Q.col(j)) * m.Q.col(i);
            else            m.Q.col(j) -= m.Q.col(i).dot(m.Q.col(j)) * m.Q.col(i) / (colNorm*colNorm); // Q.col(i) is not normalized if i < ncfix
                /* clang-format on */
            }
        }

        return m;
    }

    template<typename MatrixT>
    MGS_ColPiv_Result<typename MatrixT::Scalar> modified_gram_schmidt_colpiv_dgks(const MatrixT &A, long ncfix = 0) {
        using Scalar = typename MatrixT::Scalar;
        // using MatrixType = typename MGS_ColPiv_Result<Scalar>::MatrixType;
        using IdxT       = typename MGS_ColPiv_Result<Scalar>::IdxT;
        using RealScalar = typename MGS_ColPiv_Result<Scalar>::RealScalar;
        using VectorReal = typename MGS_ColPiv_Result<Scalar>::VectorReal;
        using VectorIdxT = typename MGS_ColPiv_Result<Scalar>::VectorIdxT;

        const IdxT ncols = A.cols();

        MGS_ColPiv_Result<Scalar> m;
        m.Q            = A;
        m.nCols        = ncols;
        m.Rdiag        = VectorReal::Zero(ncols);
        m.initColNorms = A.colwise().norm();
        m.ncfix        = ncfix;

        // Initialize a permutation vector that keeps track of column pivoting
        m.permutation = VectorIdxT::LinSpaced(ncols, 0, ncols - 1);

        RealScalar AsqrtSize = std::abs(std::sqrt<RealScalar>(A.cols()));
        RealScalar eps       = std::numeric_limits<RealScalar>::epsilon();
        // Perform the Modified Gram-Schmidt process with column pivoting
        // The first ncfix columns (from M) are fixed/locked, while the rest are mutable
        for(IdxT i = 0; i < ncols; ++i) {
            // Pivot among mutable columns (i >= ncfix)
            if(i >= ncfix) {
                // Determine the index of the column (from i to end) with the maximal norm.
                IdxT       pivot   = i;
                RealScalar maxNorm = m.Q.col(i).norm();
                for(IdxT j = i + 1; j < ncols; ++j) {
                    RealScalar norm_j = m.Q.col(j).norm();
                    if(norm_j > maxNorm) {
                        maxNorm = norm_j;
                        pivot   = j;
                    }
                }
                // Swap the current column with the column having maximum residual norm.
                if(pivot != i) {
                    m.Q.col(i).swap(m.Q.col(pivot));
                    std::swap(m.permutation(i), m.permutation(pivot));
                    std::swap(m.initColNorms(i), m.initColNorms(pivot));
                }
            }

            auto colNorm = m.Q.col(i).norm();
            m.Rdiag(i)   = colNorm;

            // Tolerance for zero detection
            RealScalar colNormTol = eps * AsqrtSize * m.initColNorms(i);

            // Avoid normalizing a near-zero vector.
            if(colNorm < colNormTol or m.initColNorms(i) < eps * AsqrtSize or !std::isfinite(colNorm)) {
                m.Q.col(i).setZero(); // Set to zero explicitly
                continue;
            }
            assert(m.Q.col(i).allFinite());
            m.nonZeroCols.emplace_back(i);

            // DGKS re-orthogonalization on Q.col(i)
            // First CGS‐style sweep: project out previous q_j
            if(i >= ncfix) {
                for(IdxT j = 0; j < i; ++j) {
                    Scalar alpha = m.Q.col(j).dot(m.Q.col(i));
                    m.Q.col(i) -= alpha * m.Q.col(j);
                }
                // Second sweep: mop up the rounding residues
                for(IdxT j = 0; j < i; ++j) {
                    Scalar beta = m.Q.col(j).dot(m.Q.col(i));
                    m.Q.col(i) -= beta * m.Q.col(j);
                }
                // Update the norm
                colNorm    = m.Q.col(i).norm();
                m.Rdiag(i) = colNorm;
                m.Q.col(i) /= colNorm; // Renormalize
            }
            assert(m.Q.col(i).allFinite());
            assert(colNorm != 0);
            // Project the subsequent mutable columns
            for(long j = i + 1; j < ncols; ++j) {
                if(j < ncfix) {
                    continue; // Skip fixed columns
                }
                // Subtract projection onto Q.col(i)
                /* clang-format off */
            if(i >= ncfix)  m.Q.col(j) -= m.Q.col(i).dot(m.Q.col(j)) * m.Q.col(i);
            else            m.Q.col(j) -= m.Q.col(i).dot(m.Q.col(j)) * m.Q.col(i) / (colNorm*colNorm); // Q.col(i) is not normalized if i < ncfix
                /* clang-format on */
            }
        }

        return m;
    }

}
