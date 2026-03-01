#pragma once
#include "util.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <Eigen/Core>
namespace x2 {
    template<typename Scalar>
    struct Matrix {
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));

        private:
        MatrixType hi_, lo_;

        mutable MatrixType cache_sum_;
        mutable bool       cache_valid_ = false;

        void invalidate_cache_() const noexcept { cache_valid_ = false; }

        void ensure_cache_() const {
            if(!cache_valid_ || cache_sum_.rows() != hi_.rows() || cache_sum_.cols() != hi_.cols()) {
                cache_sum_.resize(hi_.rows(), hi_.cols());
                cache_sum_   = (hi_ + lo_);
                cache_valid_ = true;
            }
        }

        public:
        Matrix() = default;

        Matrix(Eigen::Index rows, Eigen::Index cols) {
            hi_.setZero(rows, cols);
            lo_.setZero(rows, cols);
            invalidate_cache_();
        }

        explicit Matrix(const Eigen::Ref<const MatrixType> &A) {
            hi_ = A;
            lo_.setZero(A.rows(), A.cols());
            invalidate_cache_();
        }

        Eigen::Index rows() const { return hi_.rows(); }
        Eigen::Index cols() const { return hi_.cols(); }
        Eigen::Index size() const { return hi_.size(); }

        void resize(Eigen::Index rows, Eigen::Index cols) {
            hi_.resize(rows, cols);
            lo_.resize(rows, cols);
            invalidate_cache_();
        }

        void setZero() {
            hi_.setZero();
            lo_.setZero();
            invalidate_cache_();
        }

        bool allFinite() const { return hi_.allFinite() && lo_.allFinite(); }

        // ---- Accessors ----
        const MatrixType &hi() const { return hi_; }
        const MatrixType &lo() const { return lo_; }

        MatrixType &hi() {
            invalidate_cache_();
            return hi_;
        }
        MatrixType &lo() {
            invalidate_cache_();
            return lo_;
        }

        const Scalar *hi_data() const { return hi_.data(); }
        const Scalar *lo_data() const { return lo_.data(); }

        Scalar *hi_data() {
            invalidate_cache_();
            return hi_.data();
        }
        Scalar *lo_data() {
            invalidate_cache_();
            return lo_.data();
        }

        // Cached sum (hi + lo). Do not keep this reference across later mutations.
        const MatrixType &to_MatrixType() const {
            ensure_cache_();
            return cache_sum_;
        }

        MatrixType to_MatrixType_copy() const { return MatrixType(to_MatrixType()); }

        void renorm() {
            invalidate_cache_();
            x2_detail::renorm(hi_.data(), lo_.data(), size());
        }

        RealScalar norm() const {
            assert(hi_.size() == lo_.size());
            return x2::norm(hi_.data(), lo_.data(), hi_.size());
        }
    };

    template<typename Scalar>
    struct MatrixMap {
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));

        private:
        Eigen::Map<MatrixType> hi_;
        Eigen::Map<MatrixType> lo_;

        public:
        MatrixMap(Scalar *hi_ptr, Scalar *lo_ptr, Eigen::Index rows, Eigen::Index cols) : hi_(hi_ptr, rows, cols), lo_(lo_ptr, rows, cols) {}

        explicit MatrixMap(x2::Matrix<Scalar> &m) : hi_(m.hi_data(), m.rows(), m.cols()), lo_(m.lo_data(), m.rows(), m.cols()) {}

        MatrixMap &operator=(const x2::Matrix<Scalar> &other) {
            assert(rows() == other.rows());
            assert(cols() == other.cols());
            hi_ = other.hi();
            lo_ = other.lo();
            return *this;
        }

        // ---- Accessors ----
        const Eigen::Map<MatrixType> &hi() const { return hi_; }
        const Eigen::Map<MatrixType> &lo() const { return lo_; }

        Eigen::Map<MatrixType> &hi() { return hi_; }
        Eigen::Map<MatrixType> &lo() { return lo_; }

        const Scalar *hi_data() const { return hi_.data(); }
        const Scalar *lo_data() const { return lo_.data(); }

        Scalar *hi_data() { return hi_.data(); }
        Scalar *lo_data() { return lo_.data(); }

        Eigen::Index rows() const {
            assert(hi_.rows() == lo_.rows());
            return hi_.rows();
        }

        Eigen::Index cols() const {
            assert(hi_.cols() == lo_.cols());
            return hi_.cols();
        }

        Eigen::Index size() const { return hi_.size(); }

        void renorm() { x2_detail::renorm(hi_.data(), lo_.data(), size()); }

        RealScalar norm() const {
            assert(hi_.size() == lo_.size());
            return x2::norm(hi_.data(), lo_.data(), static_cast<Eigen::Index>(hi_.size()));
        }
    };

    template<typename Scalar>
    struct ConstMatrixMap {
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));

        private:
        Eigen::Map<const MatrixType> hi_;
        Eigen::Map<const MatrixType> lo_;

        public:
        ConstMatrixMap(const Scalar *hi_ptr, const Scalar *lo_ptr, Eigen::Index rows, Eigen::Index cols) : hi_(hi_ptr, rows, cols), lo_(lo_ptr, rows, cols) {}

        explicit ConstMatrixMap(const x2::Matrix<Scalar> &m) : hi_(m.hi_data(), m.rows(), m.cols()), lo_(m.lo_data(), m.rows(), m.cols()) {}

        // ---- Accessors ----
        const Eigen::Map<const MatrixType> &hi() const { return hi_; }
        const Eigen::Map<const MatrixType> &lo() const { return lo_; }

        const Scalar *hi_data() const { return hi_.data(); }
        const Scalar *lo_data() const { return lo_.data(); }

        Eigen::Index rows() const {
            assert(hi_.rows() == lo_.rows());
            return hi_.rows();
        }

        Eigen::Index cols() const {
            assert(hi_.cols() == lo_.cols());
            return hi_.cols();
        }

        Eigen::Index size() const { return hi_.size(); }

        RealScalar norm() const {
            assert(hi_.size() == lo_.size());
            return x2::norm(hi_.data(), lo_.data(), static_cast<Eigen::Index>(hi_.size()));
        }
    };

}
