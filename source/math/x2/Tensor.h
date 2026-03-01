#pragma once
#include "general/sfinae.h"
#include "math/tenx.h"
#include "util.h"
#include <cassert>

namespace x2 {

    template<typename Scalar_, int rank>
    class Tensor {
        public:
        using Scalar     = Scalar_;
        using RealScalar = Eigen::NumTraits<Scalar>::Real;
        using TensorType = Eigen::Tensor<Scalar, rank>;

        private:
        TensorType hi_, lo_;

        // Cache for (hi + lo)
        mutable TensorType cache_sum_;
        mutable bool       cache_valid_ = false;

        void invalidate_cache_() const noexcept { cache_valid_ = false; }

        void ensure_cache_() const {
            if(!cache_valid_ || cache_sum_.dimensions() != hi_.dimensions()) {
                cache_sum_.resize(hi_.dimensions());
                cache_sum_   = (hi_ + lo_);
                cache_valid_ = true;
            }
        }

        public:
        Tensor() = default;

        Tensor(const Eigen::array<Eigen::Index, rank> &dims) {
            hi_.resize(dims);
            lo_.resize(dims);
        }

        template<typename... Dims>
        requires(std::integral<Dims> && ...)
        Tensor(Dims... dims) {
            static_assert(sizeof...(Dims) == rank);
            hi_.resize(dims...);
            lo_.resize(dims...);
        }

        Tensor(const Eigen::TensorRef<const TensorType> &A) {
            hi_ = A;
            lo_.resize(A.dimensions());
            lo_.setZero();
        }

        Tensor(const Eigen::TensorRef<const TensorType> &hi, const Eigen::TensorRef<const TensorType> &lo) {
            hi_ = hi;
            lo_ = lo;
        }
        Tensor(const Tensor &A) : hi_(A.hi()), lo_(A.lo()) {}

        Tensor(Tensor &&other) noexcept : hi_(std::move(other.hi_)), lo_(std::move(other.lo_)) {
            invalidate_cache_();
            other.invalidate_cache_();
        }
        Tensor &operator=(const Tensor &A) {
            hi_ = A.hi();
            lo_ = A.lo();
            invalidate_cache_();
            return *this;
        }
        Tensor &operator=(Tensor &&other) noexcept {
            if(this == &other) return *this;
            hi_ = std::move(other.hi_);
            lo_ = std::move(other.lo_);

            // destination cache no longer trustworthy
            invalidate_cache_();
            // also invalidate source cache
            other.invalidate_cache_();
            return *this;
        }
        Tensor &operator=(const Eigen::TensorRef<const TensorType> &A) {
            hi_ = A;
            lo_.resize(A.dimensions());
            lo_.setZero();
            invalidate_cache_();
            return *this;
        }
        void resize(const Eigen::array<Eigen::Index, rank> &dims) {
            hi_.resize(dims);
            lo_.resize(dims);
            invalidate_cache_();
        }

        template<typename... Dims>
        void resize(Dims... dims) {
            static_assert(sizeof...(Dims) == rank);
            hi_.resize(dims...);
            lo_.resize(dims...);
            invalidate_cache_();
        }

        // ---- Accessors ----
        const TensorType &hi() const { return hi_; }
        const TensorType &lo() const { return lo_; }

        TensorType &hi() {
            invalidate_cache_();
            return hi_;
        }
        TensorType &lo() {
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

        Eigen::Index size() const { return hi_.size(); }

        Eigen::array<Eigen::Index, rank> dimensions() const {
            assert(hi_.dimensions() == lo_.dimensions());
            return hi_.dimensions();
        }

        Eigen::Index dimension(Eigen::Index n) const {
            assert(static_cast<int>(n) >= 0 && static_cast<int>(n) < rank);
            assert(hi_.dimension(n) == lo_.dimension(n));
            return hi_.dimension(n);
        }

        void setZero() {
            invalidate_cache_();
            hi_.setZero();
            lo_.setZero();
        }

        bool allFinite() const {
            auto finite_elem = [](Scalar elem) {
                if constexpr(Eigen::NumTraits<Scalar>::IsComplex) {
                    return std::isfinite(std::real(elem)) && std::isfinite(std::imag(elem));
                } else {
                    return std::isfinite(elem);
                }
            };

            for(Eigen::Index i = 0; i < hi_.size(); ++i)
                if(!finite_elem(hi_.data()[i])) return false;

            for(Eigen::Index i = 0; i < lo_.size(); ++i)
                if(!finite_elem(lo_.data()[i])) return false;

            return true;
        }

        // Cached downcast view (hi + lo). Returns a const reference to internal cache.
        // If you assign it to a value, you still get a copy as usual.
        const TensorType &to_EigenTensor() const {
            ensure_cache_();
            return cache_sum_;
        }

        // Returns a copy by value
        TensorType to_EigenTensor_copy() const { return to_EigenTensor(); }

        template<typename T>
        decltype(auto) cast() const {
            static_assert(std::is_floating_point_v<RealScalar> || std::is_same_v<RealScalar, fp128>);
            if constexpr(std::is_same_v<Scalar, T>) {
                return (*this); // returns const Tensor<Scalar,rank>&
            } else {
                return x2::Tensor<T, rank>(hi_.template cast<T>(), lo_.template cast<T>());
            }
        }

        // In-place TwoSum renorm: (hi,lo) <- TwoSum(hi,lo) elementwise
        void renorm() {
            invalidate_cache_();
            x2_detail::renorm(hi_.data(), lo_.data(), size());
        }

        void shuffle(const Eigen::array<int, rank> &perm) {
            invalidate_cache_();
            hi_ = TensorType(hi_.shuffle(perm));
            lo_ = TensorType(lo_.shuffle(perm));
            assert(allFinite());
        }

        void conjugate() {
            if constexpr(!std::is_floating_point_v<Scalar>) {
                invalidate_cache_();
                hi_ = TensorType(hi_.conjugate());
                lo_ = TensorType(lo_.conjugate());
            }
        }

        // Frobenius norm of (hi + lo)
        RealScalar norm() const {
            assert(hi_.size() == lo_.size());
            return x2::norm(hi_.data(), lo_.data(), hi_.size());
        }
    };

    template<typename Scalar_, int rank>
    class TensorMap {
        public:
        using Scalar     = Scalar_;
        using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
        using TensorType = Eigen::Tensor<Scalar, rank>;

        private:
        Eigen::TensorMap<TensorType> hi;
        Eigen::TensorMap<TensorType> lo;

        public:
        TensorMap(Scalar *hi_ptr, Scalar *lo_ptr, const Eigen::array<Eigen::Index, rank> &dims) : hi(hi_ptr, dims), lo(lo_ptr, dims) {}

        TensorMap(x2::Tensor<Scalar, rank> &t) : hi(t.hi_data(), t.dimensions()), lo(t.lo_data(), t.dimensions()) {}

        TensorMap &operator=(const x2::Tensor<Scalar, rank> &other) {
            assert(this->dimensions() == other.dimensions());
            // dimensions() already implies hi/lo same dims, but keep the spirit:
            assert(this->lo.dimensions() == other.hi().dimensions());
            this->hi = other.hi();
            this->lo = other.lo();
            return *this;
        }

        const Scalar *hi_data() const { return hi.data(); }
        const Scalar *lo_data() const { return lo.data(); }

        Scalar *hi_data() { return hi.data(); }
        Scalar *lo_data() { return lo.data(); }

        Eigen::Index size() const { return hi.size(); }

        Eigen::array<Eigen::Index, rank> dimensions() const {
            assert(hi.dimensions() == lo.dimensions());
            return hi.dimensions();
        }

        Eigen::Index dimension(Eigen::Index n) const {
            assert(hi.dimension(n) == lo.dimension(n));
            return hi.dimension(n);
        }

        RealScalar norm() const {
            assert(hi.size() == lo.size());
            return x2::norm(hi.data(), lo.data(), hi.size());
        }
    };

    template<typename Scalar_, int rank>
    class ConstTensorMap {
        public:
        using Scalar     = Scalar_;
        using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
        using TensorType = Eigen::Tensor<Scalar, rank>;

        private:
        Eigen::TensorMap<const TensorType> hi;
        Eigen::TensorMap<const TensorType> lo;

        public:
        ConstTensorMap(const Scalar *hi_ptr, const Scalar *lo_ptr, const Eigen::array<Eigen::Index, rank> &dims) : hi(hi_ptr, dims), lo(lo_ptr, dims) {}

        ConstTensorMap(const x2::Tensor<Scalar, rank> &t) : hi(t.hi_data(), t.dimensions()), lo(t.lo_data(), t.dimensions()) {}

        const Scalar *hi_data() const { return hi.data(); }
        const Scalar *lo_data() const { return lo.data(); }

        Eigen::Index size() const { return hi.size(); }

        Eigen::array<Eigen::Index, rank> dimensions() const {
            assert(hi.dimensions() == lo.dimensions());
            return hi.dimensions();
        }

        Eigen::Index dimension(Eigen::Index n) const {
            assert(hi.dimension(n) == lo.dimension(n));
            return hi.dimension(n);
        }

        RealScalar norm() const {
            assert(hi.size() == lo.size());
            return x2::norm(hi.data(), lo.data(), hi.size());
        }
    };

}