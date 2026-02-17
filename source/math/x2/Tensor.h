#pragma once
#include "general/sfinae.h"
#include "math/tenx.h"
#include "util.h"
#include <cassert>

namespace x2 {
    template<typename Scalar, int Rank> struct Tensor;

    template<typename Scalar, int Rank, typename Perm>
    void shuffle_inplace(Tensor<Scalar, Rank> &T, const Perm &perm) {
        // Avoid aliasing issues by shuffling into temporaries and swapping
        auto hi_new = Eigen::Tensor<Scalar, Rank>(T.hi.shuffle(perm));
        auto lo_new = Eigen::Tensor<Scalar, Rank>(T.lo.shuffle(perm));
        T.hi        = std::move(hi_new);
        T.lo        = std::move(lo_new);
    }

    template<typename Scalar_, int rank>
    struct Tensor {
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
        const TensorType &to_TensorType() const {
            ensure_cache_();
            return cache_sum_;
        }

        // Returns a copy by value
        TensorType to_TensorType_copy() const { return to_TensorType(); }

        template<typename T>
        decltype(auto) cast() const {
            static_assert(std::is_floating_point_v<RealScalar> || std::is_same_v<RealScalar, fp128>);
            if constexpr(std::is_same_v<Scalar, T>) {
                return (*this); // returns const Tensor<Scalar,rank>&
            } else {
                return x2::Tensor<T, rank>(hi_.template cast<T>(), lo_.template cast<T>());
            }
        }

        // Cheap renormalization: enforce hi carries the leading bits
        void renorm() {
            invalidate_cache_();
            TensorType s = (hi_ + lo_);
            TensorType e = (lo_ - (s - hi_));
            hi_          = std::move(s);
            lo_          = std::move(e);
            assert(allFinite());
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
    struct TensorMap {
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
    struct ConstTensorMap {
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

    // template<typename Scalar_, int rank>
    // struct Tensor {
    //     using Scalar     = Scalar_;
    //     using RealScalar = Eigen::NumTraits<Scalar>::Real;
    //     using TensorType = Eigen::Tensor<Scalar, rank>;
    //     TensorType hi, lo;
    //
    //     Tensor() = default;
    //     Tensor(const Eigen::array<Eigen::Index, rank> &dims) {
    //         hi.resize(dims);
    //         lo.resize(dims);
    //     }
    //     template<typename... Dims>
    //     requires(std::integral<Dims> && ...)
    //     Tensor(Dims... dims) {
    //         static_assert(sizeof...(Dims) == rank);
    //         hi.resize(dims...);
    //         lo.resize(dims...);
    //     }
    //     Tensor(const Eigen::TensorRef<const TensorType> &A) {
    //         this->hi = A;
    //         this->lo.resize(A.dimensions());
    //         this->lo.setZero();
    //     }
    //     Tensor(const Eigen::TensorRef<const TensorType> &hi, const Eigen::TensorRef<const TensorType> &lo) {
    //         this->hi = hi;
    //         this->lo = lo;
    //     }
    //     Tensor &operator=(const Eigen::TensorRef<const TensorType> &A) {
    //         this->hi = A;
    //         this->lo.resize(A.dimensions());
    //         this->lo.setZero();
    //         return *this;
    //     }
    //     void resize(const Eigen::array<Eigen::Index, rank> &dims) {
    //         hi.resize(dims);
    //         lo.resize(dims);
    //     }
    //     template<typename... Dims>
    //     void resize(Dims... dims) {
    //         static_assert(sizeof...(Dims) == rank);
    //         hi.resize(dims...);
    //         lo.resize(dims...);
    //     }
    //
    //     Eigen::Index size() const { return hi.size(); }
    //
    //     Eigen::array<Eigen::Index, rank> dimensions() const {
    //         assert(hi.dimensions() == lo.dimensions());
    //         return hi.dimensions();
    //     }
    //     Eigen::Index dimension(Eigen::Index n) const {
    //         assert(hi.dimension(n) == lo.dimension(n));
    //         assert(static_cast<size_t>(n) < hi.dimensions().size());
    //         return hi.dimension(n);
    //     }
    //
    //     void setZero() {
    //         hi.setZero();
    //         lo.setZero();
    //     }
    //     bool allFinite() const {
    //         for(Eigen::Index i = 0; i < hi.size(); ++i) {
    //             Scalar elem = hi.data()[i];
    //             if constexpr(Eigen::NumTraits<Scalar>::IsComplex) {
    //                 if(!std::isfinite(std::real(elem)) or !std::isfinite(std::imag(elem))) return false;
    //             } else {
    //                 if(!std::isfinite(elem)) return false;
    //             }
    //         }
    //         for(Eigen::Index i = 0; i < lo.size(); ++i) {
    //             Scalar elem = lo.data()[i];
    //             if constexpr(Eigen::NumTraits<Scalar>::IsComplex) {
    //                 if(!std::isfinite(std::real(elem)) or !std::isfinite(std::imag(elem))) return false;
    //             } else {
    //                 if(!std::isfinite(elem)) return false;
    //             }
    //         }
    //         return true;
    //     }
    //
    //     // Final downcast (do not use for intermediates)
    //     TensorType to_TensorType() const { return (hi + lo); }
    //
    //     template<typename T>
    //     decltype(auto) cast() const {
    //         static_assert(std::is_floating_point_v<RealScalar> or std::is_same_v<RealScalar, fp128>);
    //         if constexpr(std::is_same_v<Scalar, T>) {
    //             return (*this); // Returns the exact same object as a reference
    //         } else {
    //             // Cast between different precisions, e.g. fp64 to fp32
    //             return x2::Tensor<T, rank>(hi.template cast<T>(), lo.template cast<T>());
    //         }
    //     }
    //
    //     // Cheap renormalization: enforce hi carries the leading bits
    //     void renorm() {
    //         // two_sum per entry: (hi, lo) := hi + lo exactly split back into hi+lo
    //         // Vectorized-ish form:
    //         TensorType s = (hi + lo);
    //         TensorType e = (lo - (s - hi));
    //         hi           = s;
    //         lo           = e;
    //         assert(allFinite());
    //     }
    //
    //     void shuffle(const Eigen::array<int, rank> &perm) {
    //         hi = Eigen::Tensor<Scalar, rank>(hi.shuffle(perm));
    //         lo = Eigen::Tensor<Scalar, rank>(lo.shuffle(perm));
    //         assert(allFinite());
    //     }
    //
    //     void conjugate() {
    //         hi = Tensor<Scalar, rank>(hi.conjugate());
    //         lo = Tensor<Scalar, rank>(lo.conjugate());
    //     }
    //     // Frobenius norm of (hi + lo)
    //     RealScalar norm() const {
    //         assert(hi.size() == lo.size());
    //         return x2::norm(hi.data(), lo.data(), hi.size());
    //     }
    // };

    // template<typename Scalar_, Eigen::Index rank>
    // struct TensorMap {
    //     using Scalar     = Scalar_;
    //     using RealScalar = Eigen::NumTraits<Scalar>::Real;
    //     using TensorType = Eigen::Tensor<Scalar, rank>;
    //     Eigen::TensorMap<TensorType> hi;
    //     Eigen::TensorMap<TensorType> lo;
    //
    //     TensorMap(Scalar *hi_ptr, Scalar *lo_ptr, Eigen::array<Eigen::Index, rank> dims) : hi(hi_ptr, dims), lo(lo_ptr, dims) {}
    //     TensorMap(TensorType &t) : hi(t.hi.data(), t.hi.dimensions()), lo(t.lo.data(), t.lo.dimensions()) {}
    //
    //     TensorMap &operator=(const Tensor<Scalar, rank> &other) {
    //         assert(this->dimensions() == other.dimensions());
    //         assert(this->lo.dimensions() == other.hi.dimensions());
    //         this->hi = other.hi;
    //         this->lo = other.lo;
    //         return *this;
    //     }
    //
    //     // Cheap renormalization: enforce hi carries the leading bits
    //     void renorm() {
    //         // two_sum per entry: (hi, lo) := hi + lo exactly split back into hi+lo
    //         // Vectorized-ish form:
    //         TensorType s = (hi + lo);
    //         TensorType e = (lo - (s - hi));
    //         hi           = s;
    //         lo           = e;
    //     }
    //     Eigen::array<Eigen::Index, rank> dimensions() const {
    //         assert(hi.dimensions() == lo.dimensions());
    //         return hi.dimensions();
    //     }
    //     Eigen::Index dimension(Eigen::Index n) const {
    //         assert(hi.dimension(n) == lo.dimension(n));
    //         return hi.dimension(n);
    //     }
    //
    //     RealScalar norm() const {
    //         assert(hi.size() == lo.size());
    //         return x2::norm(hi.data(), lo.data(), hi.size());
    //     }
    // };
    //
    // template<typename Scalar_, Eigen::Index rank>
    // struct ConstTensorMap {
    //     using Scalar     = Scalar_;
    //     using RealScalar = Eigen::NumTraits<Scalar>::Real;
    //     using TensorType = Eigen::Tensor<Scalar, rank>;
    //     Eigen::TensorMap<const TensorType> hi;
    //     Eigen::TensorMap<const TensorType> lo;
    //
    //     ConstTensorMap(const Scalar *hi_ptr, const Scalar *lo_ptr, Eigen::array<Eigen::Index, rank> dims) : hi(hi_ptr, dims), lo(lo_ptr, dims) {}
    //     ConstTensorMap(const TensorType &t) : hi(t.hi.data(), t.hi.dimensions()), lo(t.lo.data(), t.lo.dimensions()) {}
    //
    //     Eigen::array<Eigen::Index, rank> dimensions() const {
    //         assert(hi.dimensions() == lo.dimensions());
    //         return hi.dimensions();
    //     }
    //     Eigen::Index dimension(Eigen::Index n) const {
    //         assert(hi.dimension(n) == lo.dimension(n));
    //         return hi.dimension(n);
    //     }
    //     RealScalar norm() const {
    //         assert(hi.size() == lo.size());
    //         return x2::norm(hi.data(), lo.data(), hi.size());
    //     }
    // };

}