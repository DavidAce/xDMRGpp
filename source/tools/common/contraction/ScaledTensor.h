#pragma once

#include "math/tenx.h"

template<typename Scalar, long rank>
class ScaledMutableTensorMap {
    private:
    using RealScalar                         = decltype(std::real(std::declval<Scalar>()));
    Scalar                          *ptr_raw = nullptr;
    Eigen::array<Eigen::Index, rank> dims    = {};
    int                              ex2     = 0; // We rescale by 2^ex2
    long                             nelems  = 0;

    public:
    ScaledMutableTensorMap(Scalar *ptr, Eigen::array<Eigen::Index, rank> dims_, int ex2_ = 0) : ptr_raw(ptr), dims(dims_), ex2(ex2_) {
        nelems = 1;
        for(int i = 0; i < rank; i++) nelems *= dims[i];
        assert(ptr_raw != nullptr || nelems == 0);
    }

    void rescale_in_place() {
        // Apply scaling in place
        assert(ptr_raw != nullptr);
        auto       vector = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(ptr_raw, nelems);
        RealScalar maxval = vector.cwiseAbs().maxCoeff();
        if(maxval == RealScalar{0}) return;
        assert(std::isfinite(maxval));
        int                   p2expn = 0;
        [[maybe_unused]] auto maxnew = std::frexp(maxval, &p2expn); // p2expn in we get maxval == maxnew * 2^(p2expn)
        if(std::abs(p2expn) > 1) {                                  // Only rescale if the shift is at least 2
            vector *= std::ldexp(RealScalar{1}, -p2expn);           // Rescale
            ex2    += p2expn;                                       // Accumulate the exponent
        }
    }

    void unscale_in_place() {
        // Undo any scaling in place
        assert(ptr_raw != nullptr);
        if(ex2 == 0) return;
        if(nelems == 0) {
            ex2 = 0;
            return;
        }

        auto vector  = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(ptr_raw, nelems);
        vector      *= std::ldexp(RealScalar{1}, ex2); // Unscale
        ex2          = 0;                              // Reset the exponent
    }

    Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>> get_tensor() const {
        assert(ptr_raw != nullptr);
        return Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>>(ptr_raw, dims);
    }

    Eigen::TensorMap<Eigen::Tensor<Scalar, rank>> get_tensor() {
        assert(ptr_raw != nullptr);
        return Eigen::TensorMap<Eigen::Tensor<Scalar, rank>>(ptr_raw, dims);
    }
    Eigen::Index                     size() const { return nelems; }
    Eigen::array<Eigen::Index, rank> dimensions() const { return dims; }
    Eigen::Index                     dimension(Eigen::Index idx) const { return dims[idx]; }
    const Scalar                    *data() const { return ptr_raw; }
    Scalar                          *data() { return ptr_raw; }
    int                              get_exponent() const { return ex2; }
    void                             set_exponent(int p2expn) { ex2 = p2expn; }
    void                             add_exponent(int p2expn) { ex2 += p2expn; }
};

template<typename Scalar, long rank>
class ScaledConstTensorMap {
    private:
    using RealScalar = decltype(std::real(std::declval<Scalar>()));

    // External (non-owning) view
    const Scalar                    *cptr   = nullptr;
    Eigen::array<Eigen::Index, rank> dims   = {};
    long                             nelems = 0;

    // Optional owned copy (only allocated if rescaling is actually applied)
    std::vector<Scalar> storage;

    // Exponent so that: original = current * 2^ex2
    // If storage is active, "current" refers to storage.
    // If storage inactive, "current" refers to cptr (unmodified), so ex2 must be 0.
    int ex2 = 0;

    const Scalar *active_ptr() const {
        if(!storage.empty()) return storage.data();
        return cptr;
    }

    public:
    ScaledConstTensorMap() = delete;

    ScaledConstTensorMap(const Scalar *ptr, const Eigen::array<Eigen::Index, rank> &dims_) : cptr(ptr), dims(dims_) {
        nelems = 1;
        for(int i = 0; i < rank; ++i) nelems *= static_cast<long>(dims[i]);
        assert(cptr != nullptr || nelems == 0);
    }

    // True if we currently use owned storage (i.e., scaling has been applied)
    bool is_owned() const { return !storage.empty(); }

    // The main operation: only copies if it actually decides to scale.
    // min_shift = 2 means "only scale if abs(p2expn) > 1" (your current rule).
    void rescale_if_needed() {
        if(nelems == 0) return;
        assert(cptr != nullptr);

        // If already owned+scaled, you can choose to rescale again, but usually you do not want that.
        // I’ll make it a no-op if already owned.
        if(is_owned()) return;

        // Scan maxabs external data
        auto       vec    = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(cptr, nelems);
        RealScalar maxval = vec.cwiseAbs().maxCoeff();
        if(maxval == RealScalar{0}) return;
        assert(std::isfinite(maxval));

        int p2expn = 0;
        (void) std::frexp(maxval, &p2expn);

        if(std::abs(p2expn) <= 1) {
            // No scaling needed: remain non-owning, ex2 stays 0
            return;
        }

        // Allocate/copy/scale
        storage.resize(static_cast<std::size_t>(nelems));
        std::copy(cptr, cptr + nelems, storage.data());

        auto vector  = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(storage.data(), nelems);
        vector      *= std::ldexp(RealScalar{1}, -p2expn);

        ex2 = p2expn;
    }

    // Undo scaling. Since the external tensor was never modified, we can simply discard the owned copy and reset exponent.
    void unscale() {
        storage.clear();
        storage.shrink_to_fit(); // optional
        ex2 = 0;
    }
    Eigen::Index size() const { return nelems; }

    Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>> get_tensor() const {
        assert(active_ptr() != nullptr || nelems == 0);
        return Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>>(active_ptr(), dims);
    }

    Eigen::array<Eigen::Index, rank> dimensions() const { return dims; }
    Eigen::Index                     dimension(Eigen::Index idx) const { return dims[idx]; }
    const Scalar                    *data() const { return active_ptr(); }

    int get_exponent() const { return ex2; }
    // void set_exponent(int p2expn) { ex2 = p2expn; }
    // void add_exponent(int p2expn) { ex2 += p2expn; }
};
