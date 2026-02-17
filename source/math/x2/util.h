#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
namespace x2 {

    template<typename Scalar_>
    inline static typename Eigen::NumTraits<Scalar_>::Real norm(const Scalar_ *const hi_ptr, const Scalar_ *const lo_ptr, Eigen::Index n) {
        // Frobenius norm of (hi + lo)
        using Scalar     = Scalar_;
        using RealScalar = Eigen::NumTraits<Scalar>::Real;

        if(n == 0) return RealScalar{0};

        using RealAcc = std::conditional_t<sizeof(RealScalar) <= sizeof(double), long double, fp128>;
        RealAcc sum   = RealAcc{0};

#pragma omp parallel for reduction(+ : sum) schedule(static)
        for(Eigen::Index i = 0; i < n; ++i) {
            const Scalar a = hi_ptr[i];
            const Scalar b = lo_ptr[i];

            if constexpr(std::is_floating_point_v<Scalar>) {
                const RealAcc ar = static_cast<RealAcc>(a);
                const RealAcc br = static_cast<RealAcc>(b);

                const RealAcc aa = ar * ar;
                const RealAcc bb = br * br;
                const RealAcc ab = ar * br;

                sum += aa + RealAcc{2} * ab + bb;
            } else {
                // |a|^2 + 2 Re(conj(a)b) + |b|^2
                const RealAcc ar = static_cast<RealAcc>(std::real(a));
                const RealAcc ai = static_cast<RealAcc>(std::imag(a));
                const RealAcc br = static_cast<RealAcc>(std::real(b));
                const RealAcc bi = static_cast<RealAcc>(std::imag(b));

                const RealAcc aa = ar * ar + ai * ai;
                const RealAcc bb = br * br + bi * bi;
                const RealAcc ab = ar * br + ai * bi; // Re(conj(a)*b)

                sum += aa + RealAcc{2} * ab + bb;
            }
        }

        if(sum < RealAcc{0}) sum = RealAcc{0}; // guard tiny negative from rounding
        return static_cast<RealScalar>(std::sqrt(sum));
    }
}
