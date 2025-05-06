#pragma once
#include "../MpoSite.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include "math/tenx.h"
#include "qm/qm.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <config/settings.h>
#include <utility>

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::get_parity_shifted_mpo(const Eigen::Tensor<T, 4> &mpo_build) const {
    if(std::abs(parity_shift_sign_mpo) != 1) return mpo_build;
    if(parity_shift_axus_mpo.empty()) return mpo_build;
    // This redefines H --> H - r*Q(σ), where
    //      * Q(σ) = 0.5 * ( I - q*prod(σ) )
    //      * σ is a pauli matrix (usually σ^z)
    //      * 0.5 is a scalar that we multiply on the left edge as well.
    //      * r is the shift direction depending on the ritz (target energy): ground state energy (r = -1, SR) or maximum energy state (r = +1, LR).
    //        We multiply r just once on the left edge.
    //      * q == parity_shift_sign_mpo is the sign of the parity sector we want to shift away from the target sector we are interested in.
    //        Often this is done to resolve a degeneracy. We multiply q just once on the left edge.
    // We add (I - q*prod(σ)) along the diagonal of the MPO
    //        MPO = |1  0  0  0|
    //              |h  1  0  0|
    //              |0  0  1  0|
    //              |0  0  0  σ|
    //
    // Example 1: Let q == +1 (e.g. because target_axis=="+z"), then
    //            Q(σ)|ψ+⟩ = 0.5*(1 - q *(+1)) |ψ+⟩ = 0|ψ+⟩
    //            Q(σ)|ψ-⟩ = 0.5*(1 - q *(-1)) |ψ-⟩ = 1|ψ-⟩
    //
    // Example 2: For fDMRG with r == -1 (ritz == SR: min energy state)  we can add the projection on H directly, as
    //                  H --> (H - r*Q(σ))
    //            Then, if q == +1 we get and
    //                  (H - r*Q(σ)) |ψ+⟩ = (E + 0.5(1-1)) |ψ+⟩ = (E + 0) |ψ+⟩ <--- min energy state
    //                  (H - r*Q(σ)) |ψ-⟩ = (E + 0.5(1+1)) |ψ-⟩ = (E + 1) |ψ-⟩
    //            If q == -1 instead we get
    //                  (H - r*Q(σ)) |ψ+⟩ = (E + 0.5(1+1)) |ψ+⟩ = (E + 1) |ψ+⟩
    //                  (H - r*Q(σ)) |ψ-⟩ = (E + 0.5(1-1)) |ψ-⟩ = (E + 0) |ψ-⟩ <--- min energy state
    //
    // Example 3: For fDMRG with r == +1 (ritz == LR: max energy state) we can add the projection on H directly, as
    //                  H --> (H - r*Q(σ))
    //            Then, if q == +1 we get
    //                  (H - r*Q(σ)) |ψ+⟩ = (E - 0.5(1-1)) |ψ+⟩ = (E - 0) |ψ+⟩ <--- max energy state
    //                  (H - r*Q(σ)) |ψ-⟩ = (E - 0.5(1+1)) |ψ-⟩ = (E - 1) |ψ-⟩
    //            If q == -1 instead we get
    //                  (H - r*Q(σ)) |ψ+⟩ = (E - 0.5(1+1)) |ψ+⟩ = (E - 1) |ψ+⟩
    //                  (H - r*Q(σ)) |ψ-⟩ = (E - 0.5(1-1)) |ψ-⟩ = (E - 0) |ψ-⟩ <--- max energy state
    //
    //
    auto d0 = mpo_build.dimension(0);
    auto d1 = mpo_build.dimension(1);
    auto d2 = mpo_build.dimension(2);
    auto d3 = mpo_build.dimension(3);
    auto id = tenx::asScalarType<T>(qm::spin::half::tensor::id);
    auto pl = tenx::asScalarType<T>(qm::spin::half::tensor::get_pauli(parity_shift_axus_mpo));

    Eigen::Tensor<T, 4> mpo_with_parity_shift_op(d0 + 2, d1 + 2, d2, d3);
    mpo_with_parity_shift_op.setZero();
    mpo_with_parity_shift_op.slice(tenx::array4{0, 0, 0, 0}, mpo_build.dimensions())             = mpo_build;
    mpo_with_parity_shift_op.slice(tenx::array4{d0, d1, 0, 0}, extent4).reshape(extent2)         = id;
    mpo_with_parity_shift_op.slice(tenx::array4{d0 + 1, d1 + 1, 0, 0}, extent4).reshape(extent2) = pl;
    return mpo_with_parity_shift_op;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO_edge_left(const Eigen::Tensor<T, 4> &mpo) const {
    using Real = decltype(std::real(std::declval<T>()));
    if(mpo.size() == 0) throw except::runtime_error("mpo({}): can't build the left edge: mpo has not been built yet", get_position());
    auto                ldim = mpo.dimension(0);
    Eigen::Tensor<T, 1> ledge(ldim);
    if(ldim == 1) {
        // Thin edge (it was probably already applied to the left-most MPO)
        ledge.setConstant(T{1});
    } else {
        ledge.setZero();
        if(parity_shift_sign_mpo == 0) {
            /*
             *  MPO = |1 0|
             *        |h 1|
             *  So the left edge picks out the row for h
             */
            ledge(ldim - 1) = 1;
        } else {
            /*
             *  MPO = |1  0  0  0|
             *        |h  1  0  0|
             *        |0  0  1  0|
             *        |0  0  0  σ|
             *  So the left edge picks out the row for h, as well as 1 and σ along the diagonal
             *  We also put the signs r,q and factor 0.5 if this is the first site.
             */
            // This redefines H --> H - r*Q(σ), where
            //      * Q(σ) = 0.5 * ( I - q*prod(σ) )
            //      * σ is a pauli matrix (usually σ^z)
            //      * 0.5 is a scalar that we multiply on the left edge as well.
            //      * r is the shift direction depending on the ritz (target energy): ground state energy (r = -1, SR) or maximum energy state (r = +1, LR).
            //        We multiply r just once on the left edge.
            //      * q == parity_shift_sign_mpo is the sign of the parity sector we want to shift away from the target sector we are interested in.
            //        Often this is done to resolve a degeneracy. We multiply q just once on the left edge.
            double a = 1.0; // The shift amount. Use a large amount to make sure the target energy in that axis is actually extremal
            double q = 1.0; // The parity sign to shift
            double r = 1.0; // The energy shift direction (-1 is down, +1 is up)

            if(position == 0) {
                a = global_energy_upper_bound;
                q = -parity_shift_sign_mpo;
                switch(parity_shift_ritz_mpo) {
                    case OptRitz::SR: r = -1.0; break;
                    case OptRitz::LR: r = +1.0; break;
                    case OptRitz::SM: r = -1.0; break;
                    case OptRitz::NONE: r = 0.0; break;
                    // case OptRitz::LM: r = -1.0; break; // TODO this one probably doesn't make sense
                    default: throw except::runtime_error("expected ritz SR or LR, got: {}", enum2sv(parity_shift_ritz_mpo));
                }
            }

            if(get_position() == 0)
                tools::log->trace("Shifting MPO energy of the {:+}{} parity sector by E -> E{:+.8f} (estimated Emax)", q, parity_shift_axus_mpo, a);
            ledge(ldim - 3) = static_cast<Real>(1.0); // The bottom left corner
            ledge(ldim - 2) = static_cast<Real>(-a * r * 0.5);
            ledge(ldim - 1) = static_cast<Real>(-a * r * 0.5 * q);
        }
    }
    return ledge;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO_edge_right(const Eigen::Tensor<T, 4> &mpo) const {
    using Real = decltype(std::real(std::declval<T>()));
    if(mpo.size() == 0) throw except::runtime_error("mpo({}): can't build the right edge: mpo has not been built yet", get_position());
    auto                rdim = mpo.dimension(1);
    Eigen::Tensor<T, 1> redge(rdim);
    if(rdim == 1) {
        // Thin edge (it was probably already applied to the right-most MPO
        redge.setConstant(T{1});
    } else {
        redge.setZero();

        if(parity_shift_sign_mpo == 0) {
            /*
             *  MPO = |1 0|
             *        |h 1|
             *  So the right edge picks out the column for h
             */
            redge(0) = 1; // The bottom left corner
        } else {
            /*
             *  MPO = |1  0  0  0|
             *        |h  1  0  0|
             *        |0  0  1  0|
             *        |0  0  0  σ|
             *  So the right edge picks out the column for h, as well as 1 and σ along the diagonal
             */
            redge(0)        = static_cast<Real>(1.0); // The bottom left corner of the original non-parity-shifted mpo
            redge(rdim - 2) = static_cast<Real>(1.0);
            redge(rdim - 1) = static_cast<Real>(1.0);
        }
    }
    return redge;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::apply_edge_left(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeL) const {
    if(mpo.dimension(0) == 1 or get_position() != 0) return mpo;
    if(mpo.dimension(0) != edgeL.dimension(0))
        throw except::logic_error("apply_edge_left: dimension mismatch: mpo {} | edgeL {}", mpo.dimensions(), edgeL.dimensions());
    auto  tmp     = mpo;
    auto  dim     = tmp.dimensions();
    auto &threads = tenx::threads::get();
    tmp.resize(tenx::array4{1, dim[1], dim[2], dim[3]});
    tmp.device(*threads->dev) = edgeL.reshape(tenx::array2{1, edgeL.size()}).contract(mpo, tenx::idx({1}, {0}));
    return tmp;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 4> MpoSite<Scalar>::apply_edge_right(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeR) const {
    if(mpo.dimension(1) == 1 or get_position() + 1 != settings::model::model_size) return mpo;
    if(mpo.dimension(1) != edgeR.dimension(0))
        throw except::logic_error("apply_edge_right: dimension mismatch: mpo {} | edgeR {}", mpo.dimensions(), edgeR.dimensions());
    auto  tmp     = mpo;
    auto  dim     = tmp.dimensions();
    auto &threads = tenx::threads::get();
    tmp.resize(tenx::array4{dim[0], 1, dim[2], dim[3]});
    tmp.device(*threads->dev) = mpo.contract(edgeR.reshape(tenx::array2{edgeR.size(), 1}), tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
    return tmp;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO2_edge_left() const {
    using Real = decltype(std::real(std::declval<T>()));
    if(mpo_squared.has_value()) {
        auto ldim = mpo_squared->dimension(0);
        if(ldim == 1) {
            // Thin edge (it was probably already applied to the right-most MPO
            auto ledge2 = Eigen::Tensor<T, 1>(ldim);
            ledge2.setConstant(T{1});
            return ledge2;
        }
    }
    /* Start by making a left edge that would fit a raw mpo
     *  MPO = |1 0|
     *        |h 1|
     *  The left edge should pick out the last row
     */
    auto mpo1  = get_mpo(Scalar{0});
    auto d0    = mpo1.dimension(0);
    auto ledge = Eigen::Tensor<T, 1>(d0);
    ledge.setZero();
    ledge(d0 - 1)              = 1;
    Eigen::Tensor<T, 1> ledge2 = ledge.contract(ledge, tenx::idx()).reshape(tenx::array1{d0 * d0});
    if(std::abs(parity_shift_sign_mpo2) != 1.0) return ledge2;

    double q = 1.0;
    double a = 1.0;                               // The shift amount (select 1.0 to shift up by 1.0)
    if(position == 0) q = parity_shift_sign_mpo2; // Selects the opposite sector sign (only needed on one MPO)
    auto ledge2_with_shift = Eigen::Tensor<T, 1>(d0 * d0 + 2);
    ledge2_with_shift.setZero();
    ledge2_with_shift.slice(tenx::array1{0}, ledge2.dimensions()) = ledge2;
    ledge2_with_shift(d0 * d0 + 0)                                = static_cast<Real>(a * 0.5);        // 1.0;
    ledge2_with_shift(d0 * d0 + 1)                                = static_cast<Real>(a * 0.5 * (-q)); // 1.0;

    return ledge2_with_shift;
}

template<typename Scalar>
template<typename T>
Eigen::Tensor<T, 1> MpoSite<Scalar>::get_MPO2_edge_right() const {
    if(mpo_squared.has_value()) {
        auto rdim = mpo_squared->dimension(1);
        if(rdim == 1) {
            // Thin edge (it was probably already applied to the right-most MPO
            auto redge2 = Eigen::Tensor<T, 1>(rdim);
            redge2.setConstant(1.0);
            return redge2;
        }
    }
    /* Start by making a right edge that would fit a raw mpo
     *  MPO = |1 0|
     *        |h 1|
     *  The right edge should pick out the first column
     */
    auto mpo1  = get_mpo(Scalar{0});
    auto d0    = mpo1.dimension(1);
    auto redge = Eigen::Tensor<T, 1>(d0);
    redge.setZero();
    redge(0)                   = 1;
    Eigen::Tensor<T, 1> redge2 = redge.contract(redge.conjugate(), tenx::idx()).reshape(tenx::array1{d0 * d0});
    if(std::abs(parity_shift_sign_mpo2) != 1.0) return redge2;
    auto redge2_with_shift = Eigen::Tensor<T, 1>(d0 * d0 + 2);
    redge2_with_shift.setZero();
    redge2_with_shift.slice(tenx::array1{0}, redge2.dimensions()) = redge2;
    redge2_with_shift(d0 * d0 + 0)                                = T{1}; // 0.5;
    redge2_with_shift(d0 * d0 + 1)                                = T{1}; // 0.5 * q;
    return redge2_with_shift;
}