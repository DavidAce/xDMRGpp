#pragma once
#include "tensors/state/StateInfinite.h"
#include "config/settings.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/common/split.h"
#include "tools/common/views.h"
#include "tools/infinite/measure.h"
#include "tools/infinite/mps.h"



template<typename Scalar>
StateInfinite<Scalar>::StateInfinite() : MPS_A(std::make_unique<MpsSite<Scalar>>()), MPS_B(std::make_unique<MpsSite<Scalar>>()) {
    tools::log->trace("Constructing state");
}

// We need to define the destructor and other special functions
// because we enclose data in unique_ptr for this pimpl idiom.
// Otherwise, unique_ptr will forcibly inline its own default deleter.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
template<typename Scalar>
StateInfinite<Scalar>::~StateInfinite() = default; // default dtor
template<typename Scalar>
StateInfinite<Scalar>::StateInfinite(StateInfinite &&other) noexcept = default; // default move ctor
template<typename Scalar>
StateInfinite<Scalar> &StateInfinite<Scalar>::operator=(StateInfinite &&other) noexcept = default; // default move assign

/* clang-format off */
template<typename Scalar>
StateInfinite<Scalar>::StateInfinite(const StateInfinite &other) noexcept :
    MPS_A(std::make_unique<MpsSite<Scalar>>(*other.MPS_A)),
    MPS_B(std::make_unique<MpsSite<Scalar>>(*other.MPS_B)),
    swapped(other.swapped),
    cache(other.cache),
    name(other.name),
    algo(other.algo),
    measurements(other.measurements),
    lowest_recorded_variance(other.lowest_recorded_variance){
}
template<typename Scalar>
StateInfinite<Scalar> & StateInfinite<Scalar>::operator=(const StateInfinite &other) noexcept{
    if(this == &other) return *this;
    MPS_A                    = std::make_unique<MpsSite<Scalar>>(*other.MPS_A);
    MPS_B                    = std::make_unique<MpsSite<Scalar>>(*other.MPS_B);
    swapped                  = other.swapped;
    cache                    = other.cache;
    name                     = other.name;
    algo                     = other.algo;
    measurements             = other.measurements;
    lowest_recorded_variance = other.lowest_recorded_variance;
    return *this;
}

/* clang-format on */

template<typename Scalar>
void StateInfinite<Scalar>::initialize(ModelType model_type) {
    tools::log->trace("Initializing state");
    long spin_dim = 2;
    switch(model_type) {
        case ModelType::ising_tf_rf: spin_dim = settings::model::ising_tf_rf::spin_dim; break;
        case ModelType::ising_sdual: spin_dim = settings::model::ising_sdual::spin_dim; break;
        default: spin_dim = 2;
    }
    Eigen::Tensor<Scalar, 3> M(spin_dim, 1, 1);
    Eigen::Tensor<Scalar, 1> L(1);
    // Default is a product state, spins pointing up in z.
    M.setZero();
    M(0, 0, 0) = 1;
    L.setConstant(1.0);
    MPS_A = std::make_unique<MpsSite<Scalar>>(M, L, 0, 0, "A");
    MPS_B = std::make_unique<MpsSite<Scalar>>(M, L, 1, 0, "B");
    MPS_A->set_LC(L);
}

template<typename Scalar>
void StateInfinite<Scalar>::set_name(std::string_view statename) {
    name = statename;
}
template<typename Scalar>
std::string StateInfinite<Scalar>::get_name() const {
    return name;
}

template<typename Scalar>
void StateInfinite<Scalar>::set_algorithm(const AlgorithmType &algo_type) {
    algo = algo_type;
}
template<typename Scalar>
AlgorithmType StateInfinite<Scalar>::get_algorithm() const {
    return algo;
}

template<typename Scalar>
std::pair<size_t, size_t> StateInfinite<Scalar>::get_positions() {
    return std::make_pair(MPS_A->get_position(), MPS_B->get_position());
}
template<typename Scalar>
size_t StateInfinite<Scalar>::get_positionA() {
    return MPS_A->get_position();
}
template<typename Scalar>
size_t StateInfinite<Scalar>::get_positionB() {
    return MPS_B->get_position();
}

template<typename Scalar>
long StateInfinite<Scalar>::chiC() const {
    return MPS_A->get_LC().dimension(0);
}
template<typename Scalar>
long StateInfinite<Scalar>::chiA() const {
    return MPS_A->get_L().dimension(0);
}
template<typename Scalar>
long StateInfinite<Scalar>::chiB() const {
    return MPS_B->get_L().dimension(0);
}

template<typename Scalar>
long StateInfinite<Scalar>::get_spin_dimA() const {
    return MPS_A->spin_dim();
}
template<typename Scalar>
long StateInfinite<Scalar>::get_spin_dimB() const {
    return MPS_B->spin_dim();
}
template<typename Scalar>
double StateInfinite<Scalar>::get_truncation_error() const {
    // Should get the current limit on allowed bond dimension for the duration of the simulation
    return get_mps_siteA().get_truncation_error();
}

template<typename Scalar>
Eigen::DSizes<long, 3> StateInfinite<Scalar>::dimensions() const {
    return Eigen::DSizes<long, 3>{get_spin_dimA() * get_spin_dimB(), chiA(), chiB()};
}

template<typename Scalar>
const MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_siteA() const {
    return *MPS_A;
}
template<typename Scalar>
const MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_siteB() const {
    return *MPS_B;
}
template<typename Scalar>
MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_siteA() {
    return *MPS_A;
}
template<typename Scalar>
MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_siteB() {
    return *MPS_B;
}
template<typename Scalar>
const MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_site(size_t pos) const {
    if(pos == 0) return *MPS_A;
    if(pos == 1) return *MPS_B;
    throw except::runtime_error("Got wrong site position {}. Expected 0 or 1", pos);
}
template<typename Scalar>
MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_site(size_t pos) {
    if(pos == 0) return *MPS_A;
    if(pos == 1) return *MPS_B;
    throw except::runtime_error("Got wrong site position {}. Expected 0 or 1", pos);
}
template<typename Scalar>
const MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_site(std::string_view pos) const {
    if(pos == "A") return *MPS_A;
    if(pos == "B") return *MPS_B;
    throw except::runtime_error("Got wrong site position {}. Expected 0 or 1", pos);
}
template<typename Scalar>
MpsSite<Scalar> &StateInfinite<Scalar>::get_mps_site(std::string_view pos) {
    if(pos == "A") return *MPS_A;
    if(pos == "B") return *MPS_B;
    throw except::runtime_error("Got wrong site position {}. Expected 0 or 1", pos);
}

/* clang-format off */
template<typename Scalar> const Eigen::Tensor<Scalar, 3> &StateInfinite<Scalar>::A_bare() const { return MPS_A->get_M_bare(); }
template<typename Scalar> const Eigen::Tensor<Scalar, 3> &StateInfinite<Scalar>::A() const { return MPS_A->get_M(); }
template<typename Scalar> const Eigen::Tensor<Scalar, 3> &StateInfinite<Scalar>::B() const { return MPS_B->get_M(); }
template<typename Scalar> const Eigen::Tensor<Scalar, 2> &StateInfinite<Scalar>::LC_diag() const { if(cache.LC_diag) return cache.LC_diag.value(); else cache.LC_diag = tenx::asDiagonal(MPS_A->get_LC()); return cache.LC_diag.value();}
template<typename Scalar> const Eigen::Tensor<Scalar, 2> &StateInfinite<Scalar>::LA_diag() const { if(cache.LA_diag) return cache.LA_diag.value(); else cache.LA_diag = tenx::asDiagonal(MPS_A->get_L()); return cache.LA_diag.value();}
template<typename Scalar> const Eigen::Tensor<Scalar, 2> &StateInfinite<Scalar>::LB_diag() const { if(cache.LB_diag) return cache.LB_diag.value(); else cache.LB_diag = tenx::asDiagonal(MPS_B->get_L()); return cache.LB_diag.value();}
template<typename Scalar> const Eigen::Tensor<Scalar, 2> &StateInfinite<Scalar>::LC_diag_inv() const { if(cache.LC_diag_inv) return cache.LC_diag_inv.value(); else cache.LC_diag_inv = tenx::asDiagonalInversed(MPS_A->get_LC()); return cache.LC_diag_inv.value();}
template<typename Scalar> const Eigen::Tensor<Scalar, 2> &StateInfinite<Scalar>::LA_diag_inv() const { if(cache.LA_diag_inv) return cache.LA_diag_inv.value(); else cache.LA_diag_inv = tenx::asDiagonalInversed(MPS_A->get_L()); return cache.LA_diag_inv.value();}
template<typename Scalar> const Eigen::Tensor<Scalar, 2> &StateInfinite<Scalar>::LB_diag_inv() const { if(cache.LB_diag_inv) return cache.LB_diag_inv.value(); else cache.LB_diag_inv = tenx::asDiagonalInversed(MPS_B->get_L()); return cache.LB_diag_inv.value();}
template<typename Scalar> const Eigen::Tensor<Scalar, 1> &StateInfinite<Scalar>::LC() const { return MPS_A->get_LC(); }
template<typename Scalar> const Eigen::Tensor<Scalar, 1> &StateInfinite<Scalar>::LA() const { return MPS_A->get_L(); }
template<typename Scalar> const Eigen::Tensor<Scalar, 1> &StateInfinite<Scalar>::LB() const { return MPS_B->get_L(); }

template<typename Scalar> const Eigen::Tensor<Scalar, 3> &StateInfinite<Scalar>::GA() const {
    if(cache.GA) return cache.GA.value();
    else {
        Eigen::Tensor<Scalar,1> L_inv = MPS_A->get_L().inverse();
        cache.GA = tools::common::contraction::contract_bnd_mps(L_inv, MPS_A->get_M_bare());
    }
    return cache.GA.value();
}
template<typename Scalar> const Eigen::Tensor<Scalar, 3> &StateInfinite<Scalar>::GB() const {
    if(cache.GB) return cache.GB.value();
    else{
        Eigen::Tensor<Scalar,1> L_inv = MPS_B->get_L().inverse();
        cache.GB = tools::common::contraction::contract_mps_bnd(MPS_B->get_M_bare(), L_inv);
    }
    return cache.GB.value();
}

/* clang-format on */

template<typename Scalar>
const Eigen::Tensor<Scalar, 3> &StateInfinite<Scalar>::get_2site_mps(Scalar norm) const {
    /*!
 * Returns a two-site tensor
     @verbatim
        1--[ LA ]--[ GA ]--[ LC ]-- [ GB ] -- [ LB ]--3
                     |                 |
                     0                 2
       which in our notation is simply A * B  (because A = LA * GA * LC and B = GB * LB),
       becomes

        1--[ 2site_tensor ]--2
                  |
                  0

     @endverbatim
 */

    if(cache.twosite_mps) return cache.twosite_mps.value();
    cache.twosite_mps = tools::common::contraction::contract_mps_mps(A(), B()) / norm;
    return cache.twosite_mps.value();
}

template<typename Scalar>
void StateInfinite<Scalar>::assert_validity() const {
    MPS_A->assert_validity();
    MPS_B->assert_validity();
}
template<typename Scalar>
bool StateInfinite<Scalar>::is_real() const {
    return MPS_A->is_real() and MPS_B->is_real();
}
template<typename Scalar>
bool StateInfinite<Scalar>::has_nan() const {
    return MPS_A->has_nan() or MPS_B->has_nan();
}
template<typename Scalar>
bool StateInfinite<Scalar>::is_swapped() const {
    return swapped;
}
//
// template<typename T>
// Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> StateInfinite<Scalar>::get_H_local_matrix() const {
//    Eigen::Tensor<T, 5> tempL;
//    Eigen::Tensor<T, 5> tempR;
//    if constexpr(std::is_same<T, double>::value) {
//        if(not Lblock->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from Lblock when building H_local");
//        }
//        if(not Rblock->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from Rblock when building H_local");
//        }
//        if(not HA->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from MPO A when building H_local");
//        }
//        if(not HB->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from MPO B when building H_local");
//        }
//        tempL = Lblock->block.contract(HA->MPO(), tenx::idx({2}, {0})).real().shuffle(tenx::array5{4, 1, 3, 0, 2}).real();
//        tempR = Rblock->block.contract(HB->MPO(), tenx::idx({2}, {1})).real().shuffle(tenx::array5{4, 1, 3, 0, 2}).real();
//    } else {
//        tempL = Lblock->block.contract(HA->MPO(), tenx::idx({2}, {0})).shuffle(tenx::array5{4, 1, 3, 0, 2});
//        tempR = Rblock->block.contract(HB->MPO(), tenx::idx({2}, {1})).shuffle(tenx::array5{4, 1, 3, 0, 2});
//    }
//    long                shape   = mps_sites->chiA() * mps_sites->spindim() * mps_sites->chiB() * mps_sites->spin_dim_A();
//    Eigen::Tensor<T, 8> H_local = tempL.contract(tempR, tenx::idx({4}, {4})).shuffle(tenx::array8{0, 1, 4, 5, 2, 3, 6, 7});
//    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(H_local.data(), shape, shape);
//}

// template Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>               StateInfinite<Scalar>::get_H_local_matrix<double>() const;
// template Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> StateInfinite<Scalar>::get_H_local_matrix<std::complex<double>>() const;
//
// template<typename T>
// Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> StateInfinite<Scalar>::get_H_local_sq_matrix() const {
//    Eigen::Tensor<T, 6> tempL;
//    Eigen::Tensor<T, 6> tempR;
//    if constexpr(std::is_same<T, double>::value) {
//        if(not Lblock2->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from Lblock2 when building H_local_sq");
//        }
//        if(not Rblock2->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from Rblock2 when building H_local_sq");
//        }
//        if(not HA->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from MPO A when building H_local_sq");
//        }
//        if(not HB->isReal()) {
//            throw std::runtime_error("Discarding imaginary data from MPO B when building H_local_sq");
//        }
//        tempL = Lblock2->block.contract(HA->MPO(), tenx::idx({2}, {0}))
//                    .contract(HA->MPO(), tenx::idx({2, 5}, {0, 2}))
//                    .real()
//                    .shuffle(tenx::array6{5, 1, 3, 0, 2, 4});
//        tempR = Rblock2->block.contract(HB->MPO(), tenx::idx({2}, {1}))
//                    .contract(HB->MPO(), tenx::idx({2, 5}, {1, 2}))
//                    .real()
//                    .shuffle(tenx::array6{5, 1, 3, 0, 2, 4});
//    } else {
//        tempL = Lblock2->block.contract(HA->MPO(), tenx::idx({2}, {0}))
//                    .contract(HA->MPO(), tenx::idx({2, 5}, {0, 2}))
//                    .shuffle(tenx::array6{5, 1, 3, 0, 2, 4});
//        tempR = Rblock2->block.contract(HB->MPO(), tenx::idx({2}, {1}))
//                    .contract(HB->MPO(), tenx::idx({2, 5}, {1, 2}))
//                    .shuffle(tenx::array6{5, 1, 3, 0, 2, 4});
//    }
//
//    long                shape   = mps_sites->chiA() * mps_sites->spindim() * mps_sites->chiB() * mps_sites->spin_dim_A();
//    Eigen::Tensor<T, 8> H_local = tempL.contract(tempR, tenx::idx({4, 5}, {4, 5})).shuffle(tenx::array8{0, 1, 4, 5, 2, 3, 6, 7});
//    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(H_local.data(), shape, shape);
//}

// template Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>               StateInfinite<Scalar>::get_H_local_sq_matrix<double>() const;
// template Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> StateInfinite<Scalar>::get_H_local_sq_matrix<std::complex<double>>() const;

// void StateInfinite<Scalar>::enlarge_environment(int direction) {
//    assert_positions();
//    auto position_A_new = mps_sites->MPS_A->get_position() + 1;
//    *Lblock             = Lblock->enlarge(*mps_sites->MPS_A, *HA);
//    *Rblock             = Rblock->enlarge(*mps_sites->MPS_B, *HB);
//    *Lblock2            = Lblock2->enlarge(*mps_sites->MPS_A, *HA);
//    *Rblock2            = Rblock2->enlarge(*mps_sites->MPS_B, *HB);
//    set_positions(position_A_new);
//    if(direction != 0) throw std::runtime_error("Ooops, direction != 0");
//    if (direction == 1){
//        *Lblock  = Lblock->enlarge(*mps_sites->MPS_A,  *HA);
//        *Lblock2 = Lblock2->enlarge(*mps_sites->MPS_A, *HA);
//        Lblock->set_position (HB->get_position());
//        Lblock2->set_position(HB->get_position());
//    }else if (direction == -1){
//        *Rblock  = Rblock->enlarge(*mps_sites->MPS_B,  *HB);
//        *Rblock2 = Rblock2->enlarge(*mps_sites->MPS_B, *HB);
//        Rblock->set_position (HA->get_position());
//        Rblock2->set_position(HA->get_position());
//    }else if(direction == 0){
//
//
//    }
//    clear_measurements();
//    assert_positions();
//}

template<typename Scalar>
void StateInfinite<Scalar>::set_positions(size_t position) {
    MPS_A->set_position(position);
    MPS_B->set_position(position + 1);
}

template<typename Scalar>
void StateInfinite<Scalar>::set_mps(const Eigen::Tensor<Scalar, 3> &twosite_tensor, MergeEvent mevent, std::optional<svd::config> svd_cfg) {
    tools::infinite::mps::merge_twosite_tensor(*this, twosite_tensor, mevent, svd_cfg);
}

template<typename Scalar>
void StateInfinite<Scalar>::set_mps(const std::vector<MpsSite<Scalar>> &mps_list) {
    if(mps_list.size() != 2) throw except::runtime_error("Expected 2 sites, got: {}", mps_list.size());
    const auto &mpsA = *std::next(mps_list.begin(), 0);
    const auto &mpsB = *std::next(mps_list.begin(), 1);
    set_mps(mpsA, mpsB);
    clear_cache();
}

template<typename Scalar>
void StateInfinite<Scalar>::set_mps(const MpsSite<Scalar> &mpsA, const MpsSite<Scalar> &mpsB) {
    if(not mpsA.isCenter()) throw std::runtime_error("Given mps for site A is not a center");
    MPS_A = std::make_unique<MpsSite<Scalar>>(mpsA);
    MPS_B = std::make_unique<MpsSite<Scalar>>(mpsB);
    clear_cache();
}

template<typename Scalar>
void StateInfinite<Scalar>::set_mps(const Eigen::Tensor<Scalar, 3> &MA, const Eigen::Tensor<Scalar, 1> &LC, const Eigen::Tensor<Scalar, 3> &MB) {
    MPS_A->set_M(MA);
    MPS_A->set_LC(LC);
    MPS_B->set_M(MB);
    clear_cache();
}
template<typename Scalar>
void StateInfinite<Scalar>::set_mps(const Eigen::Tensor<Scalar, 1> &LA, const Eigen::Tensor<Scalar, 3> &MA, const Eigen::Tensor<Scalar, 1> &LC,
                                    const Eigen::Tensor<Scalar, 3> &MB, const Eigen::Tensor<Scalar, 1> &LB) {
    MPS_A->set_mps(MA, LA, 0, "A");
    MPS_A->set_LC(LC);
    MPS_B->set_mps(MB, LB, 0, "B");
    clear_cache();
}

template<typename Scalar>
bool StateInfinite<Scalar>::is_limited_by_bond(long bond_lim) const {
    return chiC() >= bond_lim;
}
template<typename Scalar>
bool StateInfinite<Scalar>::is_truncated(double truncation_error_limit) const {
    return get_truncation_error() > truncation_error_limit;
}
template<typename Scalar>
void StateInfinite<Scalar>::clear_cache() const {
    cache = Cache();
}
template<typename Scalar>
void StateInfinite<Scalar>::clear_measurements() const {
    measurements                                      = MeasurementsStateInfinite<Scalar>();
    tools::common::views<Scalar>::components_computed = false;
}

template<typename Scalar>
void StateInfinite<Scalar>::swap_AB() {
    tools::log->trace("Swapping AB");
    swapped = !swapped;
    // Store the positions
    auto position_left  = MPS_A->get_position();
    auto position_right = MPS_B->get_position();

    // Swap Gamma
    Eigen::Tensor<Scalar, 1> LC = MPS_A->get_LC();
    MPS_A->unset_LC();
    MPS_B->unset_LC();
    MPS_A.swap(MPS_B);
    MPS_A->set_LC(MPS_A->get_L());
    MPS_A->set_L(LC);
    MPS_B->set_L(LC);
    MPS_A->set_position(position_left);
    MPS_B->set_position(position_right);
    clear_cache();
    clear_measurements();
}
