#include "../views.h"
#include "../log.h"
#include "math/cast.h"
#include "math/eig.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/state/StateInfinite.h"



template<eig::Side side, typename Scalar>
auto dominant_eig(const Eigen::Tensor<Scalar, 2> &transfer_mat, int L, int ncv) {
    eig::solver solver;
    solver.eigs(transfer_mat.data(), L, 1, ncv, eig::Ritz::LM, eig::Form::NSYM, side, std::nullopt, eig::Shinv::OFF, eig::Vecs::OFF, eig::Dephase::ON);
    auto eigvec = eig::view::get_eigvec<Scalar>(solver.result, 0, side).normalized();
    auto eigval = eig::view::get_eigval<Scalar>(solver.result, 0);
    return std::make_pair(eigvec, eigval);
}

template<typename Scalar>
void tools::common::views<Scalar>::compute_mps_components(const StateInfinite<Scalar> &state) {
    if(components_computed) return;
    // On even thetas we have  chiA = chiB on the outer bond
    // On odd thetas we have chiC on the outer bond.
    //    int chiA2 = (int)(state.MPS->chiA()*state.MPS->chiA());
    int chiB2 = safe_cast<int>(state.chiB() * state.chiB());
    int chiC2 = safe_cast<int>(state.chiC() * state.chiC());

    theta                                           = get_theta(state);
    Eigen::Tensor<Scalar, 2> theta_evn_transfer_mat = get_transfer_matrix_theta_evn(state).reshape(tenx::array2{chiB2, chiB2});
    Eigen::Tensor<Scalar, 2> theta_odd_transfer_mat = get_transfer_matrix_theta_odd(state).reshape(tenx::array2{chiC2, chiC2});

    //    int ncvA = std::min(16, chiA2);
    int ncvC                                           = std::min(16, chiC2);
    int ncvB                                           = std::min(16, chiB2);
    [[maybe_unused]] auto [eigvec_R_evn, eigval_R_evn] = dominant_eig<eig::Side::R>(theta_evn_transfer_mat, chiB2, ncvB);
    [[maybe_unused]] auto [eigvec_L_evn, eigval_L_evn] = dominant_eig<eig::Side::L>(theta_evn_transfer_mat, chiB2, ncvB);
    [[maybe_unused]] auto [eigvec_R_odd, eigval_R_odd] = dominant_eig<eig::Side::R>(theta_odd_transfer_mat, chiC2, ncvC);
    [[maybe_unused]] auto [eigvec_L_odd, eigval_L_odd] = dominant_eig<eig::Side::L>(theta_odd_transfer_mat, chiC2, ncvC);

    Scalar normalization_evn = std::sqrt(std::abs((eigvec_L_evn.transpose() * eigvec_R_evn).sum()));
    Scalar normalization_odd = std::sqrt(std::abs((eigvec_L_odd.transpose() * eigvec_R_odd).sum()));

    r_evn = tenx::TensorMap(eigvec_R_evn).reshape(tenx::array2{state.chiB(), state.chiB()}) / normalization_evn;
    l_evn = tenx::TensorMap(eigvec_L_evn).reshape(tenx::array2{state.chiB(), state.chiB()}) / normalization_evn;
    r_odd = tenx::TensorMap(eigvec_R_odd).reshape(tenx::array2{state.chiC(), state.chiC()}) / normalization_odd;
    l_odd = tenx::TensorMap(eigvec_L_odd).reshape(tenx::array2{state.chiC(), state.chiC()}) / normalization_odd;

    theta                = get_theta(state);
    theta_sw             = get_theta_swapped(state);
    theta_evn_normalized = get_theta_evn(state, sqrt(eigval_R_evn));
    theta_odd_normalized = get_theta_odd(state, sqrt(eigval_R_odd));

    LAGA = state.A_bare(); // Modified! May contain some error? Check git history for comparison
    LCGB = state.LC_diag()
               .contract(state.GB(), tenx::idx({1}, {1}))
               .shuffle(tenx::array3{1, 0, 2}); // Modified! May contain some error? Check git history for comparison

    transfer_matrix_evn = theta_evn_normalized.contract(theta_evn_normalized.conjugate(), tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array4{0, 2, 1, 3});
    transfer_matrix_odd = theta_odd_normalized.contract(theta_odd_normalized.conjugate(), tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array4{0, 2, 1, 3});
    Eigen::Tensor<Scalar, 4> transfer_matrix_LCGB_unnormalized = LCGB.contract(LCGB.conjugate(), tenx::idx({0}, {0})).shuffle(tenx::array4{0, 2, 1, 3});
    Eigen::Tensor<Scalar, 4> transfer_matrix_LAGA_unnormalized = LAGA.contract(LAGA.conjugate(), tenx::idx({0}, {0})).shuffle(tenx::array4{0, 2, 1, 3});

    Eigen::Tensor<Scalar, 0> l_evn_LBGA_r_odd =
        l_evn.contract(transfer_matrix_LAGA_unnormalized, tenx::idx({0, 1}, {0, 1})).contract(r_odd, tenx::idx({0, 1}, {0, 1}));
    Eigen::Tensor<Scalar, 0> l_odd_LCGB_r_evn =
        l_odd.contract(transfer_matrix_LCGB_unnormalized, tenx::idx({0, 1}, {0, 1})).contract(r_evn, tenx::idx({0, 1}, {0, 1}));

    transfer_matrix_LCGB = transfer_matrix_LCGB_unnormalized / l_odd_LCGB_r_evn(0);
    transfer_matrix_LAGA = transfer_matrix_LAGA_unnormalized / l_evn_LBGA_r_odd(0);
    LCGB                 = LCGB / sqrt(l_odd_LCGB_r_evn(0));
    LAGA                 = LAGA / sqrt(l_evn_LBGA_r_odd(0));
    components_computed  = true;
    if(tools::log->level() == spdlog::level::trace) {
        Eigen::Tensor<Scalar, 0> le_re     = l_evn.contract(r_evn, tenx::idx({0, 1}, {0, 1}));
        Eigen::Tensor<Scalar, 0> lo_ro     = l_odd.contract(r_odd, tenx::idx({0, 1}, {0, 1}));
        Eigen::Tensor<Scalar, 0> le_laga   = l_evn.contract(transfer_matrix_LAGA, tenx::idx({0, 1}, {0, 1})).contract(r_odd, tenx::idx({0, 1}, {0, 1}));
        Eigen::Tensor<Scalar, 0> lo_lcgb   = l_odd.contract(transfer_matrix_LCGB, tenx::idx({0, 1}, {0, 1})).contract(r_evn, tenx::idx({0, 1}, {0, 1}));
        Eigen::Tensor<Scalar, 0> thth      = theta.contract(theta.conjugate(), tenx::idx({1, 3, 0, 2}, {1, 3, 0, 2}));
        Eigen::Tensor<Scalar, 0> then_then = theta_evn_normalized.contract(theta_evn_normalized.conjugate(), tenx::idx({0, 2}, {0, 2}))
                                                 .contract(l_evn, tenx::idx({0, 2}, {0, 1}))
                                                 .contract(r_evn, tenx::idx({0, 1}, {0, 1}));
        Eigen::Tensor<Scalar, 0> thon_thon = theta_odd_normalized.contract(theta_odd_normalized.conjugate(), tenx::idx({0, 2}, {0, 2}))
                                                 .contract(l_odd, tenx::idx({0, 2}, {0, 1}))
                                                 .contract(r_odd, tenx::idx({0, 1}, {0, 1}));
        Eigen::Tensor<Scalar, 0> tren_le = transfer_matrix_evn.contract(l_evn, tenx::idx({0, 1}, {0, 1})).contract(r_evn, tenx::idx({0, 1}, {0, 1}));
        Eigen::Tensor<Scalar, 0> trod_lo = transfer_matrix_odd.contract(l_odd, tenx::idx({0, 1}, {0, 1})).contract(r_odd, tenx::idx({0, 1}, {0, 1}));
        std::string              str     = "Check\n";
        str += fmt::format(" l_odd_LCGB_r_evn                                = {:.15f}\n", fp(l_odd_LCGB_r_evn(0)));
        str += fmt::format(" l_evn_LBGA_r_odd                                = {:.15f}\n", fp(l_evn_LBGA_r_odd(0)));
        str += fmt::format(" < l_evn | r_evn >                               = {:.15f}\n", fp(le_re(0)));
        str += fmt::format(" < l_odd | r_odd >                               = {:.15f}\n", fp(lo_ro(0)));
        str += fmt::format(" < l_evn | LAGA  | r_odd >                       = {:.15f}\n", fp(le_laga(0)));
        str += fmt::format(" < l_odd | LCGB  | r_evn >                       = {:.15f}\n", fp(lo_lcgb(0)));
        str += fmt::format(" < theta     | theta >                           = {:.15f}\n", fp(thth(0)));
        str += fmt::format(" < theta_evn_normalized | theta_evn_normalized > = {:.15f}\n", fp(then_then(0)));
        str += fmt::format(" < theta_odd_normalized | theta_odd_normalized > = {:.15f}\n", fp(thon_thon(0)));
        str += fmt::format(" < theta_evn_normalized | theta_evn_normalized > = {:.15f}\n", fp(tren_le(0)));
        str += fmt::format(" < theta_odd_normalized | theta_odd_normalized > = {:.15f}\n", fp(trod_lo(0)));
        tools::log->trace(str);
    }
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_theta(const StateFinite<Scalar> &state, Scalar norm)
/*!
 * Returns a two-site MPS
     @verbatim
        1--[ LB ]--[ GA ]--[ LC ]-- [ GB ] -- [ LB ]--3
                     |                 |
                     0                 2

    where we define LB*GA*LC = M on the "A" site and GB*LB = M on the right site
     @endverbatim
 */
{
    auto        pos  = state.get_position();
    const auto &mpsA = state.get_mps_site(pos);
    const auto &mpsB = state.get_mps_site(pos + 1);
    return mpsA.get_M().contract(mpsB.get_M(), tenx::idx({2}, {1})) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_theta(const StateInfinite<Scalar> &state, Scalar norm)
/*!
 * Returns a two-site MPS
     @verbatim
        1--[ LB ]--[ GA ]--[ LA ]-- [ GB ] -- [ LB ]--3
                     |                 |
                     0                 2
     @endverbatim
 */
{
    return state.A().contract(state.B(), tenx::idx({2}, {1})) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_theta_swapped(const StateInfinite<Scalar> &state, Scalar norm)
/*!
 * Returns a two-site MPS with A and B swapped
     @verbatim
        1--[ LC ]--[ GB ]--[ LB ]-- [ GA ] -- [ LC ]--3
                     |                 |
                     0                 2
     @endverbatim
 */
{
    return state
               .LC_diag() // whatever L_A was in the previous moves
               .contract(state.B(), tenx::idx({1}, {1}))
               .contract(state.GA(), tenx::idx({2}, {1}))
               .contract(state.LC_diag(), tenx::idx({3}, {0}))
               .shuffle(tenx::array4{1, 0, 2, 3}) /
           norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::views::get_theta_evn(const StateInfinite<Scalar> &state, Scalar norm)
/*!
 * Returns a right normalized two-site MPS
     @verbatim
        1--[ LA ]--[ GA ]-- [ LC ] -- [ GB ]--3
                     |                 |
                     0                 2
     @endverbatim
 */

{
    return state.A().contract(state.GB(), tenx::idx({2}, {1})) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_theta_odd(const StateInfinite<Scalar> &state, Scalar norm)
/*!
 * Returns a two-site MPS with A and B swapped
     @verbatim
        1--[ LC ]--[ GB ]--[ LB ] -- [ GA ]--3
                     |                 |
                     0                 2
     @endverbatim
 */
{
    return state.LC_diag().contract(state.B(), tenx::idx({1}, {1})).contract(state.GA(), tenx::idx({2}, {1})).shuffle(tenx::array4{1, 0, 2, 3}) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_zero(const StateInfinite<Scalar> &state) {
    long                     dim = state.chiC();
    Eigen::Tensor<Scalar, 2> I   = tenx::asDiagonal(Eigen::Tensor<Scalar, 1>(dim));
    I.setConstant(1.0);
    std::array<Eigen::IndexPair<long>, 0> pair = {};
    return I.contract(I, pair).shuffle(tenx::array4{0, 2, 1, 3});
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_LBGA(const StateInfinite<Scalar> &state, Scalar norm) {
    return state.A().contract(state.A().conjugate(), tenx::idx({0}, {0})).shuffle(tenx::array4{0, 3, 1, 2}) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_GALC(const StateInfinite<Scalar> &state, Scalar norm) {
    return state.LC_diag()
               .contract(state.GA(), tenx::idx({2}, {0}))
               .contract(state.GA().conjugate(), tenx::idx({0}, {0}))
               .contract(state.LC_diag(), tenx::idx({3}, {0}))
               .shuffle(tenx::array4{0, 2, 1, 3}) /
           norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_GBLB(const StateInfinite<Scalar> &state, Scalar norm) {
    return state.B().contract(state.B().conjugate(), tenx::idx({0}, {0})).shuffle(tenx::array4{0, 2, 1, 3}) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_LCGB(const StateInfinite<Scalar> &state, Scalar norm) {
    return state.LC_diag()
               .contract(state.GB(), tenx::idx({1}, {1}))
               .contract(state.GB().conjugate(), tenx::idx({1}, {0}))
               .contract(state.LC_diag(), tenx::idx({2}, {1}))
               .shuffle(tenx::array4{0, 3, 1, 2}) /
           norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_theta_evn(const StateInfinite<Scalar> &state, Scalar norm) {
    return get_theta_evn(state).contract(get_theta_evn(state).conjugate(), tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array4{0, 2, 1, 3}) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_theta_odd(const StateInfinite<Scalar> &state, Scalar norm) {
    return get_theta_odd(state).contract(get_theta_odd(state).conjugate(), tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array4{0, 2, 1, 3}) / norm;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> tools::common::views<Scalar>::get_transfer_matrix_AB(const StateInfinite<Scalar> &state, int p) {
    Eigen::Tensor<Scalar, 4> temp = get_transfer_matrix_zero(state);
    Eigen::Tensor<Scalar, 4> temp2;
    for(int i = 0; i < p - 2; i++) {
        if(num::mod(i, 2) == 0) {
            temp2 = temp.contract(get_transfer_matrix_LBGA(state), tenx::idx({2, 3}, {0, 1}));

        } else {
            temp2 = temp.contract(get_transfer_matrix_LCGB(state), tenx::idx({2, 3}, {0, 1}));
        }
        temp = std::move(temp2);
    }
    return temp;
}

//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_theta(const class_mps_2site  &MPS, Scalar norm)
///*!
// * Returns a two-site MPS
//     @verbatim
//        1--[ LA ]--[ GA ]--[ LC ]-- [ GB ] -- [ LB ]--3
//                     |                 |
//                     0                 2
//     @endverbatim
// */
//{
//    return
//            MPS.A().contract(MPS.B(), tenx::idx({2},{1})) / norm;
//}
//
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_theta_swapped(const class_mps_2site  &MPS, Scalar norm)
///*!
// * Returns a two-site MPS with A and B swapped
//     @verbatim
//        1--[ LC ]--[ GB ]--[ LB ]-- [ GA ] -- [ LA ]--3
//                     |                 |
//                     0                 2
//     @endverbatim
// */
//{
//    return MPS.LC_diag() //whatever L_A was in the previous moves
//                    .contract(MPS.B(),        tenx::idx({1},{1}))
//                    .contract(MPS.GA(),       tenx::idx({2},{1}))
//                    .contract(MPS.LA_diag(),       tenx::idx({3},{0}))
//                    .shuffle(tenx::array4{1,0,2,3})
//            /norm;
//}
//
//
//
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_theta_evn(const class_mps_2site  &MPS, Scalar norm)
///*!
// * Returns a right normalized two-site MPS
//     @verbatim
//        1--[ LA ]--[ GA ]-- [ LC ] -- [ GB ]--3
//                     |                 |
//                     0                 2
//     @endverbatim
// */
//
//{
//    return  MPS.A()
//             .contract(MPS.GB(),  tenx::idx({2},{1}))
//            /norm;
//}
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_theta_odd(const class_mps_2site  &MPS, Scalar norm)
///*!
// * Returns a two-site MPS with A and B swapped
//     @verbatim
//        1--[ LC ]--[ GB ]-- [ LB ] -- [ GA ]--3
//                     |                 |
//                     0                 2
//     @endverbatim
// */
//{
//    return MPS.LC_diag()
//                    .contract(MPS.B(),           tenx::idx({1},{1}))
//                    .contract(MPS.GA(),          tenx::idx({2},{1}))
//                    .shuffle(tenx::array4{1,0,2,3})
//            /norm;
//}
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_zero(const class_mps_2site  &MPS) {
//    Eigen::Tensor<Scalar,1> I = MPS.MPS_A->get_LC();
//    I.setConstant(1.0);
//    Eigen::tenx::array<Eigen::IndexPair<long>,0> pair = {};
//
//    return asDiagonal(I).contract(asDiagonal(I), pair ).shuffle(tenx::array4{0,2,1,3});
//}
//
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_LBGA(const class_mps_2site  &MPS, Scalar norm)  {
//    return MPS.A().contract(MPS.A().conjugate() , tenx::idx({0},{0}))
//                   .shuffle(tenx::array4{0,3,1,2})
//           /norm;
//}
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_GALC(const class_mps_2site  &MPS, Scalar norm)  {
//    return MPS.LC_diag()
//                   .contract(MPS.GA(),               tenx::idx({2},{0}))
//                   .contract(MPS.GA().conjugate(),   tenx::idx({0},{0}))
//                   .contract(MPS.LC_diag(), tenx::idx({3}, {0}) )
//                   .shuffle(tenx::array4{0,2,1,3})
//           /norm;
//}
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_GBLB(const class_mps_2site  &MPS, Scalar norm)  {
//    return MPS.B().contract(MPS.B().conjugate() ,   tenx::idx({0},{0}))
//                   .shuffle(tenx::array4{0,2,1,3})
//           /norm;
//}
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_LCGB(const class_mps_2site  &MPS, Scalar norm)  {
//    return MPS.LC_diag()
//                    .contract(MPS.GB(),               tenx::idx({1},{1}))
//                    .contract(MPS.GB().conjugate(),   tenx::idx({1},{0}))
//                    .contract(MPS.LC_diag(), tenx::idx({2}, {1}) )
//                    .shuffle(tenx::array4{0,3,1,2})
//            /norm;
//}
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_theta_evn(const class_mps_2site  &MPS, Scalar norm)  {
//    using namespace tools::common::views;
//    return get_theta_evn(MPS).contract(get_theta_evn(MPS).conjugate(), tenx::idx({0,2},{0,2})).shuffle(tenx::array4{0,2,1,3}) / norm;
//}
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_theta_odd(const class_mps_2site  &MPS, Scalar norm)  {
//    return get_theta_odd(MPS).contract(get_theta_odd(MPS).conjugate(), tenx::idx({0,2},{0,2})).shuffle(tenx::array4{0,2,1,3}) / norm;
//}
//
//
// Eigen::Tensor<Scalar,4>
// tools::common::views::get_transfer_matrix_AB(const class_mps_2site  &MPS, int p) {
//    Eigen::Tensor<Scalar,4> temp = get_transfer_matrix_zero(MPS);
//    Eigen::Tensor<Scalar,4> temp2;
//    for (int i = 0; i < p-2; i++){
//        if(num::mod(i,2) == 0){
//            temp2 = temp.contract(get_transfer_matrix_LBGA(MPS), tenx::idx({2,3},{0,1}));
//
//        }else{
//            temp2 = temp.contract(get_transfer_matrix_LCGB(MPS), tenx::idx({2,3},{0,1}));
//        }
//        temp = temp2;
//
//
//    }
//    return temp;
//}
//
//
