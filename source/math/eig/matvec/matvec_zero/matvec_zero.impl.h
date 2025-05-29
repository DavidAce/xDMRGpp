#pragma once
#include "../../log.h"
#include "../matvec_zero.h"
#include "general/sfinae.h"
#include "math/eig/solver.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include <Eigen/Cholesky>
#include <primme/primme.h>
// #include <tblis/tblis.h>
// #include <tblis/util/thread.h>
// #include <tci/tci_config.h>

namespace eig {

#ifdef NDEBUG
    inline constexpr bool debug = false;
#else
    inline constexpr bool debug = true;
#endif
}


template<typename T>
template<typename S, typename EnvType>
MatVecZero<T>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<S>>> &mpss_, /*!< The MPS sites  */
                          const std::vector<std::reference_wrapper<const MpoSite<S>>> &mpos_, /*!< The Hamiltonian MPO's  */
                          const env_pair<const EnvType &>                             &envs_  /*!< The left and right environments.  */
) {
    static_assert(sfinae::is_any_v<EnvType, EnvEne<S>, EnvVar<S>>);

    if(mpss_.size() != 1) throw except::runtime_error("MatVecZero: requires a single mps site. Got {} sites", mpss_.size());
    if(mpos_.size() != 1) throw except::runtime_error("MatVecZero: requires a single mpo site. Got {} sites", mpos_.size());

    auto &mps = mpss_.front().get();
    auto &mpo = mpos_.front().get();
    if(!mps.isCenter())
        throw except::runtime_error("MatVecZero: mps at pos {} must be a center matrix. Got {}", mps.template get_position<long>(), mps.get_label());
    if constexpr(!sfinae::is_std_complex_v<T> and sfinae::is_std_complex_v<S>) {
        if(not tenx::isReal(mpo.MPO())) throw except::runtime_error("mpo is not real");
        if(not tenx::isReal(envs_.L.get_block())) throw except::runtime_error("envL is not real");
        if(not tenx::isReal(envs_.R.get_block())) throw except::runtime_error("envR is not real");
    }

    // Contract the bare mps and mpo into the environment
    envR = envs_.R.template get_block_as<T>();
    if constexpr(std::is_same_v<EnvType, EnvEne<S>>) {
        envL = envs_.L.template get_block_as<T>()
                   .contract(mps.template get_M_bare_as<T>(), tenx::idx({0}, {1}))
                   .contract(mps.template get_M_bare_as<T>().conjugate(), tenx::idx({0}, {1}))
                   .contract(mpo.template MPO_as<T>(), tenx::idx({0, 1, 3}, {0, 2, 3}));
    }
    if constexpr(std::is_same_v<EnvType, EnvVar<S>>) {
        envL = envs_.L.template get_block_as<T>()
                   .contract(mps.template get_M_bare_as<T>(), tenx::idx({0}, {1}))
                   .contract(mps.template get_M_bare_as<T>().conjugate(), tenx::idx({0}, {1}))
                   .contract(mpo.template MPO2_as<T>(), tenx::idx({0, 1, 3}, {0, 2, 3}));
    }

    long spin_dim = mpo.get_spin_dimension();
    shape_mps     = {spin_dim, envL.dimension(0), envR.dimension(0)};
    size_mps      = spin_dim * envL.dimension(0) * envR.dimension(0);
    shape_bond    = {envL.dimension(0), envR.dimension(0)};
    size_bond     = envL.dimension(0) * envR.dimension(0);
}
/* clang-format off */
template MatVecZero<fp32>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<fp32>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<fp32>>> &mpos_, const env_pair<const EnvEne<fp32> &> &envs_);
template MatVecZero<fp64>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<fp64>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<fp64>>> &mpos_, const env_pair<const EnvEne<fp64> &> &envs_);
template MatVecZero<fp128>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<fp128>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<fp128>>> &mpos_, const env_pair<const EnvEne<fp128> &> &envs_);
template MatVecZero<fp32>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<cx32>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<cx32>>> &mpos_, const env_pair<const EnvEne<cx32> &> &envs_);
template MatVecZero<fp64>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos_, const env_pair<const EnvEne<cx64> &> &envs_);
template MatVecZero<fp128>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos_, const env_pair<const EnvEne<cx128> &> &envs_);
template MatVecZero<cx32>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<cx32>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<cx32>>> &mpos_, const env_pair<const EnvEne<cx32> &> &envs_);
template MatVecZero<cx64>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos_, const env_pair<const EnvEne<cx64> &> &envs_);
template MatVecZero<cx128>::MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpss_, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos_, const env_pair<const EnvEne<cx128> &> &envs_);
/* clang-format off */

template<typename T> void MatVecZero<T>::init_timers() {
    t_factorOP = std::make_unique<tid::ur>("Time FactorOp");
    t_genMat   = std::make_unique<tid::ur>("Time genMat");
    t_multOPv  = std::make_unique<tid::ur>("Time MultOpv");
    t_multAx   = std::make_unique<tid::ur>("Time MultAx");
    t_multPc   = std::make_unique<tid::ur>("Time MultPc");
}
template<typename T>
int MatVecZero<T>::rows() const {
    return safe_cast<int>(size_bond);
}

template<typename T>
int MatVecZero<T>::cols() const {
    return safe_cast<int>(size_bond);
}

template<typename T>
void MatVecZero<T>::FactorOP() {
    throw std::runtime_error("template<typename T> void MatVecZero<T>::FactorOP(): Not implemented");
}

template<typename T>
void MatVecZero<T>::MultOPv([[maybe_unused]] T *bond_in_, [[maybe_unused]] T *bond_out_) {
    throw std::runtime_error("template<typename T> void MatVecZero<T>::MultOPv(T *bond_in_, T *bond_out_): Not implemented");
}

template<typename T>
void MatVecZero<T>::MultOPv([[maybe_unused]] void *x, [[maybe_unused]] int *ldx, [[maybe_unused]] void *y, [[maybe_unused]] int *ldy,
                            [[maybe_unused]] int *blockSize, [[maybe_unused]] primme_params *primme, [[maybe_unused]] int *err) {
    throw std::runtime_error("template<typename T> void MatVecZero<T>::MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] "
                             "primme_params *primme, int *err): Not implemented");
}

template<typename T>
void MatVecZero<T>::MultAx(T *bond_in_, T *bond_out_) {
    auto token    = t_multAx->tic_token();
    auto bond_in  = Eigen::TensorMap<Eigen::Tensor<T, 2>>(bond_in_, shape_bond);
    auto bond_out = Eigen::TensorMap<Eigen::Tensor<T, 2>>(bond_out_, shape_bond);
    // auto bond_rank2 = tenx::asDiagonal(bond_in).contract(envL, tenx::idx({0}, {0})).contract(envR, tenx::idx({0, 2}, {0, 2}));
    // bond_out        = tenx::extractDiagonal(bond_rank2);
    bond_out = bond_in.contract(envL, tenx::idx({0}, {0})).contract(envR, tenx::idx({0, 2}, {0, 2}));
    num_mv++;
}
template<typename T>
void MatVecZero<T>::perform_op(const T *bond_in_, T *bond_out_) const {
    auto token    = t_multAx->tic_token();
    auto bond_in  = Eigen::TensorMap<Eigen::Tensor<const T, 2>>(bond_in_, shape_bond);
    auto bond_out = Eigen::TensorMap<Eigen::Tensor<T, 2>>(bond_out_, shape_bond);
    // auto bond_rank2 = tenx::asDiagonal(bond_in).contract(envL, tenx::idx({0}, {0})).contract(envR, tenx::idx({0, 2}, {0, 2}));
    // bond_out        = tenx::extractDiagonal(bond_rank2);
    bond_out = bond_in.contract(envL, tenx::idx({0}, {0})).contract(envR, tenx::idx({0, 2}, {0, 2}));
    num_mv++;
}

template<typename T>
void MatVecZero<T>::MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, [[maybe_unused]] int *err) {
    // #pragma omp parallel for for schedule(static, 8)
    for(int i = 0; i < *blockSize; i++) {
        T *bond_in_ptr  = static_cast<T *>(x) + *ldx * i;
        T *bond_out_ptr = static_cast<T *>(y) + *ldy * i;
        MultAx(bond_in_ptr, bond_out_ptr);
    }
    *err = 0;
}

template<typename T>
void MatVecZero<T>::print() const {}

template<typename T>
void MatVecZero<T>::reset() {
    if(t_factorOP) t_factorOP->reset();
    if(t_multOPv) t_multOPv->reset();
    if(t_genMat) t_genMat->reset();
    if(t_multAx) t_multAx->reset();
    num_mv = 0;
    num_op = 0;
}

template<typename T>
void MatVecZero<T>::set_shift(Cplx shift) {
    // Here we set an energy shift directly on the MPO.
    // This only works if the MPO is not compressed already.
    if(readyShift) return;
    if(sigma == shift) return;
    auto shift_per_mpo = shift / static_cast<Real>(mpos.size());
    auto sigma_per_mpo = sigma / static_cast<Real>(mpos.size());
    for(size_t idx = 0; idx < mpos.size(); ++idx) {
        // The MPO is a rank4 tensor ijkl where the first 2 ij indices draw a simple
        // rank2 matrix, where each element is also a matrix with the size
        // determined by the last 2 indices kl.
        // When we shift an MPO, all we do is subtract a diagonal matrix from
        // the bottom left corner of the ij-matrix.
        auto &mpo  = mpos[idx];
        auto  dims = mpo.dimensions();
        if(dims[2] != dims[3]) throw except::logic_error("MPO has different spin dimensions up and down: {}", dims);
        auto spindim = dims[2];
        long offset1 = dims[0] - 1;

        // Setup extents and handy objects
        std::array<long, 4> offset4{offset1, 0, 0, 0};
        std::array<long, 4> extent4{1, 1, spindim, spindim};
        std::array<long, 2> extent2{spindim, spindim};
        auto                id = tenx::TensorIdentity<T>(spindim);
        // We undo the previous sigma and then subtract the new one. We are aiming for [A - I*shift]
        if constexpr(sfinae::is_std_complex_v<T>)
            mpo.slice(offset4, extent4).reshape(extent2) += id * (sigma_per_mpo - shift_per_mpo);
        else
            mpo.slice(offset4, extent4).reshape(extent2) += id * std::real(sigma_per_mpo - shift_per_mpo);
        eig::log->debug("Shifted MPO {} energy by {:.16f}", idx, fp(shift_per_mpo));
    }
    sigma = shift;
    if(not mpos_shf.empty()) {
        mpos_shf.clear();
        for(const auto &mpo : mpos) mpos_shf.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1}));
    }

    readyShift = true;
}

template<typename T>
void MatVecZero<T>::set_mode(const eig::Form form_) {
    form = form_;
}
template<typename T>
void MatVecZero<T>::set_side(const eig::Side side_) {
    side = side_;
}

template<typename T>
typename  MatVecZero<T>::Cplx MatVecZero<T>::get_shift() const {
    return sigma;
}

template<typename T>
eig::Form MatVecZero<T>::get_form() const {
    return form;
}
template<typename T>
eig::Side MatVecZero<T>::get_side() const {
    return side;
}

template<typename T>
const std::vector<Eigen::Tensor<T, 4>> &MatVecZero<T>::get_mpos() const {
    return mpos;
}
template<typename T>
const Eigen::Tensor<T, 3> &MatVecZero<T>::get_envL() const {
    return envL;
}
template<typename T>
const Eigen::Tensor<T, 3> &MatVecZero<T>::get_envR() const {
    return envR;
}

template<typename T>
long MatVecZero<T>::get_size_mps() const {
    return size_mps;
}
template<typename T>
long MatVecZero<T>::get_size_bond() const {
    return size_bond;
}

template<typename T>
std::array<long, 3> MatVecZero<T>::get_shape_mps() const {
    return shape_mps;
}
template<typename T>
std::array<long, 2> MatVecZero<T>::get_shape_bond() const {
    return shape_bond;
}

template<typename T>
std::vector<std::array<long, 4>> MatVecZero<T>::get_shape_mpo() const {
    std::vector<std::array<long, 4>> shapes;
    for(const auto &mpo : mpos) shapes.emplace_back(mpo.dimensions());
    return shapes;
}

template<typename T>
std::array<long, 3> MatVecZero<T>::get_shape_envL() const {
    return envL.dimensions();
}
template<typename T>
std::array<long, 3> MatVecZero<T>::get_shape_envR() const {
    return envR.dimensions();
}
template<typename T>
Eigen::Tensor<T, 4> MatVecZero<T>::get_tensor() const {
    return envL.contract(envR, tenx::idx({2}, {2})).shuffle(tenx::array4{0, 2, 1, 3});
}

template<typename T>
typename MatVecZero<T>::MatrixType MatVecZero<T>::get_matrix() const {
    return tenx::MatrixCast(get_tensor(), rows(), cols());
}

template<typename T>
bool MatVecZero<T>::isReadyFactorOp() const {
    return readyFactorOp;
}
template<typename T>
bool MatVecZero<T>::isReadyShift() const {
    return readyShift;
}

