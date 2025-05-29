// #define DMRG_ENABLE_TBLIS
#pragma once
#include "../../log.h"
#include "../matvec_mpos.h"
#include "config/settings.h"
#include "debug/info.h"
#include "general/sfinae.h"
#include "io/fmt_f128_t.h"
#include "math/eig/solver.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/contraction/InvMatVecCfg.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <h5pp/h5pp.h>
#include <math/stat.h>
#include <primme/primme.h>
#include <queue>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/src/IterativeSolvers/IncompleteLU.h>

// #if defined(DMRG_ENABLE_TBLIS)
//     #include <tblis/tblis.h>
//     #include <tblis/util/thread.h>
//     #include <tci/tci_config.h>
// #endif
namespace eig {

#ifdef NDEBUG
    inline constexpr bool debug_matvec_mpos = false;
#else
    inline constexpr bool debug_matvec_mpos = false;
#endif
}

template<typename Scalar>
template<typename T, typename EnvType>
MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<T>>> &mpos_, /*!< The Hamiltonian MPO's  */
                               const env_pair<const EnvType &>                             &envs_  /*!< The left and right environments.  */
) {
    static_assert(sfinae::is_any_v<EnvType, EnvEne<T>, EnvVar<T>>);
    mpos_A.reserve(mpos_.size());
    fullsystem = envs_.L.get_sites() == 0 and envs_.R.get_sites() == 0; //  mpos.size() == settings::model::model_size;

    if constexpr(std::is_same_v<EnvType, EnvEne<T>>) {
        for(const auto &mpo_ : mpos_) mpos_A.emplace_back(mpo_.get().template MPO_as<Scalar>());
    }
    if constexpr(std::is_same_v<EnvType, EnvVar<T>>) {
        for(const auto &mpo_ : mpos_) mpos_A.emplace_back(mpo_.get().template MPO2_as<Scalar>());
    }
    envL_A = envs_.L.template get_block_as<Scalar>();
    envR_A = envs_.R.template get_block_as<Scalar>();

    long spin_dim = 1;
    for(const auto &mpo : mpos_A) spin_dim *= mpo.dimension(2);
    spindims.reserve(mpos_A.size());
    for(const auto &mpo : mpos_A) spindims.emplace_back(mpo.dimension(2));

    shape_mps = {spin_dim, envL_A.dimension(0), envR_A.dimension(0)};
    size_mps  = spin_dim * envL_A.dimension(0) * envR_A.dimension(0);

    // if(mpos.size() == settings::model::model_size) {
    //     auto t_spm = tid::ur("t_spm");
    //     t_spm.tic();
    //     eig::log->info("making sparse matrix ... ", t_spm.get_last_interval());
    //     sparseMatrix = get_sparse_matrix();
    //     t_spm.toc();
    //     eig::log->info("making sparse matrix ... {:.3e} s | nnz {} / {} = {:.16f}", t_spm.get_last_interval(), sparseMatrix.nonZeros(), sparseMatrix.size(),
    //                    static_cast<double>(sparseMatrix.nonZeros()) / static_cast<double>(sparseMatrix.size()));
    // }

    // If we have 5 or fewer mpos, it is faster to just merge them once and apply them in one contraction.
    init_mpos_A();

    t_factorOP = std::make_unique<tid::ur>("Time FactorOp");
    t_genMat   = std::make_unique<tid::ur>("Time genMat");
    t_multOPv  = std::make_unique<tid::ur>("Time MultOpv");
    t_multAx   = std::make_unique<tid::ur>("Time MultAx");
    t_multPc   = std::make_unique<tid::ur>("Time MultPc");
}

template<typename Scalar>
template<typename T, typename EnvTypeA, typename EnvTypeB>
MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<T>>> &mpos_, /*!< The Hamiltonian MPO's  */
                               const env_pair<const EnvTypeA &>                            &enva_, /*!< The left and right environments.  */
                               const env_pair<const EnvTypeB &>                            &envb_)
    : MatVecMPOS(mpos_, enva_) {
    // static_assert(sfinae::is_any_v<EnvTypeA, EnvVar>);
    // static_assert(sfinae::is_any_v<EnvTypeB, EnvEne>);
    if constexpr(std::is_same_v<EnvTypeB, EnvEne<T>>) {
        for(const auto &mpo_ : mpos_) mpos_B.emplace_back(mpo_.get().template MPO_as<Scalar>());
    }
    if constexpr(std::is_same_v<EnvTypeB, EnvVar<T>>) {
        for(const auto &mpo_ : mpos_) mpos_B.emplace_back(mpo_.get().template MPO2_as<Scalar>());
    }
    envL_B = envb_.L.template get_block_as<Scalar>();
    envR_B = envb_.R.template get_block_as<Scalar>();

    init_mpos_B();
}

template<typename Scalar>
void MatVecMPOS<Scalar>::init_mpos_A() {
    if(mpos_A.size() <= 5) {
        constexpr auto contract_idx    = tenx::idx({1}, {0});
        constexpr auto shuffle_idx     = tenx::array6{0, 3, 1, 4, 2, 5};
        auto          &threads         = tenx::threads::get();
        auto           contracted_mpos = mpos_A.front();
        for(size_t idx = 0; idx + 1 < mpos_A.size(); ++idx) {
            const auto &mpoL = idx == 0 ? mpos_A[idx] : contracted_mpos;
            const auto &mpoR = mpos_A[idx + 1];
            auto new_dims    = std::array{mpoL.dimension(0), mpoR.dimension(1), mpoL.dimension(2) * mpoR.dimension(2), mpoL.dimension(3) * mpoR.dimension(3)};
            auto temp        = Eigen::Tensor<Scalar, 4>(new_dims);
            temp.device(*threads->dev) = mpoL.contract(mpoR, contract_idx).shuffle(shuffle_idx).reshape(new_dims);
            contracted_mpos            = std::move(temp);
        }
        mpos_A   = {contracted_mpos}; // Replace by a single pre-contracted mpo
        spindims = {mpos_A.front().dimension(2)};
    } else {
        // We pre-shuffle each mpo to speed up the sequential contraction
        for(const auto &mpo : mpos_A) mpos_A_shf.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1}));
    }
}

template<typename Scalar>
void MatVecMPOS<Scalar>::init_mpos_B() {
    if(mpos_B.size() <= 5) {
        constexpr auto contract_idx    = tenx::idx({1}, {0});
        constexpr auto shuffle_idx     = tenx::array6{0, 3, 1, 4, 2, 5};
        auto          &threads         = tenx::threads::get();
        auto           contracted_mpos = mpos_B.front();
        for(size_t idx = 0; idx + 1 < mpos_B.size(); ++idx) {
            const auto &mpoL = idx == 0 ? mpos_B[idx] : contracted_mpos;
            const auto &mpoR = mpos_B[idx + 1];
            auto new_dims    = std::array{mpoL.dimension(0), mpoR.dimension(1), mpoL.dimension(2) * mpoR.dimension(2), mpoL.dimension(3) * mpoR.dimension(3)};
            auto temp        = Eigen::Tensor<Scalar, 4>(new_dims);
            temp.device(*threads->dev) = mpoL.contract(mpoR, contract_idx).shuffle(shuffle_idx).reshape(new_dims);
            contracted_mpos            = std::move(temp);
        }
        mpos_B = {contracted_mpos}; // Replace by a single pre-contracted mpo
    } else {
        // We pre-shuffle each mpo to speed up the sequential contraction
        for(const auto &mpo : mpos_B) mpos_B_shf.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1}));
    }
}

template<typename Scalar>
int MatVecMPOS<Scalar>::rows() const {
    return safe_cast<int>(size_mps);
}

template<typename Scalar>
int MatVecMPOS<Scalar>::cols() const {
    return safe_cast<int>(size_mps);
}

template<typename Scalar>
std::vector<long> MatVecMPOS<Scalar>::get_offset(long x, long rank, long size) const {
    std::vector<long> indices(rank);
    for(int i = 0; i < rank; i++) {
        indices[i] = x % size;
        x /= size;
    }
    return indices;
}

template<typename Scalar>
template<size_t rank>
constexpr std::array<long, rank> MatVecMPOS<Scalar>::get_offset(long flatindex, const std::array<long, rank> &dimensions) const {
    std::array<long, rank> indices;
    for(size_t i = 0; i < rank; i++) {
        indices[i] = flatindex % (dimensions[i]);
        flatindex /= (dimensions[i]);
    }
    return indices;
}

template<typename Scalar>
std::vector<long> MatVecMPOS<Scalar>::get_offset(long flatindex, size_t rank, const std::vector<long> &dimensions) const {
    std::vector<long> indices(rank);
    for(size_t i = 0; i < rank; i++) {
        indices[i] = flatindex % (dimensions[i]);
        flatindex /= (dimensions[i]);
    }
    return indices;
}

// template<typename TI>
// TI HammingDist(TI x, TI y) {
//     TI dist = 0;
//     TI val  = x ^ y; // calculate differ bit
//     while(val)       // this dist variable calculates set bits in a loop
//     {
//         ++dist;
//         if(dist > 4) return dist;
//         val &= val - 1;
//     }
//     return dist;
// }

template<typename Scalar>
long MatVecMPOS<Scalar>::round_dn(long num, long multiple) const {
    return (num / multiple) * multiple;
}

template<typename Scalar>
long MatVecMPOS<Scalar>::round_up(long num, long multiple) const {
    if(multiple == 0) return num;
    long remainder = num % multiple;
    if(remainder == 0) return num;
    return num + multiple - remainder;
}

template<typename Scalar>
template<auto rank>
constexpr long MatVecMPOS<Scalar>::ravel_multi_index(const std::array<long, rank> &multi_index, const std::array<long, rank> &dimensions,
                                                     char order) const noexcept {
    assert(order == 'F' or order == 'C');
    if(order == 'F') {
        long index   = 0;
        long dimprod = 1;
        for(size_t i = 0; i < rank; ++i) {
            index += multi_index[i] * dimprod;
            dimprod *= dimensions[i];
        }
        return index;
    }
    if(order == 'C') {
        long index = 0;
        for(size_t i = 0; i < rank; ++i) index = index * dimensions[i] + multi_index[i];
        return index;
    }

    return -1;
}

template<typename Scalar>
template<auto rank>
constexpr std::array<long, rank> MatVecMPOS<Scalar>::get_extent(long N, const std::array<long, rank> &dimensions, const std::array<long, rank> &offsets) const {
    // Finds the largest subindex extents of a linear index, guaranteed to be i
    long offset   = ravel_multi_index(offsets, dimensions);
    long maxcount = std::reduce(dimensions.begin(), dimensions.end(), 1l, std::multiplies<long>());
    assert(N + offset <= maxcount);
    // if(N == maxcount) { return dimensions; }
    if(N + offset > maxcount) throw except::logic_error("N ({}) is outside of bounds for dimensions {}", N, dimensions);
    std::array<long, rank> extents;
    extents.fill(1);
    for(size_t i = 0; i < rank; i++) {
        long count  = std::reduce(extents.begin(), extents.end(), 1l, std::multiplies<long>());
        long newdim = std::min(static_cast<long>(std::ceil(static_cast<double>(N) / static_cast<double>(count))), dimensions[i]);
        assert(newdim >= 0);
        long newcount = count * newdim;
        if(newcount + offset <= maxcount) extents[i] = newdim;
        // extents[i] = std::min(static_cast<long>(std::ceil(static_cast<double>(N) / static_cast<double>(count))), (dimensions[i] - offsets[i]));
    }
    return extents;
}

template<typename Scalar>
template<auto rank>
constexpr std::array<long, rank> MatVecMPOS<Scalar>::get_extent(const std::array<long, rank> &I0, const std::array<long, rank> &IN,
                                                                const std::array<long, rank> &dimensions) const {
    // Finds the largest subindex extents of a linear index, guaranteed to be i
    auto extent = dimensions;
    if(I0 == IN) {
        extent.fill(1);
        return extent;
    }
    auto INN = get_offset(1l + ravel_multi_index(IN, dimensions), dimensions);
    if(INN == std::array<long, rank>{0}) INN = dimensions;
    for(size_t idx = rank - 1; idx < rank; --idx) {
        extent[idx] = std::clamp(INN[idx] - I0[idx] + 1, 1l, dimensions[idx] - I0[idx]);
        if(INN[idx] > I0[idx]) break;
    }
    return extent;
}

template<typename Scalar>
Scalar MatVecMPOS<Scalar>::get_matrix_element(long I, long J, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL,
                                              const Eigen::Tensor<Scalar, 3> &ENVR) const {
    if(I < 0 or I >= size_mps) return 0;
    if(J < 0 or J >= size_mps) return 0;

    if(MPOS.empty()) {
        return I == J ? Scalar(1.0) : Scalar(0.0); // Assume an identity matrix
    }

    auto rowindices = std::array<long, 3>{};
    auto colindices = std::array<long, 3>{};

    if(I == J) {
        rowindices = get_offset(I, shape_mps);
        colindices = rowindices;
    } else {
        rowindices = get_offset(I, shape_mps);
        colindices = get_offset(J, shape_mps);
    }
    auto ir = rowindices[0];
    auto jr = rowindices[1];
    auto kr = rowindices[2];
    auto ic = colindices[0];
    auto jc = colindices[1];
    auto kc = colindices[2];
    // Index i is special, since it is composed of all the mpo physical indices.
    // auto idxs  = get_offset(i, mpos.size(), mpos.front().dimension(2)); // Maps i to tensor indices to select the physical indices in the
    // mpos

    bool shouldBeZero = false;
    // if(fullsystem) {
    //     auto pr = std::popcount(static_cast<unsigned int>(ir));
    //     auto pc = std::popcount(static_cast<unsigned int>(ic));
    //     if(((pr ^ pc) & 1) != 0) { return 0.0; } // different parity
    //     auto hd = std::popcount(static_cast<unsigned int>(ir) ^ static_cast<unsigned int>(ic));
    //     if(hd > 4) { return 0.0; }
    //     auto pd = std::abs(pr - pc);
    //     if(pd > 4) { return 0.0; } // popcount difference is larger than 4
    //     if(pd & 1) { return 0.0; } // popcount difference is odd
    //     // if(((pr ^ pc) & 1) != 0) { shouldBeZero = true; } // different parity
    //     // if(hd > 4) { shouldBeZero = true; }
    //     // if(pd > 4) { shouldBeZero = true; } // popcount difference is larger than 4
    //     // if(pd & 1) { shouldBeZero = true; } // popcount difference is odd
    // }

    auto irxs  = get_offset(ir, MPOS.size(), spindims); // Maps ir to tensor indices to select the upper physical indices in the mpos
    auto icxs  = get_offset(ic, MPOS.size(), spindims); // Maps ic to tensor indices to select the lower physical indices in the mpos
    auto mpo_i = Eigen::Tensor<Scalar, 4>();
    auto temp  = Eigen::Tensor<Scalar, 4>();

    constexpr auto shf = tenx::array6{0, 3, 1, 4, 2, 5};
    for(size_t mdx = 0; mdx < MPOS.size(); ++mdx) {
        const auto &mpo = MPOS[mdx];
        auto        dim = mpo.dimensions();
        auto        off = std::array<long, 4>{0, 0, irxs[mdx], icxs[mdx]};
        auto        ext = std::array<long, 4>{dim[0], dim[1], 1, 1};
        if(mdx == 0) {
            mpo_i = mpo.slice(off, ext);
            if(tenx::isZero(mpo_i, std::numeric_limits<RealScalar>::epsilon())) { return Scalar{0.0}; }
            continue;
        }
        auto shp = std::array<long, 4>{mpo_i.dimension(0), dim[1], 1, 1};
        temp.resize(shp);
        temp  = mpo_i.contract(mpo.slice(off, ext), tenx::idx({1}, {0})).shuffle(shf).reshape(shp);
        mpo_i = std::move(temp);
        if(tenx::isZero(mpo_i, std::numeric_limits<RealScalar>::epsilon())) {
            // eig::log->info("({}, {}) = < {} | {} > = 0 (mdx {}, pr {}, pc {}, pd {}, hd {})", I, J, irxs, icxs, mdx, pr, pc, pd,
            // HammingDist(static_cast<size_t>(ir), static_cast<size_t>(ic)));
            return Scalar{0.0};
        }
    }

    if(fullsystem and mpo_i.size() == 1) {
        if(mpo_i.coeff(0) != Scalar{0.0} and shouldBeZero) {
            auto valmsg = std::format("{:.16f}{:+.16f}i", std::real(mpo_i.coeff(0)), std::imag(mpo_i.coeff(0)));
            eig::log->info("({}, {}) = < {} | {} > = {}", I, J, irxs, icxs, valmsg);
        }
        return mpo_i.coeff(0);
    }

    auto ext_j = std::array<long, 3>{1, 1, ENVL.dimension(2)};
    auto ext_k = std::array<long, 3>{1, 1, ENVR.dimension(2)};
    auto off_j = std::array<long, 3>{jr, jc, 0};
    auto off_k = std::array<long, 3>{kr, kc, 0};
    // auto envL_j = envL.slice(off_j, ext_j);
    // auto envR_k = envR.slice(off_k, ext_k);
    auto envL_j     = Eigen::Tensor<Scalar, 3>(ENVL.slice(off_j, ext_j));
    auto envR_k     = Eigen::Tensor<Scalar, 3>(ENVR.slice(off_k, ext_k));
    auto envL_j_map = Eigen::Map<const VectorType>(envL_j.data(), envL_j.size());
    auto envR_k_map = Eigen::Map<const VectorType>(envR_k.data(), envR_k.size());
    auto mpo_i_map  = Eigen::Map<const MatrixType>(mpo_i.data(), mpo_i.dimension(0), mpo_i.dimension(1));
    return envL_j_map.transpose() * mpo_i_map * envR_k_map;
    // eig::log->info("({:3}, {:3}) = [{:3} {:3} {:3}]  [{:3} {:3} {:3}] = {:.16f}", I, J, ir, jr, kr, ic, jc, kc, result);

    // return result;
    // Eigen::Tensor<Scalar, 6> elem(1, 1, 1, 1, 1, 1);
    // elem = envL_j.contract(mpo_i, tenx::idx({2}, {0})).contract(envR_k, tenx::idx({2}, {2})).shuffle(tenx::array6{2, 0, 4, 3, 1, 5});
    // return elem.coeff(0);
}

template<typename Scalar>
typename MatVecMPOS<Scalar>::VectorType MatVecMPOS<Scalar>::get_diagonal_new(long offset, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS,
                                                                             const Eigen::Tensor<Scalar, 3> &ENVL, const Eigen::Tensor<Scalar, 3> &ENVR) const {
    if(MPOS.empty()) return VectorType::Ones(size_mps); // Assume an identity matrix
    auto res = VectorType(size_mps);
#pragma omp parallel for
    for(long I = 0; I < size_mps; ++I) { res[I] = get_matrix_element(I, I + offset, MPOS, ENVL, ENVR); }
    return res;
}

template<typename Scalar>
typename MatVecMPOS<Scalar>::MatrixType MatVecMPOS<Scalar>::get_diagonal_block(long offset, long extent, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS,
                                                                               const Eigen::Tensor<Scalar, 3> &ENVL,
                                                                               const Eigen::Tensor<Scalar, 3> &ENVR) const {
    if(MPOS.empty()) return MatrixType::Identity(extent, extent);
    extent = std::min(extent, size_mps - offset);
    if(offset >= size_mps) { return {}; }
    auto  t_old   = tid::ur("old");
    auto  t_new   = tid::ur("new");
    auto  res     = MatrixType(extent, extent);
    auto &threads = tenx::threads::get();
    if(MPOS.size() > 1) {
        t_old.tic();
#pragma omp parallel for collapse(2)
        for(long J = 0; J < extent; J++) {
            for(long I = J; I < extent; I++) {
                if(I + offset >= size_mps) continue;
                if(J + offset >= size_mps) continue;
                // res.template selfadjointView<Eigen::Lower>()(I, J) = get_matrix_element(I + offset, J + offset); // Lower part is sufficient
                auto elem = get_matrix_element(I + offset, J + offset, MPOS, ENVL, ENVR);
                res(I, J) = elem;
                if constexpr(std::is_same_v<Scalar, cx64>)
                    res(J, I) = std::conj(elem);
                else
                    res(J, I) = elem;
            }
        }
        t_old.toc();
        return res;
    }

    {
        t_new.tic();
        res.setZero();
        long J0 = offset;              // index of the first column
        long JN = offset + extent - 1; // index of the last column
        long JY = J0;                  // JY iterates  J0 ... JN
        while(JY <= JN) {
            long IX = JY;                  // index of the first row
            long IN = offset + extent - 1; // index of the last row

            // We now define a new sub-block [R0...RN, C0...CN] which lies inside [IX...IN, JY...JN] with boundaries
            // at multiples of the dim0*dim1 of the mps indices, so that we always contract entire tiles along dim3.
            long R0 = std::clamp(round_dn(IX, shape_mps[0] * shape_mps[1]) - 0l, 0l, size_mps - 1); // index of the first row in the current sub-block
            long RN = std::clamp(round_up(IN, shape_mps[0] * shape_mps[1]) - 1l, 0l, size_mps - 1); // index of the last row in the current sub-block
            long C0 = JY;                                                                           // index of the first column
            long CN = JY; // index of the last column  (calculate one column at a time by default)
            if(R0 == C0) {
                // We are at a tensor dimension boundary, so we can actually calculate more columns when we start from here
                // There is no point in taking too many columns though, since we discard the top right triangle of the sub block (due to hermiticity)
                // long sqrt_shape2 = std::clamp(static_cast<long>(std::sqrt(shape_mps[2])), 1l, shape_mps[2]);
                // CN = std::clamp(round_up(C0 + 1, shape_mps[0] * shape_mps[1]) - 1, C0, JN);
                CN = std::clamp(round_up(C0 + static_cast<long>(std::sqrt(extent)), shape_mps[0] * shape_mps[1]) - 1, C0, JN);
                // CN = std::clamp(C0 + 1, C0, JN);
            }
            // Map the sub-block indices to ijk indices and extents of the mps tensor.
            auto R0_ijk   = get_offset(R0, shape_mps);
            auto RN_ijk   = get_offset(RN, shape_mps);
            auto C0_ijk   = get_offset(C0, shape_mps);
            auto CN_ijk   = get_offset(CN, shape_mps);
            auto R_ext    = get_extent(R0_ijk, RN_ijk, shape_mps);
            auto C_ext    = get_extent(C0_ijk, CN_ijk, shape_mps);
            auto ext_blk2 = std::array<long, 2>{R_ext[0] * R_ext[1] * R_ext[2], C_ext[0] * C_ext[1] * C_ext[2]};
            auto off_res  = std::array<long, 2>{IX - offset, JY - offset};
            auto ext_res  = std::array<long, 2>{IN - IX + 1, CN - C0 + 1};

            auto get_subblock = [&](const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_, const Eigen::Tensor<Scalar, 3> &envl_,
                                    const Eigen::Tensor<Scalar, 3> &envr_) -> MatrixType {
                if(mpos_.empty()) {
                    auto subblock = MatrixType(ext_res[0], ext_res[1]);
                    subblock.setZero();
                    // Set 1's along the diagonal of the supermatrix
                    for(long R = 0; R < subblock.rows(); ++R) {
                        for(long C = 0; C < subblock.cols(); ++C) {
                            long I = IX - R;
                            long J = JY - C;
                            if(I == J) subblock(R, C) = 1;
                        }
                    }

                    return subblock;
                }
                std::array<long, 3> off_envl = {R0_ijk[1], C0_ijk[1], 0};
                std::array<long, 3> ext_envl = {R_ext[1], C_ext[1], mpos_.front().dimension(0)};
                std::array<long, 3> off_envr = {R0_ijk[2], C0_ijk[2], 0};
                std::array<long, 3> ext_envr = {R_ext[2], C_ext[2], mpos_.front().dimension(1)};
                std::array<long, 4> off_mpos = {off_envl[2], off_envr[2], R0_ijk[0], C0_ijk[0]};
                std::array<long, 4> ext_mpos = {ext_envl[2], ext_envr[2], R_ext[0], C_ext[0]};
                auto                block    = Eigen::Tensor<Scalar, 2>(ext_blk2);
                if(envl_.dimension(0) <= envr_.dimension(0)) {
                    block.device(*threads->dev) = envl_.slice(off_envl, ext_envl)
                                                      .contract(mpos_.front().slice(off_mpos, ext_mpos), tenx::idx({2}, {0}))
                                                      .contract(envr_.slice(off_envr, ext_envr), tenx::idx({2}, {2}))
                                                      .shuffle(tenx::array6{2, 0, 4, 3, 1, 5})
                                                      .reshape(ext_blk2);
                } else {
                    block.device(*threads->dev) = envr_.slice(off_envr, ext_envr)
                                                      .contract(mpos_.front().slice(off_mpos, ext_mpos), tenx::idx({2}, {1}))
                                                      .contract(envl_.slice(off_envl, ext_envl), tenx::idx({2}, {2}))
                                                      .shuffle(tenx::array6{2, 4, 0, 3, 5, 1})
                                                      .reshape(ext_blk2);
                }
                return Eigen::Map<MatrixType>(block.data(), ext_blk2[0], ext_blk2[1]).block(IX - R0, JY - C0, ext_res[0], ext_res[1]);
            };

            res.block(off_res[0], off_res[1], ext_res[0], ext_res[1]) = get_subblock(MPOS, ENVL, ENVR);
            MatrixType blkres                                         = res.block(off_res[0], off_res[1], ext_res[0], ext_res[1]);
            // MatrixType blkdbg = dbg.block(off_res[0], off_res[1], ext_res[0], ext_res[1]);
            // if(!blkres.isApprox(blkdbg)) {
            if constexpr(eig::debug_matvec_mpos)
                eig::log->trace("IX {:5} JY {:5} J0 {:5} JN {:5} R0 {:5} RN {:5} C0 {:5} CN {:5} R0_ijk {} RN_ijk {} C0_ijk {}, CN_ijk {} R_ext {} C_ext {} "
                                "ext_blk2 {} off_res {} ext_res {}",
                                IX, JY, J0, JN, R0, RN, C0, CN, R0_ijk, RN_ijk, C0_ijk, CN_ijk, R_ext, C_ext, ext_blk2, off_res, ext_res);
            JY += (CN - C0 + 1);
        }

        // t_new.toc();
        return res.template selfadjointView<Eigen::Lower>();
    }
}

template<typename Scalar>
typename MatVecMPOS<Scalar>::MatrixType
    MatVecMPOS<Scalar>::get_diagonal_block(long offset, long extent, Scalar shift, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS_A,
                                           const Eigen::Tensor<Scalar, 3> &ENVL_A, const Eigen::Tensor<Scalar, 3> &ENVR_A,
                                           const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS_B, const Eigen::Tensor<Scalar, 3> &ENVL_B,
                                           const Eigen::Tensor<Scalar, 3> &ENVR_B) const {
    if(MPOS_A.empty()) return MatrixType::Identity(extent, extent);
    extent = std::min(extent, size_mps - offset);
    if(offset >= size_mps) { return {}; }
    auto  t_old   = tid::ur("old");
    auto  t_new   = tid::ur("new");
    auto  res     = MatrixType(extent, extent);
    auto &threads = tenx::threads::get();
    if(MPOS_A.size() > 1) {
        t_old.tic();
#pragma omp parallel for collapse(2)
        for(long J = 0; J < extent; J++) {
            for(long I = J; I < extent; I++) {
                if(I + offset >= size_mps) continue;
                if(J + offset >= size_mps) continue;
                // res.template selfadjointView<Eigen::Lower>()(I, J) = get_matrix_element(I + offset, J + offset); // Lower part is sufficient
                Scalar elemA = get_matrix_element(I + offset, J + offset, MPOS_A, ENVL_A, ENVR_A);
                Scalar elemB = shift != Scalar{0.0} ? get_matrix_element(I + offset, J + offset, MPOS_B, ENVL_B, ENVR_B) : Scalar{0.0};
                Scalar elem  = elemA - shift * elemB;
                res(I, J)    = elem;
                if constexpr(std::is_same_v<Scalar, cx64>)
                    res(J, I) = std::conj(elem);
                else
                    res(J, I) = elem;
            }
        }
        t_old.toc();
        return res;
    }

    {
        t_new.tic();
        res.setZero();
        long J0 = offset;              // index of the first column
        long JN = offset + extent - 1; // index of the last column
        long JY = J0;                  // JY iterates  J0 ... JN
        while(JY <= JN) {
            long IX = JY;                  // index of the first row
            long IN = offset + extent - 1; // index of the last row

            // We now define a new sub-block [R0...RN, C0...CN] which lies inside [IX...IN, JY...JN] with boundaries
            // at multiples of the dim0*dim1 of the mps indices, so that we always contract entire tiles along dim3.
            long R0 = std::clamp(round_dn(IX, shape_mps[0] * shape_mps[1]) - 0l, 0l, size_mps - 1); // index of the first row in the current sub-block
            long RN = std::clamp(round_up(IN, shape_mps[0] * shape_mps[1]) - 1l, 0l, size_mps - 1); // index of the last row in the current sub-block
            long C0 = JY;                                                                           // index of the first column
            long CN = JY; // index of the last column  (calculate one column at a time by default)
            if(R0 == C0) {
                // We are at a tensor dimension boundary, so we can actually calculate more columns when we start from here
                // There is no point in taking too many columns though, since we discard the top right triangle of the sub block (due to hermiticity)
                // long sqrt_shape2 = std::clamp(static_cast<long>(std::sqrt(shape_mps[2])), 1l, shape_mps[2]);
                // CN = std::clamp(round_up(C0 + 1, shape_mps[0] * shape_mps[1]) - 1, C0, JN);
                CN = std::clamp(round_up(C0 + static_cast<long>(std::sqrt(extent)), shape_mps[0] * shape_mps[1]) - 1, C0, JN);
                // CN = std::clamp(C0 + 1, C0, JN);
            }
            // Map the sub-block indices to ijk indices and extents of the mps tensor.
            auto R0_ijk   = get_offset(R0, shape_mps);
            auto RN_ijk   = get_offset(RN, shape_mps);
            auto C0_ijk   = get_offset(C0, shape_mps);
            auto CN_ijk   = get_offset(CN, shape_mps);
            auto R_ext    = get_extent(R0_ijk, RN_ijk, shape_mps);
            auto C_ext    = get_extent(C0_ijk, CN_ijk, shape_mps);
            auto ext_blk2 = std::array<long, 2>{R_ext[0] * R_ext[1] * R_ext[2], C_ext[0] * C_ext[1] * C_ext[2]};
            auto off_res  = std::array<long, 2>{IX - offset, JY - offset};
            auto ext_res  = std::array<long, 2>{IN - IX + 1, CN - C0 + 1};

            auto get_subblock = [&](const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_, const Eigen::Tensor<Scalar, 3> &envl_,
                                    const Eigen::Tensor<Scalar, 3> &envr_) -> MatrixType {
                if(mpos_.empty()) {
                    auto subblock = MatrixType(ext_res[0], ext_res[1]);
                    subblock.setZero();
                    // Set 1's along the diagonal of the supermatrix
                    for(long R = 0; R < subblock.rows(); ++R) {
                        for(long C = 0; C < subblock.cols(); ++C) {
                            long I = IX - R;
                            long J = JY - C;
                            if(I == J) subblock(R, C) = Scalar{1.0};
                        }
                    }

                    return subblock;
                }
                std::array<long, 3> off_envl = {R0_ijk[1], C0_ijk[1], 0};
                std::array<long, 3> ext_envl = {R_ext[1], C_ext[1], mpos_.front().dimension(0)};
                std::array<long, 3> off_envr = {R0_ijk[2], C0_ijk[2], 0};
                std::array<long, 3> ext_envr = {R_ext[2], C_ext[2], mpos_.front().dimension(1)};
                std::array<long, 4> off_mpos = {off_envl[2], off_envr[2], R0_ijk[0], C0_ijk[0]};
                std::array<long, 4> ext_mpos = {ext_envl[2], ext_envr[2], R_ext[0], C_ext[0]};
                auto                block    = Eigen::Tensor<Scalar, 2>(ext_blk2);
                if(envl_.dimension(0) <= envr_.dimension(0)) {
                    block.device(*threads->dev) = envl_.slice(off_envl, ext_envl)
                                                      .contract(mpos_.front().slice(off_mpos, ext_mpos), tenx::idx({2}, {0}))
                                                      .contract(envr_.slice(off_envr, ext_envr), tenx::idx({2}, {2}))
                                                      .shuffle(tenx::array6{2, 0, 4, 3, 1, 5})
                                                      .reshape(ext_blk2);
                } else {
                    block.device(*threads->dev) = envr_.slice(off_envr, ext_envr)
                                                      .contract(mpos_.front().slice(off_mpos, ext_mpos), tenx::idx({2}, {1}))
                                                      .contract(envl_.slice(off_envl, ext_envl), tenx::idx({2}, {2}))
                                                      .shuffle(tenx::array6{2, 4, 0, 3, 5, 1})
                                                      .reshape(ext_blk2);
                }
                return Eigen::Map<MatrixType>(block.data(), ext_blk2[0], ext_blk2[1]).block(IX - R0, JY - C0, ext_res[0], ext_res[1]);
            };

            res.block(off_res[0], off_res[1], ext_res[0], ext_res[1]) = get_subblock(MPOS_A, ENVL_A, ENVR_A);
            if(shift != Scalar{0.0}) res.block(off_res[0], off_res[1], ext_res[0], ext_res[1]) -= get_subblock(MPOS_B, ENVL_B, ENVR_B) * shift;
            MatrixType blkres = res.block(off_res[0], off_res[1], ext_res[0], ext_res[1]);
            // MatrixType blkdbg = dbg.block(off_res[0], off_res[1], ext_res[0], ext_res[1]);
            // if(!blkres.isApprox(blkdbg)) {
            if constexpr(eig::debug_matvec_mpos)
                eig::log->trace("IX {:5} JY {:5} J0 {:5} JN {:5} R0 {:5} RN {:5} C0 {:5} CN {:5} R0_ijk {} RN_ijk {} C0_ijk {}, CN_ijk {} R_ext {} C_ext {} "
                                "ext_blk2 {} off_res {} ext_res {}",
                                IX, JY, J0, JN, R0, RN, C0, CN, R0_ijk, RN_ijk, C0_ijk, CN_ijk, R_ext, C_ext, ext_blk2, off_res, ext_res);
            JY += (CN - C0 + 1);
        }

        // t_new.toc();
        return res.template selfadjointView<Eigen::Lower>();
    }
}

// template<typename Scalar>
// typename MatVecMPOS<Scalar>::MatrixType MatVecMPOS<Scalar>::get_diagonal_block_old(long offset, long extent, Scalar shift, const
// std::vector<Eigen::Tensor<Scalar, 4>> &MPOS_A,
//                                                                          const Eigen::Tensor<Scalar, 3> &ENVL_A, const Eigen::Tensor<Scalar, 3> &ENVR_A,
//                                                                          const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS_B, const Eigen::Tensor<Scalar, 3>
//                                                                          &ENVL_B, const Eigen::Tensor<Scalar, 3> &ENVR_B) const {
//     if(MPOS_A.empty()) return MatrixType::Identity(extent, extent);
//     extent = std::min(extent, size_mps - offset);
//     if(offset >= size_mps) { return {}; }
//     auto t_old = tid::ur("old");
//     auto t_new = tid::ur("new");
//     auto res   = MatrixType(extent, extent);
//
//     if(MPOS_A.size() > 1) {
//         t_old.tic();
// #pragma omp parallel for collapse(2)
//         for(long J = 0; J < extent; J++) {
//             for(long I = J; I < extent; I++) {
//                 if(I + offset >= size_mps) continue;
//                 if(J + offset >= size_mps) continue;
//                 // res.template selfadjointView<Eigen::Lower>()(I, J) = get_matrix_element(I + offset, J + offset); // Lower part is sufficient
//                 auto elemA = get_matrix_element(I + offset, J + offset, MPOS_A, ENVL_A, ENVR_A);
//                 auto elemB = get_matrix_element(I + offset, J + offset, MPOS_B, ENVL_B, ENVR_B);
//                 auto elem  = elemA - shift * elemB;
//                 res(I, J)  = elem;
//                 if constexpr(std::is_same_v<Scalar, cx64>)
//                     res(J, I) = std::conj(elem);
//                 else
//                     res(J, I) = elem;
//             }
//         }
//         t_old.toc();
//         return res;
//     }
//
//     {
//         // auto dbg = MatrixType(extent, extent);
//         // #pragma omp parallel for collapse(2)
//         // for(long J = 0; J < extent; J++) {
//         //     for(long I = J; I < extent; I++) {
//         //         if(I + offset >= size_mps) continue;
//         //         if(J + offset >= size_mps) continue;
//         //         // res.template selfadjointView<Eigen::Lower>()(I, J) = get_matrix_element(I + offset, J + offset); // Lower part is sufficient
//         //         auto elemA = get_matrix_element(I + offset, J + offset, MPOS_A, ENVL_A, ENVR_A);
//         //         auto elemB = get_matrix_element(I + offset, J + offset, MPOS_B, ENVL_B, ENVR_B);
//         //         auto elem  = elemA - shift * elemB;
//         //         dbg(I, J)  = elem;
//         //         if constexpr(std::is_same_v<Scalar, cx64>)
//         //             dbg(J, I) = std::conj(elem);
//         //         else
//         //             dbg(J, I) = elem;
//         //     }
//         // }
//
//         // t_new.tic();
//         res.setZero();
//         long J0 = offset;
//         long JN = offset + extent - 1;
//         long JY = J0;
//         while(JY <= JN) {
//             long I0 = JY;
//             long IN = offset + extent - 1;
//             long R0 = std::clamp(round_dn(I0, shape_mps[0] * shape_mps[1]), 0l, size_mps - 1);
//             long RN = std::clamp(round_up(IN, shape_mps[0] * shape_mps[1]), 0l, size_mps - 1);
//             long C0 = JY;
//             long CN = JY;
//             if(R0 == C0) {
//                 // We are at a tensor dimension boundary, so we can actually calculate more columns when we start from here
//                 // There is no point in taking too many columns though, since we discard the top right triangle of the sub block
//                 CN = std::clamp(round_up(C0 + 1, shape_mps[0] * shape_mps[1]) - 1, C0, JN);
//             }
//
//             auto R0_ijk   = get_offset(R0, shape_mps);
//             auto RN_ijk   = get_offset(RN, shape_mps);
//             auto C0_ijk   = get_offset(C0, shape_mps);
//             auto CN_ijk   = get_offset(CN, shape_mps);
//             auto R_ext    = get_extent(R0_ijk, RN_ijk, shape_mps);
//             auto C_ext    = get_extent(C0_ijk, CN_ijk, shape_mps);
//             auto ext_blk2 = std::array<long, 2>{R_ext[0] * R_ext[1] * R_ext[2], C_ext[0] * C_ext[1] * C_ext[2]};
//             auto off_res  = std::array<long, 2>{I0 - offset, JY - offset};
//             auto ext_res  = std::array<long, 2>{IN - I0 + 1, CN - C0 + 1};
//
//             auto get_tile2 = [&](const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL, const Eigen::Tensor<Scalar, 3>
//             &ENVR) -> MatrixType {
//                 if(MPOS.empty()) {
//                     auto block = MatrixType(ext_res[0], ext_res[1]);
//                     block.setZero();
//                     // Set 1's along the diagonal of the supermatrix
//                     for(long R = 0; R < block.rows(); ++R) {
//                         for(long C = 0; C < block.cols(); ++C) {
//                             long I = I0 - R;
//                             long J = JY - C;
//                             if(I == J) block(R, C) = 1;
//                         }
//                     }
//
//                     return block;
//                 }
//                 std::array<long, 3> off_envl = {R0_ijk[1], C0_ijk[1], 0};
//                 std::array<long, 3> ext_envl = {R_ext[1], C_ext[1], MPOS.front().dimension(0)};
//                 std::array<long, 3> off_envr = {R0_ijk[2], C0_ijk[2], 0};
//                 std::array<long, 3> ext_envr = {R_ext[2], C_ext[2], MPOS.front().dimension(1)};
//                 std::array<long, 4> off_mpos = {off_envl[2], off_envr[2], R0_ijk[0], C0_ijk[0]};
//                 std::array<long, 4> ext_mpos = {ext_envl[2], ext_envr[2], R_ext[0], C_ext[0]};
//
//                 auto  block   = Eigen::Tensor<Scalar, 2>(ext_blk2);
//                 auto &threads = tenx::threads::get();
//
//                 block.device(*threads->dev) = ENVL.slice(off_envl, ext_envl)
//                                                   .contract(MPOS.front().slice(off_mpos, ext_mpos), tenx::idx({2}, {0}))
//                                                   .contract(ENVR.slice(off_envr, ext_envr), tenx::idx({2}, {2}))
//                                                   .shuffle(tenx::array6{2, 0, 4, 3, 1, 5})
//                                                   .reshape(ext_blk2);
//                 return Eigen::Map<MatrixType>(block.data(), ext_blk2[0], ext_blk2[1]).block(I0 - R0, JY - C0, ext_res[0], ext_res[1]);
//             };
//
//             res.block(off_res[0], off_res[1], ext_res[0], ext_res[1]) = get_tile2(MPOS_A, ENVL_A, ENVR_A) - get_tile2(MPOS_B, ENVL_B, ENVR_B) * shift;
//
//             // MatrixType blkres = res.block(off_res[0], off_res[1], ext_res[0], ext_res[1]);
//             // MatrixType blkdbg = dbg.block(off_res[0], off_res[1], ext_res[0], ext_res[1]);
//             // if(!blkres.isApprox(blkdbg)) {
//             //     eig::log->info("I0       {}", I0);
//             //     eig::log->info("J0       {}", J0);
//             //     eig::log->info("IN       {}", IN);
//             //     eig::log->info("JN       {}", JN);
//             //     eig::log->info("R0       {}", R0);
//             //     eig::log->info("RN       {}", RN);
//             //     eig::log->info("C0       {}", C0);
//             //     eig::log->info("CN       {}", CN);
//             //     eig::log->info("R0_ijk   {}", R0_ijk);
//             //     eig::log->info("RN_ijk   {}", RN_ijk);
//             //     eig::log->info("C0_ijk   {}", C0_ijk);
//             //     eig::log->info("CN_ijk   {}", CN_ijk);
//             //     eig::log->info("R_ext    {}", R_ext);
//             //     eig::log->info("C_ext    {}", C_ext);
//             //     eig::log->info("ext_blk2 {}", ext_blk2);
//             //     // eig::log->info("ext_blk6 {}", ext_blk6);
//             //     // eig::log->info("off_envl {}", off_envl);
//             //     // eig::log->info("ext_envl {}", ext_envl);
//             //     // eig::log->info("off_envr {}", off_envr);
//             //     // eig::log->info("ext_envr {}", ext_envr);
//             //     // eig::log->info("off_mpos {}", off_mpos);
//             //     // eig::log->info("ext_mpos {}", ext_mpos);
//             //     // eig::log->info("off_resC {}", off_resC);
//             //     // eig::log->info("ext_resC {}", ext_resC);
//             //     for(long r = 0; r < blkres.rows(); ++r) {
//             //         VectorType vecC = blkres.row(r);
//             //         VectorType vecF = blkdbg.row(r);
//             //         eig::log->info("({},{}:{}):  {} | {} ", r, JY, JY + CN - C0 + 1, vecC, vecF);
//             //     }
//             //     throw except::logic_error("matC and matF mismatch");
//             // }
//             JY += (CN - C0 + 1);
//         }
//
//         // t_new.toc();
//         return res.template selfadjointView<Eigen::Lower>();
//     }
// }

template<typename Scalar>
typename MatVecMPOS<Scalar>::VectorType MatVecMPOS<Scalar>::get_row(long row_idx, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS,
                                                                    const Eigen::Tensor<Scalar, 3> &ENVL, const Eigen::Tensor<Scalar, 3> &ENVR) const {
    auto res = VectorType(size_mps);
#pragma omp parallel for
    for(long J = 0; J < size_mps; ++J) { res[J] = get_matrix_element(row_idx, J, MPOS, ENVL, ENVR); }
    return res;
}
template<typename Scalar>
typename MatVecMPOS<Scalar>::VectorType MatVecMPOS<Scalar>::get_col(long col_idx, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS,
                                                                    const Eigen::Tensor<Scalar, 3> &ENVL, const Eigen::Tensor<Scalar, 3> &ENVR) const {
    auto res = VectorType(size_mps);
#pragma omp parallel for
    for(long I = 0; I < size_mps; ++I) { res[I] = get_matrix_element(I, col_idx, MPOS, ENVL, ENVR); }
    return res;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::FactorOP() {
    auto t_token = t_factorOP->tic_token();
    if(readyFactorOp) { return; }
    if(factorization == eig::Factorization::NONE) {
        readyFactorOp = true;
        return;
    }
    MatrixType A_matrix = get_matrix();
    if(not readyShift and std::abs(get_shift()) != Scalar{0.0}) { A_matrix.diagonal() -= VectorType::Constant(rows(), get_shift()); }

    if(factorization == eig::Factorization::LDLT) {
        eig::log->debug("LDLT Factorization");
        ldlt.compute(A_matrix);
    } else if(factorization == eig::Factorization::LLT) {
        eig::log->debug("LLT Factorization");
        llt.compute(A_matrix);
    } else if(factorization == eig::Factorization::LU) {
        eig::log->debug("LU Factorization");
        lu.compute(A_matrix);
    } else if(factorization == eig::Factorization::NONE) {
        /* We don't actually invert a matrix here: we let an iterative matrix-free solver apply OP^-1 x */
        if(not readyShift) throw std::runtime_error("Cannot FactorOP with Factorization::NONE: Shift value sigma has not been set on the MPO.");
    }
    eig::log->debug("Finished factorization");
    readyFactorOp = true;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultOPv([[maybe_unused]] Scalar *mps_in_, [[maybe_unused]] Scalar *mps_out_) {
    throw except::runtime_error("MatVecMPOS<{}>::MultOPv(...) not implemented", sfinae::type_name<Scalar>());
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, int *err) {
    auto token = t_multOPv->tic_token();
    switch(side) {
        case eig::Side::R: {
            for(int i = 0; i < *blockSize; i++) {
                Scalar *x_ptr = static_cast<Scalar *>(x) + *ldx * i;
                Scalar *y_ptr = static_cast<Scalar *>(y) + *ldy * i;
                if(factorization == eig::Factorization::LDLT) {
                    Eigen::Map<VectorType> x_map(x_ptr, *ldx);
                    Eigen::Map<VectorType> y_map(y_ptr, *ldy);
                    y_map.noalias() = ldlt.solve(x_map);
                } else if(factorization == eig::Factorization::LLT) {
                    Eigen::Map<VectorType> x_map(x_ptr, *ldx);
                    Eigen::Map<VectorType> y_map(y_ptr, *ldy);
                    y_map.noalias() = llt.solve(x_map);
                } else if(factorization == eig::Factorization::LU) {
                    Eigen::Map<VectorType> x_map(x_ptr, *ldx);
                    Eigen::Map<VectorType> y_map(y_ptr, *ldy);
                    y_map.noalias() = lu.solve(x_map);
                } else {
                    throw except::runtime_error("Invalid factorization: {}", eig::FactorizationToString(factorization));
                }
                num_op++;
            }
            break;
        }
        case eig::Side::L: {
            throw std::runtime_error("Left sided matrix-free MultOPv has not been implemented");
            break;
        }
        case eig::Side::LR: {
            throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
        }
    }
    *err = 0;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultAx(const Scalar *mps_in_, Scalar *mps_out_) const {
    auto token   = t_multAx->tic_token();
    auto mps_in  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_in_, shape_mps);
    auto mps_out = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(mps_out_, shape_mps);
    if(mpos_A.size() == 1) {
        tools::common::contraction::matrix_vector_product(mps_out, mps_in, mpos_A.front(), envL_A, envR_A);
    } else {
        tools::common::contraction::matrix_vector_product(mps_out, mps_in, mpos_A_shf, envL_A, envR_A);
    }
    num_mv++;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultAx(Scalar *mps_in_, Scalar *mps_out_) {
    MultAx(static_cast<const Scalar *>(mps_in_), mps_out_);
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, [[maybe_unused]] int *err) const {
    for(int i = 0; i < *blockSize; i++) {
        Scalar *mps_in_ptr  = static_cast<Scalar *>(x) + *ldx * i;
        Scalar *mps_out_ptr = static_cast<Scalar *>(y) + *ldy * i;
        MultAx(static_cast<const Scalar *>(mps_in_ptr), mps_out_ptr);
    }
    *err = 0;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::perform_op(const Scalar *mps_in_, Scalar *mps_out_) const {
    MultAx(mps_in_, mps_out_);
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultBx(const Scalar *mps_in_, Scalar *mps_out_) const {
    if(mpos_B.empty() and mpos_B_shf.empty()) return;
    auto token   = t_multAx->tic_token();
    auto mps_in  = Eigen::TensorMap<Eigen::Tensor<const Scalar, 3>>(mps_in_, shape_mps);
    auto mps_out = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(mps_out_, shape_mps);
    if(mpos_B.size() == 1) {
        tools::common::contraction::matrix_vector_product(mps_out, mps_in, mpos_B.front(), envL_B, envR_B);
    } else if(!mpos_B_shf.empty()) {
        tools::common::contraction::matrix_vector_product(mps_out, mps_in, mpos_B_shf, envL_B, envR_B);
    }
    num_mv++;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultBx(Scalar *mps_in_, Scalar *mps_out_) const {
    MultBx(static_cast<const Scalar *>(mps_in_), mps_out_);
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultInvBx(const Scalar *mps_in_, Scalar *mps_out_, long maxiters, RealScalar tolerance) const {
    if(mpos_B.empty() and mpos_B_shf.empty()) return;
    auto token   = t_multAx->tic_token();
    auto mps_in  = Eigen::TensorMap<Eigen::Tensor<const Scalar, 3>>(mps_in_, shape_mps);
    auto mps_out = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(mps_out_, shape_mps);
    if(invJcbDiagB.size() == 0) invJcbDiagB = get_diagonal_new(0, mpos_B, envL_B, envR_B).array().cwiseInverse();
    auto invCfg = InvMatVecCfg<Scalar>{.maxiters = maxiters, .tolerance = tolerance, .invdiag = invJcbDiagB.data()};
    if(mpos_B.size() == 1) {
        tools::common::contraction::matrix_inverse_vector_product(mps_out, mps_in, mpos_B.front(), envL_B, envR_B, invCfg);
    } else if(!mpos_B_shf.empty()) {
        throw except::runtime_error("MatVecMPOS<{}>::MultInvBx(...) not implemented for multiple mpos", sfinae::type_name<Scalar>());
    }
    num_mv++;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultBx(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, [[maybe_unused]] int *err) const {
    for(int i = 0; i < *blockSize; i++) {
        Scalar *mps_in_ptr  = static_cast<Scalar *>(x) + *ldx * i;
        Scalar *mps_out_ptr = static_cast<Scalar *>(y) + *ldy * i;
        MultBx(static_cast<const Scalar *>(mps_in_ptr), mps_out_ptr);
    }
    *err = 0;
}

// template<typename Scalar>
// std::vector<std::pair<Scalar, long>> MatVecMPOS<Scalar>::get_k_smallest(const VectorType &vec, size_t k) const {
//     using idx_pair = std::pair<Scalar, long>;
//     std::priority_queue<Scalar> pq;
//     for(auto d : vec) {
//         if(pq.size() >= k && pq.top() > d) {
//             pq.push(d);
//             pq.pop();
//         } else if(pq.size() < k) {
//             pq.push(d);
//         }
//     }
//     Scalar                kth_element = pq.top();
//     std::vector<idx_pair> result;
//     for(long i = 0; i < vec.size(); i++)
//         if(vec[i] <= kth_element) { result.emplace_back(idx_pair{vec[i], i}); }
//     return result;
// }

// template<typename Scalar>
// std::vector<long> MatVecMPOS<Scalar>::get_k_smallest(const VectorType &vec, size_t k) const {
//     std::priority_queue<Scalar> pq;
//     for(auto d : vec) {
//         if(pq.size() >= k && pq.top() > d) {
//             pq.push(d);
//             pq.pop();
//         } else if(pq.size() < k) {
//             pq.push(d);
//         }
//     }
//     Scalar            kth_element = pq.top();
//     std::vector<long> result;
//     for(long i = 0; i < vec.size(); i++)
//         if(vec[i] <= kth_element) { result.emplace_back(i); }
//     return result;
// }
//
// template<typename Scalar>
// std::vector<long> MatVecMPOS<Scalar>::get_k_largest(const VectorType &vec, size_t k) const {
//     using idx_pair = std::pair<Scalar, long>;
//     std::priority_queue<idx_pair, std::vector<idx_pair>, std::greater<idx_pair>> q;
//     for(long i = 0; i < vec.size(); ++i) {
//         if(q.size() < k)
//             q.emplace(vec[i], i);
//         else if(q.top().first < vec[i]) {
//             q.pop();
//             q.emplace(vec[i], i);
//         }
//     }
//     k = q.size();
//     std::vector<long> res(k);
//     for(size_t i = 0; i < k; ++i) {
//         res[k - i - 1] = q.top().second;
//         q.pop();
//     }
//     return res;
// }

template<typename Scalar>
Eigen::Tensor<Scalar, 3> MatVecMPOS<Scalar>::operator*(const Eigen::Tensor<Scalar, 3> &x) const {
    assert(x.size() == get_size());
    Eigen::Tensor<Scalar, 3> y(x.dimensions());
    MultAx(x.data(), y.data());
    return y;
}
template<typename Scalar>
Eigen::Tensor<Scalar, 1> MatVecMPOS<Scalar>::operator*(const Eigen::Tensor<Scalar, 1> &x) const {
    assert(x.size() == get_size());
    Eigen::Tensor<Scalar, 1> y(x.size());
    MultAx(x.data(), y.data());
    return y;
}
template<typename Scalar>
typename MatVecMPOS<Scalar>::VectorType MatVecMPOS<Scalar>::operator*(const VectorType &x) const {
    assert(x.size() == get_size());
    VectorType y(x.size());
    MultAx(x.data(), y.data());
    return y;
}
template<typename Scalar>
typename MatVecMPOS<Scalar>::MatrixType MatVecMPOS<Scalar>::MultAX(const Eigen::Ref<const MatrixType> &X) const {
    assert(X.rows() == get_size());
    MatrixType Y(X.rows(), X.cols());
    for(Eigen::Index i = 0; i < X.cols(); ++i) { MultAx(X.col(i).data(), Y.col(i).data()); }
    return Y;
}
template<typename Scalar>
typename MatVecMPOS<Scalar>::VectorType MatVecMPOS<Scalar>::MultAx(const Eigen::Ref<const VectorType> &x) const {
    assert(x.rows() == get_size());
    VectorType y(x.rows());
    MultAx(x.data(), y.data());
    return y;
}

template<typename Derived>
Eigen::Matrix<double, Eigen::Dynamic, 1> cond(const Eigen::MatrixBase<Derived> &m) {
    auto solver = Eigen::BDCSVD(m.eval());
    auto rank   = solver.nonzeroSingularValues();
    return solver.singularValues().head(rank);
}

template<typename Scalar>
void MatVecMPOS<Scalar>::CalcPc(Scalar shift) {
    // if(readyCalcPc) return;
    if(lockCalcPc) return;
    auto fname = fmt::format("MatVecMPOS<Scalar>::CalcPc()", sfinae::type_name<Scalar>());
    if(preconditioner == eig::Preconditioner::NONE) {
        eig::log->info("{}: no preconditioner chosen", fname);
        return;
    }
    if(jcbShift.has_value()) {
        RealScalar relchange = std::abs(shift - jcbShift.value()) / (std::abs(shift + jcbShift.value()) / 2);
        if(relchange < static_cast<RealScalar>(1e-1)) return; // Keep the preconditioner if the shifts haven't changed by more than 1%
        eig::log->info("{}: Recomputing the preconditioner with shift {:.16f}", fname, fp(jcbShift.value()));
    }
    jcbShift = shift;

    long jcbBlockSize = jcbMaxBlockSize;
    if(jcbBlockSize == 1) {
        if(jcbDiagA.size() + jcbDiagB.size() > 0) return; // We only need to do this once
        eig::log->trace("{}: calculating the jacobi preconditioner ... (shift = {:.16f})", fname, fp(shift));
        jcbDiagA   = get_diagonal_new(0, mpos_A, envL_A, envR_A);
        jcbDiagB   = get_diagonal_new(0, mpos_B, envL_B, envR_B);
        lockCalcPc = true;
        eig::log->debug("{}: calculating the jacobi preconditioner ... done (shift = {:.16f})", fname, fp(shift));
    } else if(jcbBlockSize > 1) {
        long nblocks = 1 + ((size_mps - 1) / jcbBlockSize); // ceil: note that the last block may be smaller than blocksize!

        eig::log->debug("{}: calculating the block jacobi preconditioner | {} | size {} | diagonal blocksize {} | nblocks {} ...", fname,
                        eig::FactorizationToString(factorization), size_mps, jcbBlockSize, nblocks);
        std::vector<fp64> sparsity;
        auto              m_rss = debug::mem_hwm_in_mb();
        auto              t_jcb = tid::ur("jcb");
        t_jcb.tic();

        dInvJcbBlocks.clear();
        sInvJcbBlocks.clear();
        lltJcbBlocks.clear();
        luJcbBlocks.clear();
        ldltJcbBlocks.clear();
        // cgJcbBlocks.clear();
        // bicgstabJcbBlocks.clear();

#pragma omp parallel for ordered schedule(dynamic, 1)
        for(long blkidx = 0; blkidx < nblocks; ++blkidx) {
            long offset = blkidx * jcbBlockSize;
            long extent = std::min((blkidx + 1) * jcbBlockSize - offset, size_mps - offset);
            // if constexpr(eig::debug_matvec_mpos) eig::log->trace("calculating block {}/{} ... done", blkidx, nblocks);
            auto t_dblk = tid::ur("dblk");
            t_dblk.tic();

            MatrixType block;
            if(mpos_B.empty()) {
                block      = get_diagonal_block(offset, extent, mpos_A, envL_A, envR_A);
                lockCalcPc = true;
            } else {
                {
                    // Instead of using the preconditioner P = (H-shift*H²) we simply take P = H² when
                    // calculating the generalized eigenvalue problem for GSI-DMRG with ritz == LM.
                    // See the discussion in https://github.com/primme/primme/issues/83
                    block      = get_diagonal_block(offset, extent, mpos_B, envL_B, envR_B);
                    lockCalcPc = true;
                }
                // if(shift == 0.0) {
                //     block = get_diagonal_block(offset, extent, mpos_A, envL_A, envR_A);
                // } else {
                //     block         = get_diagonal_block(offset, extent, shift, mpos_A, envL_A, envR_A, mpos_B, envL_B, envR_B);
                //     factorization = eig::Factorization::LU;
                // }
                // lockCalcPc = false;
            }

            t_dblk.toc();
            double sp = static_cast<double>(block.cwiseAbs().count()) / static_cast<double>(block.size());

            // auto bicg          = std::make_unique<BICGType>();
            // auto cg            = std::make_unique<CGType>();
            auto llt      = std::make_unique<LLTType>();
            auto ldlt     = std::make_unique<LDLTType>();
            auto lu       = std::make_unique<LUType>();
            auto sparseRM = std::make_unique<SparseRowM>();
            // auto sparseCM      = std::make_unique<SparseType>();
            auto blockPtr    = std::make_unique<MatrixType>();
            bool lltSuccess  = false;
            bool ldltSuccess = false;
            bool luSuccess   = false;
            bool qrSuccess   = false;
            // bool bicgSuccess   = false;
            // bool cgSuccess     = false;
            bool invertSuccess = false;
            switch(factorization) {
                case eig::Factorization::NONE: {
                    eig::log->warn("{}: No factorization has been set for the preconditioner", fname);
                    break;
                }
                case eig::Factorization::LLT: {
                    auto t_llt = tid::ur("t_llt");
                    if constexpr(eig::debug_matvec_mpos) eig::log->trace("llt factorizing block {}/{}", blkidx, nblocks);
                    t_llt.tic();
                    VectorType D = block.diagonal().cwiseAbs().cwiseSqrt().cwiseInverse();
                    block        = (D.asDiagonal() * block * D.asDiagonal()).eval();
                    if(std::real(block(0, 0)) > RealScalar{0.0} /* Checks if the matrix is positive definite */) {
                        llt->compute(block);
                    } else {
                        llt->compute(-block); // Can sometimes be negative definite in the generalized problem
                    }
                    lltSuccess = llt->info() == Eigen::Success;
                    t_llt.toc();
                    if constexpr(eig::debug_matvec_mpos)
                        eig::log->debug("llt factorized block {}/{} : info {} thread {} tdblk {:.3e} s tllt {:.3e} s", blkidx, nblocks,
                                        static_cast<int>(llt->info()), omp_get_thread_num(), t_dblk.get_time(), t_llt.get_time());

                    // eig::log->info("-- llt time: {:.3e}", t_llt.get_last_interval());

                    if(lltSuccess) break;
                    // auto solver = Eigen::SelfAdjointEigenSolver<MatrixType>(blockI, Eigen::EigenvaluesOnly);
                    // eig::log->info("blockI evals  : \n{}\n", linalg::matrix::to_string(solver.eigenvalues(), 16));
                    // eig::log->info("shift: {:.16f} {:+.16f}i", std::real(shift), std::imag(shift));
                    // eig::log->info("blockI diagonal: \n{}\n", linalg::matrix::to_string(blockI.diagonal(), 16));
                    eig::log->info("llt factorization failed on block {}/{}: time {:.3e} info {}", blkidx, nblocks, t_llt.get_last_interval(),
                                   static_cast<int>(llt->info()));
                    [[fallthrough]];
                }
                case eig::Factorization::LU: {
                    if constexpr(eig::debug_matvec_mpos) eig::log->trace("lu factorizing block {}/{}", blkidx, nblocks);
                    auto t_lu = tid::ur("t_lu");
                    t_lu.tic();
                    VectorType D = block.diagonal().cwiseAbs().cwiseSqrt().cwiseInverse();
                    block        = (D.asDiagonal() * block * D.asDiagonal()).eval();
                    lu->compute(block);
                    t_lu.toc();
                    luSuccess = true;
                    // eig::log->info("-- lu : time {:.3e}", t_lu.get_last_interval());
                    if constexpr(eig::debug_matvec_mpos)
                        eig::log->debug("lu factorized block {}/{} : info {} thread {} time: blk {:.3e} s | lu {:.3e} s", blkidx, nblocks,
                                        static_cast<int>(llt->info()), omp_get_thread_num(), t_dblk.get_time(), t_lu.get_time());
                    break;
                }
                case eig::Factorization::LDLT: {
                    eig::log->info("-- ldlt");
                    if constexpr(eig::debug_matvec_mpos) eig::log->trace("ldlt factorizing block {}/{}", blkidx, nblocks);
                    ldlt->compute(block);
                    ldltSuccess = ldlt->info() == Eigen::Success;
                    if(ldltSuccess) break;
                    eig::log->debug("ldlt factorization failed on block {}/{}: info {}", blkidx, nblocks, static_cast<int>(ldlt->info()));
                    [[fallthrough]];
                }
                case eig::Factorization::QR: {
                    auto qr   = Eigen::HouseholderQR<MatrixType>(block);
                    block     = qr.solve(MatrixType::Identity(extent, extent));
                    qrSuccess = true;
                    break;
                }
                // case eig::Factorization::ILUT: {
                //     auto t_sparse = tid::ur("sparse");
                //     t_sparse.tic();
                //     RealScalar mean = block.cwiseAbs().mean();
                //     RealScalar stdv = std::sqrt((block.cwiseAbs().array() - mean).square().sum() / static_cast<RealScalar>(block.size() - 1));
                //     // sparseI = blockI.sparseView(mean, 1e-6);
                //     *sparseRM = block.sparseView(static_cast<RealScalar>(1e-1), stdv);
                //     sparseRM->makeCompressed();
                //     sp = static_cast<double>(sparseRM->nonZeros()) / static_cast<double>(sparseRM->size());
                //     // if constexpr(eig::debug_matvec_mpos)
                //     //     eig::log->trace("bf sparseI block {}/{}: nnz: {:.3e}", blkidx, nblocks,
                //     //                     static_cast<double>(sparseRM->nonZeros()) / static_cast<double>(sparseRM->size()));
                //     t_sparse.toc();
                //     auto t_ilut = tid::ur("ilut");
                //     t_ilut.tic();
                //     bicg->setMaxIterations(1000);
                //     bicg->setTolerance(static_cast<RealScalar>(1e-8));
                //     bicg->preconditioner().setDroptol(static_cast<RealScalar>(1e-2)); // Drops elements smaller than droptol*rowwise().cwiseAbs().mean()
                //     bicg->preconditioner().setFillfactor(
                //         static_cast<RealScalar>(2.0)); // if the original matrix has nnz nonzeros, LU matrix has at most nnz * fillfactor nonseros
                //     bicg->compute(*sparseRM);
                //     bicgSuccess = bicg->info() == Eigen::Success;
                //     t_ilut.toc();
                //     if constexpr(eig::debug_matvec_mpos)
                //         eig::log->trace("ILUT factorized block {}/{} sparcity {} : info {} thread {} t_dblk {:.3e} s t_sparse {:.3e} s t_ilut {:.3e} s mean "
                //                         "{:.3e} stdv {:.3e} iter {} tol {:.3e}",
                //                         blkidx, nblocks, sp, static_cast<int>(bicg->info()), omp_get_thread_num(), t_dblk.get_time(), t_sparse.get_time(),
                //                         t_ilut.get_time(), fp(mean), fp(stdv), bicg->iterations(), fp(bicg->tolerance()));
                //
                //     if(!bicgSuccess) {
                //         // Take the diagonal of blockI instead
                //         sp = static_cast<double>(block.cwiseAbs().count()) / static_cast<double>(block.size());
                //         sparseRM->resize(0, 0);
                //         block = MatrixType(block.diagonal().cwiseInverse().asDiagonal());
                //         break;
                //     }
                //     break;
                // }
                // case eig::Factorization::ILDLT: {
                //     auto t_sparse = tid::ur("sparse");
                //     t_sparse.tic();
                //     RealScalar mean = block.cwiseAbs().mean();
                //     RealScalar stdv = std::sqrt((block.cwiseAbs().array() - mean).square().sum() / static_cast<RealScalar>(block.size() - 1));
                //     // double geom = std::exp((blockI.array() == 0.0).select(0.0, blockI.array().cwiseAbs().log()).mean());
                //     // double droptol    = 1e-2; // Drops elements smaller than droptol*rowwise().cwiseAbs().mean()
                //     // int    fillfactor = 10;   // if the original matrix has nnz nonzeros, LDLT matrix has at most nnz * fillfactor nonseros
                //     *sparseCM = block.sparseView(static_cast<RealScalar>(1e-1), stdv);
                //     sparseCM->makeCompressed();
                //     sp = static_cast<double>(sparseCM->nonZeros()) / static_cast<double>(sparseCM->size());
                //     t_sparse.toc();
                //     auto t_ildlt = tid::ur("ildlt");
                //     t_ildlt.tic();
                //     // eig::log->info("cg compute block {}/{} ", blkidx, nblocks);
                //
                //     cg->setMaxIterations(1000);
                //     cg->setTolerance(static_cast<RealScalar>(1e-8)); // Tolerance on the residual error
                //
                //     // Read about the preconditioner parameters here http://eigen.tuxfamily.org/dox/classEigen_1_1IncompleteCholesky.html
                //     cg->preconditioner().setInitialShift(static_cast<RealScalar>(1e-3)); // default is 1e-3
                //     cg->compute(*sparseCM);
                //     cgSuccess = cg->info() == Eigen::Success;
                //     t_ildlt.toc();
                //     if constexpr(eig::debug_matvec_mpos)
                //         eig::log->trace("ILDLT factorized block {}/{} sparcity {} : info {} thread {} t_dblk {:.3e} s t_sparse {:.3e} s t_ildlt {:.3e} s mean
                //         "
                //                         "{:.3e} stdv {:.3e} iter {} tol {:.3e}",
                //                         blkidx, nblocks, sp, static_cast<int>(cg->info()), omp_get_thread_num(), t_dblk.get_time(), t_sparse.get_time(),
                //                         t_ildlt.get_time(), fp(mean), fp(stdv), cg->iterations(), fp(cg->tolerance()));
                //     if(!cgSuccess) {
                //         eig::log->info("ILDLT solve failed on block {}/{}: {}", blkidx, nblocks, static_cast<int>(cg->info()));
                //         // Take the diagonal of blockI instead
                //         sparseCM->resize(0, 0);
                //         block = MatrixType(block.diagonal().cwiseInverse().asDiagonal());
                //         break;
                //     }
                //     break;
                // }
                default: throw except::runtime_error("MatvecMPOS::CalcPc(): factorization is not implemented");
            }

            if(!lltSuccess and !luSuccess and !ldltSuccess and !qrSuccess and factorization != eig::Factorization::ILUT and
               factorization != eig::Factorization::ILDLT) {
                eig::log->warn("factorization {} (and others) failed on block {}/{} ... resorting to Eigen::ColPivHouseholderQR",
                               eig::FactorizationToString(factorization), blkidx, nblocks);
                block         = Eigen::ColPivHouseholderQR<MatrixType>(block).inverse(); // Should work on any matrix
                invertSuccess = true;
            }
#pragma omp ordered
            {
                sparsity.emplace_back(sp);
                if(invertSuccess) {
                    dInvJcbBlocks.emplace_back(offset, block); //
                } else if(lltSuccess and llt->rows() > 0) {
                    lltJcbBlocks.emplace_back(offset, std::move(llt));
                } else if(ldltSuccess and ldlt->rows() > 0) {
                    ldltJcbBlocks.emplace_back(offset, std::move(ldlt));
                } else if(luSuccess and lu->rows() > 0) {
                    luJcbBlocks.emplace_back(offset, std::move(lu));
                    // }
                    // else if(bicgSuccess and bicg->rows() > 0) {
                    //     bicgstabJcbBlocks.emplace_back(offset, std::move(sparseRM), std::move(bicg));
                    // } else if(cgSuccess and cg->rows() > 0) {
                    //     cgJcbBlocks.emplace_back(offset, std::move(sparseCM), std::move(cg));
                } else if(sparseRM->size() > 0) {
                    sInvJcbBlocks.emplace_back(offset, *sparseRM);
                }
            }
        }
        t_jcb.toc();
        auto spavg = std::accumulate(sparsity.begin(), sparsity.end(), 0.0) / static_cast<double>(sparsity.size());
        eig::log->debug("{}: calculating the block jacobi preconditioner | size {} | diagonal blocksize {} | nblocks {} ... done | t "
                        "{:.3e} s | avg "
                        "sparsity {:.3e} | mem +{:.3e} MB",
                        fname, size_mps, jcbBlockSize, nblocks, t_jcb.get_last_interval(), spavg, debug::mem_hwm_in_mb() - m_rss);
    }
    readyCalcPc = true;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultPc([[maybe_unused]] void *x, [[maybe_unused]] int *ldx, [[maybe_unused]] void *y, [[maybe_unused]] int *ldy,
                                [[maybe_unused]] int *blockSize, [[maybe_unused]] primme_params *primme, [[maybe_unused]] int *err) {
    for(int i = 0; i < *blockSize; i++) {
        Scalar shift = static_cast<Scalar>(primme->ShiftsForPreconditioner[i]);
        // if(!mpos_B.empty())
        //     shift = std::abs(primme->stats.estimateMaxEVal) > std::abs(primme->stats.estimateMinEVal) ? primme->stats.estimateMaxEVal
        //                                                                                               : primme->stats.estimateMinEVal;
        // eig::log->info("shift                             : {:.16f}", shift);
        // eig::log->info("primme->stats.estimateMaxEVal     : {:.16f}", primme->stats.estimateMaxEVal);
        // eig::log->info("primme->stats.estimateMinEval     : {:.16f}", primme->stats.estimateMinEVal);
        // eig::log->info("primme->ShiftsForPreconditioner[i]: {:.16f}", primme->ShiftsForPreconditioner[i]);
        Scalar *mps_in_ptr  = static_cast<Scalar *>(x) + *ldx * i;
        Scalar *mps_out_ptr = static_cast<Scalar *>(y) + *ldy * i;
        MultPc(mps_in_ptr, mps_out_ptr, shift);
        // MultPc(mps_in_ptr, mps_out_ptr, primme->ShiftsForPreconditioner[i]);
    }
    *err = 0;
}

template<typename Scalar>
typename MatVecMPOS<Scalar>::MatrixType MatVecMPOS<Scalar>::MultPX(const Eigen::Ref<const MatrixType> &X) {
    if(jcbMaxBlockSize == 0 or preconditioner == eig::Preconditioner::NONE) return X;
    assert(X.rows() == get_size());
    MatrixType Y(X.rows(), X.cols());
    for(Eigen::Index i = 0; i < X.cols(); ++i) { MultPc(X.col(i).data(), Y.col(i).data()); }
    return Y;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::MultPc([[maybe_unused]] const Scalar *mps_in_, [[maybe_unused]] Scalar *mps_out_, Scalar shift) {
    if(preconditioner == eig::Preconditioner::NONE) return;
    auto mps_in  = Eigen::Map<const VectorType>(mps_in_, size_mps);
    auto mps_out = Eigen::Map<VectorType>(mps_out_, size_mps);
    CalcPc(shift);
    if(jcbMaxBlockSize == 1) {
        auto token = t_multPc->tic_token();
        // Diagonal jacobi preconditioner
        invJcbDiagonal = (jcbDiagA.array() - shift * jcbDiagB.array()).matrix();
        mps_out        = invJcbDiagonal.array().cwiseInverse().cwiseProduct(mps_in.array());
        num_pc++;
    } else if(jcbMaxBlockSize > 1) {
        auto token = t_multPc->tic_token();
        // eig::log->info("-- MultPc");
#pragma omp parallel for
        for(size_t idx = 0; idx < dInvJcbBlocks.size(); ++idx) {
            const auto &[offset, block]               = dInvJcbBlocks[idx];
            long extent                               = block.rows();
            mps_out.segment(offset, extent).noalias() = block.template selfadjointView<Eigen::Lower>() * mps_in.segment(offset, extent);
        }
#pragma omp parallel for
        for(size_t idx = 0; idx < sInvJcbBlocks.size(); ++idx) {
            const auto &[offset, block]               = sInvJcbBlocks[idx];
            long extent                               = block.rows();
            mps_out.segment(offset, extent).noalias() = block.template selfadjointView<Eigen::Lower>() * mps_in.segment(offset, extent);
        }
#pragma omp parallel for
        for(size_t idx = 0; idx < lltJcbBlocks.size(); ++idx) {
            const auto &[offset, solver] = lltJcbBlocks[idx];
            long extent                  = solver->rows();
            auto mps_out_segment         = Eigen::Map<VectorType>(mps_out_ + offset, extent);
            auto mps_in_segment          = Eigen::Map<const VectorType>(mps_in_ + offset, extent);
            mps_out_segment.noalias()    = solver->solve(mps_in_segment);
        }
#pragma omp parallel for
        for(size_t idx = 0; idx < ldltJcbBlocks.size(); ++idx) {
            const auto &[offset, solver] = ldltJcbBlocks[idx];
            long extent                  = solver->rows();
            auto mps_out_segment         = Eigen::Map<VectorType>(mps_out_ + offset, extent);
            auto mps_in_segment          = Eigen::Map<const VectorType>(mps_in_ + offset, extent);
            mps_out_segment.noalias()    = solver->solve(mps_in_segment);
        }
#pragma omp parallel for
        for(size_t idx = 0; idx < luJcbBlocks.size(); ++idx) {
            const auto &[offset, solver] = luJcbBlocks[idx];
            long extent                  = solver->rows();
            auto mps_out_segment         = Eigen::Map<VectorType>(mps_out_ + offset, extent);
            auto mps_in_segment          = Eigen::Map<const VectorType>(mps_in_ + offset, extent);
            mps_out_segment.noalias()    = solver->solve(mps_in_segment);
        }
        // // #pragma omp parallel for
        // for(size_t idx = 0; idx < bicgstabJcbBlocks.size(); ++idx) {
        //     const auto &[offset, sparseI, solver]     = bicgstabJcbBlocks[idx];
        //     long extent                               = solver->rows();
        //     mps_out.segment(offset, extent).noalias() = solver->solveWithGuess(mps_in.segment(offset, extent), mps_out.segment(offset, extent));
        // }
        // // #pragma omp parallel for
        // for(size_t idx = 0; idx < cgJcbBlocks.size(); ++idx) {
        //     const auto &[offset, sparseI, solver]     = cgJcbBlocks[idx];
        //     long extent                               = solver->rows();
        //     mps_out.segment(offset, extent).noalias() = solver->solveWithGuess(mps_in.segment(offset, extent), mps_out.segment(offset, extent));
        // }
        num_pc++;
    }
}

template<typename Scalar>
void MatVecMPOS<Scalar>::print() const {}

template<typename Scalar>
void MatVecMPOS<Scalar>::reset() {
    if(t_factorOP) t_factorOP->reset();
    if(t_multOPv) t_multOPv->reset();
    if(t_genMat) t_genMat->reset();
    if(t_multAx) t_multAx->reset();
    if(t_multPc) t_multPc->reset();
    num_mv = 0;
    num_op = 0;
    num_pc = 0;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::set_shift(CplxScalar shift) {
    // Here we set an energy shift directly on the MPO.
    // This only works if the MPO is not compressed already.
    if(readyShift) return;
    if(sigma == shift) return;
    CplxScalar shift_per_mpo = shift / static_cast<RealScalar>(mpos_A.size());
    CplxScalar sigma_per_mpo = sigma / static_cast<RealScalar>(mpos_A.size());
    for(size_t idx = 0; idx < mpos_A.size(); ++idx) {
        // The MPO is a rank4 tensor ijkl where the first 2 ij indices draw a simple
        // rank2 matrix, where each element is also a matrix with the size
        // determined by the last 2 indices kl.
        // When we shift an MPO, all we do is subtract a diagonal matrix from
        // the botton left corner of the ij-matrix.
        auto &mpo  = mpos_A[idx];
        auto  dims = mpo.dimensions();
        if(dims[2] != dims[3]) throw except::logic_error("MPO has different spin dimensions up and down: {}", dims);
        auto spindim = dims[2];
        long offset1 = dims[0] - 1;

        // Setup extents and handy objects
        std::array<long, 4> offset4{offset1, 0, 0, 0};
        std::array<long, 4> extent4{1, 1, spindim, spindim};
        std::array<long, 2> extent2{spindim, spindim};
        auto                id = tenx::TensorIdentity<Scalar>(spindim);
        // We undo the previous sigma and then subtract the new one. We are aiming for [A - I*shift]
        if constexpr(tenx::sfinae::is_std_complex_v<Scalar>)
            mpo.slice(offset4, extent4).reshape(extent2) += id * (sigma_per_mpo - shift_per_mpo);
        else
            mpo.slice(offset4, extent4).reshape(extent2) += id * std::real(sigma_per_mpo - shift_per_mpo);
        eig::log->debug("Shifted MPO {} energy by {:.16f}", idx, static_cast<fp64>(std::real(shift_per_mpo)));
    }
    sigma = shift;
    if(not mpos_A_shf.empty()) {
        mpos_A_shf.clear();
        for(const auto &mpo : mpos_A) mpos_A_shf.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1}));
    }

    readyShift = true;
}

template<typename Scalar>
void MatVecMPOS<Scalar>::set_mode(const eig::Form form_) {
    form = form_;
}
template<typename Scalar>
void MatVecMPOS<Scalar>::set_side(const eig::Side side_) {
    side = side_;
}
template<typename Scalar>
void MatVecMPOS<Scalar>::set_jcbMaxBlockSize(std::optional<long> size) {
    if(size.has_value()) {
        // We want the block sizes to be roughly equal, so we reduce the block size until the remainder is zero or larger than 80% of the block size
        // This ensures that the last block isn't too much smaller than the other ones
        jcbMaxBlockSize = std::clamp(size.value(), 1l, size_mps);
        long newsize    = jcbMaxBlockSize;
        long rem        = num::mod(size_mps, newsize);
        auto bestsize   = std::pair<long, long>{rem, newsize};
        while(newsize >= jcbMaxBlockSize / 2) {
            rem = num::mod(size_mps, newsize);
            if(rem > bestsize.first or rem == 0) { bestsize = std::pair<long, long>{rem, newsize}; }
            if(rem == 0) break;              // All equal size
            if(rem > 4 * newsize / 5) break; // The last is at least 80% the size of the others
            newsize -= 2;
        }
        if(bestsize.second != jcbMaxBlockSize) {
            eig::log->trace("Adjusted block size to {}", bestsize.second);
            jcbMaxBlockSize = std::clamp(bestsize.second, jcbMaxBlockSize / 2, size_mps);
        }
        // jcbMaxBlockSize = std::clamp(size.value(), 1l, size_mps);
    }
}

template<typename Scalar>
Scalar MatVecMPOS<Scalar>::get_shift() const {
    if constexpr(tenx::sfinae::is_std_complex_v<Scalar>)
        return sigma;
    else
        return std::real(sigma);
}

template<typename Scalar>
eig::Form MatVecMPOS<Scalar>::get_form() const {
    return form;
}
template<typename Scalar>
eig::Side MatVecMPOS<Scalar>::get_side() const {
    return side;
}

template<typename Scalar>
const std::vector<Eigen::Tensor<Scalar, 4>> &MatVecMPOS<Scalar>::get_mpos() const {
    return mpos_A;
}
template<typename Scalar>
const Eigen::Tensor<Scalar, 3> &MatVecMPOS<Scalar>::get_envL() const {
    return envL_A;
}
template<typename Scalar>
const Eigen::Tensor<Scalar, 3> &MatVecMPOS<Scalar>::get_envR() const {
    return envR_A;
}

template<typename Scalar>
long MatVecMPOS<Scalar>::get_size() const {
    return size_mps;
}

template<typename Scalar>
std::array<long, 3> MatVecMPOS<Scalar>::get_shape_mps() const {
    return shape_mps;
}
template<typename Scalar>
std::vector<std::array<long, 4>> MatVecMPOS<Scalar>::get_shape_mpo() const {
    std::vector<std::array<long, 4>> shapes;
    for(const auto &mpo : mpos_A) shapes.emplace_back(mpo.dimensions());
    return shapes;
}

template<typename Scalar>
std::array<long, 3> MatVecMPOS<Scalar>::get_shape_envL() const {
    return envL_A.dimensions();
}
template<typename Scalar>
std::array<long, 3> MatVecMPOS<Scalar>::get_shape_envR() const {
    return envR_A.dimensions();
}

template<typename Scalar>
Eigen::Tensor<Scalar, 6> MatVecMPOS<Scalar>::get_tensor() const {
    if(mpos_A.size() == 1) {
        auto t_token = t_genMat->tic_token();
        eig::log->debug("Generating tensor");

        auto                     d0      = shape_mps[0];
        auto                     d1      = shape_mps[1];
        auto                     d2      = shape_mps[2];
        auto                    &threads = tenx::threads::get();
        Eigen::Tensor<Scalar, 6> tensor;
        tensor.resize(tenx::array6{d0, d1, d2, d0, d1, d2});
        tensor.device(*threads->dev) =
            envL_A.contract(mpos_A.front(), tenx::idx({2}, {0})).contract(envR_A, tenx::idx({2}, {2})).shuffle(tenx::array6{2, 0, 4, 3, 1, 5});

        return tensor;
    }
    throw except::runtime_error("MatVecMPOS<{}>::get_tensor(): Not implemented for mpos.size() > 1", sfinae::type_name<Scalar>());
}
template<typename Scalar>
Eigen::Tensor<Scalar, 6> MatVecMPOS<Scalar>::get_tensor_shf() const {
    Eigen::Tensor<Scalar, 6> tensor = get_tensor();
    return tensor.shuffle(tenx::array6{1, 2, 0, 4, 5, 3});
}

template<typename Scalar>
Eigen::Tensor<Scalar, 6> MatVecMPOS<Scalar>::get_tensor_ene() const {
    if(mpos_B.size() == 1) {
        auto t_token = t_genMat->tic_token();
        eig::log->debug("Generating tensor");

        auto                     d0      = shape_mps[0];
        auto                     d1      = shape_mps[1];
        auto                     d2      = shape_mps[2];
        auto                    &threads = tenx::threads::get();
        Eigen::Tensor<Scalar, 6> tensor;
        tensor.resize(tenx::array6{d0, d1, d2, d0, d1, d2});
        tensor.device(*threads->dev) =
            envL_B.contract(mpos_B.front(), tenx::idx({2}, {0})).contract(envR_B, tenx::idx({2}, {2})).shuffle(tenx::array6{2, 0, 4, 3, 1, 5});
        return tensor;
    }
    throw except::runtime_error("MatVecMPOS<{}>::get_tensor_ene(): Not implemented for mpos.size() > 1", sfinae::type_name<Scalar>());
}

template<typename Scalar>
typename MatVecMPOS<Scalar>::MatrixType MatVecMPOS<Scalar>::get_matrix() const {
    return tenx::MatrixCast(get_tensor(), rows(), cols());
}
template<typename Scalar>
typename MatVecMPOS<Scalar>::MatrixType MatVecMPOS<Scalar>::get_matrix_shf() const {
    return tenx::MatrixCast(get_tensor_shf(), rows(), cols());
}
template<typename Scalar>
typename MatVecMPOS<Scalar>::MatrixType MatVecMPOS<Scalar>::get_matrix_ene() const {
    return tenx::MatrixCast(get_tensor_ene(), rows(), cols());
}

template<typename Scalar>
typename MatVecMPOS<Scalar>::SparseType MatVecMPOS<Scalar>::get_sparse_matrix() const {
    // Fill lower
    std::vector<Eigen::Triplet<Scalar, long>> trip;
    trip.reserve(static_cast<size_t>(size_mps));
    // #pragma omp parallel for collapse(2)
    for(long J = 0; J < size_mps; J++) {
#pragma omp parallel for
        for(long I = J; I < size_mps; I++) {
            auto elem = get_matrix_element(I, J, mpos_A, envL_A, envR_A);
            if(std::abs(elem) > std::numeric_limits<RealScalar>::epsilon()) {
#pragma omp critical
                { trip.emplace_back(Eigen::Triplet<Scalar, long>{I, J, elem}); }
            }
        }
    }
    SparseType sparseMatrix(size_mps, size_mps);
    sparseMatrix.setFromTriplets(trip.begin(), trip.end());
    return sparseMatrix;
}

template<typename Scalar>
double MatVecMPOS<Scalar>::get_sparsity() const {
    auto sp = get_sparse_matrix();
    auto n  = static_cast<double>(size_mps);
    return (static_cast<double>(sp.nonZeros()) * 2.0 - n) / (n * n);
}
template<typename Scalar>
long MatVecMPOS<Scalar>::get_non_zeros() const {
    auto sp = get_sparse_matrix();
    return sp.nonZeros();
}

template<typename Scalar>
long MatVecMPOS<Scalar>::get_jcbMaxBlockSize() const {
    return jcbMaxBlockSize;
}

template<typename Scalar>
bool MatVecMPOS<Scalar>::isReadyFactorOp() const {
    return readyFactorOp;
}
template<typename Scalar>
bool MatVecMPOS<Scalar>::isReadyShift() const {
    return readyShift;
}
