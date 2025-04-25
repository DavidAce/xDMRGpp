#include <complex>
#ifndef lapack_complex_float
    #define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
    #define lapack_complex_double std::complex<double>
#endif

// complex must be included before lapacke!
#if defined(MKL_AVAILABLE)
    #include <mkl_lapacke.h>
#elif defined(OPENBLAS_AVAILABLE)
    #include <openblas/lapacke.h>
#elif defined(FLEXIBLAS_AVAILABLE)
    #include <flexiblas/lapacke.h>
#else
    #include <lapacke.h>
#endif

//

#include "../mpo.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/float.h"
#include "math/svd.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tid/tid.h"

//
// #include <Eigen/Cholesky>
#include "general/sfinae.h"
#include "math/linalg/tensor.h"
#include <ceres/ceres.h>
#include <cppoptlib/solver/lbfgs.h>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <Eigen/QR>
#include <LBFGS.h>
#include <math/rnd.h>

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> get_inverted_mpos_initial_guess(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, long virtual_bond = 10) {
    auto impos = mpos;
    for(auto &&[idx, impo] : iter::enumerate(impos)) {
        auto dims = impo.dimensions();
        if(idx == 0)
            dims[1] = virtual_bond;
        else if(idx + 1 == impos.size())
            dims[0] = virtual_bond;
        else {
            dims[0] = virtual_bond;
            dims[1] = virtual_bond;
        }
        impo.resize(dims);
        auto impomap = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(impo.data(), impo.size());
        if constexpr(std::is_same_v<Scalar, cx64>)
            for(long i = 0; i < impomap.size(); ++i) impomap[i] = cx64(rnd::uniform_double_box(-1.0, 1.0), 0.0);
        else
            for(long i = 0; i < impomap.size(); ++i) impomap[i] = rnd::normal<fp64>(0, 1.0);

        // impo.setRandom();
        // impo.setConstant(std::pow(2, -static_cast<double>(impos.size()))); // Make the numbers smaller
    }
    return impos;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> get_B(const std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
    // Linearize it without shuffling
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(impos[pos].data(), impos[pos].size());
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> get_B_dagger(const Eigen::EigenBase<Derived> &B, const std::array<long, 4> &dims) {
    Eigen::Tensor<typename Derived::Scalar, 4> mpo =
        Eigen::TensorMap<const Eigen::Tensor<typename Derived::Scalar, 4>>(B.derived().data(), dims).conjugate().shuffle(tenx::array4{0, 1, 3, 2});
    return tenx::VectorMap(mpo);
}

template<typename DerivedBd, typename DerivedM>
Eigen::Matrix<typename DerivedM::Scalar, Eigen::Dynamic, 1> get_BdM(const Eigen::EigenBase<DerivedBd> &Bd, const Eigen::EigenBase<DerivedM> &M,
                                                                    const std::array<long, 4> &dims4) {
    static_assert(std::is_same_v<typename DerivedBd::Scalar, typename DerivedM::Scalar>);
    using Scalar = typename DerivedM::Scalar;

    auto dims4d = std::array<long, 4>{dims4[0], dims4[1], dims4[3], dims4[2]};
    auto dims8  = std::array<long, 8>{dims4[0], dims4[1], dims4[2], dims4[3], dims4[0], dims4[1], dims4[2], dims4[3]};
    auto Bdmap  = Eigen::TensorMap<Eigen::Tensor<const Scalar, 4>>(Bd.derived().data(), dims4d);
    auto Mmap   = Eigen::TensorMap<Eigen::Tensor<const Scalar, 8>>(M.derived().data(), dims8);

    Eigen::Tensor<Scalar, 4> BdM(dims4);
    auto                    &threads = tenx::threads::get();
    BdM.device(*threads->dev)        = Bdmap.contract(Mmap, tenx::idx({0, 1, 2, 3}, {0, 1, 2, 3}));
    return tenx::VectorMap(BdM);
}
template<typename Scalar>
Eigen::Tensor<Scalar, 4> get_BdM(const Eigen::Tensor<Scalar, 4> &MLE, const Eigen::Tensor<Scalar, 4> &MRE,
                                 const Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>> &B, const Eigen::Tensor<Scalar, 4> &A) {
    auto                    &threads = tenx::threads::get();
    std::array<long, 4>      dims    = B.dimensions();
    Eigen::Tensor<Scalar, 4> BdM;
    BdM.resize(dims);
    BdM.device(*threads->dev) = MLE.contract(B.conjugate(), tenx::idx({0}, {0}))
                                    .contract(A.conjugate(), tenx::idx({0, 4}, {0, 3})) // Shuffle by contracting the opposite leg
                                    .contract(A, tenx::idx({0, 5}, {0, 2}))
                                    .contract(MRE, tenx::idx({1, 3, 4}, {0, 1, 2}))
                                    .shuffle(tenx::array4{0, 3, 2, 1});
    return BdM;
}

template<typename DerivedBd, typename DerivedM, typename DerivedB>
typename DerivedM::Scalar get_BdMB(const Eigen::EigenBase<DerivedBd> &Bd, const Eigen::EigenBase<DerivedM> &M, const Eigen::EigenBase<DerivedB> &B,
                                   const std::array<long, 4> &dims4) {
    static_assert(std::is_same_v<typename DerivedB::Scalar, typename DerivedBd::Scalar>);
    static_assert(std::is_same_v<typename DerivedB::Scalar, typename DerivedM::Scalar>);
    using Scalar = typename DerivedM::Scalar;

    auto dims4d = std::array<long, 4>{dims4[0], dims4[1], dims4[3], dims4[2]};
    auto dims8  = std::array<long, 8>{dims4[0], dims4[1], dims4[2], dims4[3], dims4[0], dims4[1], dims4[2], dims4[3]};
    auto Bdmap  = Eigen::TensorMap<Eigen::Tensor<const Scalar, 4>>(Bd.derived().data(), dims4d);
    auto Mmap   = Eigen::TensorMap<Eigen::Tensor<const Scalar, 8>>(M.derived().data(), dims8);
    auto Bmap   = Eigen::TensorMap<Eigen::Tensor<const Scalar, 4>>(B.derived().data(), dims4);

    Eigen::Tensor<Scalar, 0> BdMB;
    auto                    &threads = tenx::threads::get();
    BdMB.device(*threads->dev)       = Bdmap.contract(Mmap, tenx::idx({0, 1, 2, 3}, {0, 1, 2, 3})).contract(Bmap, tenx::idx({0, 1, 2, 3}, {0, 1, 2, 3}));
    return BdMB.coeff(0);
}

template<typename Scalar>
Scalar get_BdMB(const Eigen::Tensor<Scalar, 4> &MLE, const Eigen::Tensor<Scalar, 4> &MRE, const Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>> &B,
                const Eigen::Tensor<Scalar, 4> &A) {
    auto                    &threads = tenx::threads::get();
    Eigen::Tensor<Scalar, 0> BdMB;
    BdMB.device(*threads->dev) = MLE.contract(B.conjugate(), tenx::idx({0}, {0}))        // Shuffle the physical index
                                     .contract(A.conjugate(), tenx::idx({0, 4}, {0, 3})) // Shuffle the physical index (contract the opposite leg)
                                     .contract(A, tenx::idx({0, 5}, {0, 2}))
                                     .contract(B, tenx::idx({0, 5, 2}, {0, 2, 3}))
                                     .contract(MRE, tenx::idx({0, 1, 2, 3}, {0, 1, 2, 3}));
    return BdMB.coeff(0);
}

template<typename Scalar>
Scalar get_BdN(const Eigen::Tensor<Scalar, 2> &NLE, const Eigen::Tensor<Scalar, 2> &NRE, const Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>> &B,
               const Eigen::Tensor<Scalar, 4> &A) {
    auto                    &threads = tenx::threads::get();
    Eigen::Tensor<Scalar, 0> BdN;
    BdN.device(*threads->dev) = NLE.contract(B.conjugate(), tenx::idx({0}, {0}))
                                    .contract(A.conjugate(), tenx::idx({0, 3, 2}, {0, 2, 3}))
                                    .contract(NRE, tenx::idx({0, 1}, {0, 1})); // Shuffle by contracting the opposite leg!
    return BdN.coeff(0);
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> get_B_dagger(const std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
    // Linearize it without shuffling
    Eigen::Tensor<Scalar, 4> impo_dagger = impos[pos].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(impo_dagger.data(), impo_dagger.size());
}
template<typename Scalar>
void set_B(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &B, std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
    // Assume that B can be interpreted with the correct shape
    impos[pos] = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(B.data(), impos[pos].dimensions());
}

template<typename Scalar>
std::pair<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 4>> get_M_envs(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
    assert(mpos.size() == impos.size());
    assert(mpos.front().dimension(0) == 1);
    assert(mpos.back().dimension(1) == 1);
    assert(impos.front().dimension(0) == 1);
    assert(impos.back().dimension(1) == 1);

    Eigen::Tensor<Scalar, 4> MLE(1, 1, 1, 1), MRE(1, 1, 1, 1); // Left and right environments
    Eigen::Tensor<Scalar, 4> MLE_temp, MRE_temp;
    MLE.setConstant(1.0);
    MRE.setConstant(1.0);

    auto &threads = tenx::threads::get();
    for(size_t idx = 0; idx < pos; ++idx) {
        const auto              &mpo         = mpos[idx];
        const auto              &impo        = impos[idx];
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        MLE_temp.resize(impo_dagger.dimension(1), mpo_dagger.dimension(1), mpo.dimension(1), impo.dimension(1));
        MLE_temp.device(*threads->dev) = MLE.contract(impo_dagger, tenx::idx({0}, {0}))
                                             .contract(mpo_dagger, tenx::idx({0, 5}, {0, 2}))
                                             .contract(mpo, tenx::idx({0, 5}, {0, 2}))
                                             .contract(impo, tenx::idx({0, 5, 2}, {0, 2, 3}));
        MLE = std::move(MLE_temp);
    }
    for(size_t idx = mpos.size() - 1; idx > pos; --idx) {
        const auto              &mpo         = mpos[idx];
        const auto              &impo        = impos[idx];
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        MRE_temp.resize(impo_dagger.dimension(0), mpo_dagger.dimension(0), mpo.dimension(0), impo.dimension(0));
        MRE_temp.device(*threads->dev) = MRE.contract(impo_dagger, tenx::idx({0}, {1}))
                                             .contract(mpo_dagger, tenx::idx({0, 5}, {1, 2}))
                                             .contract(mpo, tenx::idx({0, 5}, {1, 2}))
                                             .contract(impo, tenx::idx({0, 5, 2}, {1, 2, 3}));
        MRE = std::move(MRE_temp);
    }
    return {MLE, MRE};
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> get_M(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                            const std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
    assert(mpos.size() == impos.size());
    assert(mpos.front().dimension(0) == 1);
    assert(mpos.back().dimension(1) == 1);
    assert(impos.front().dimension(0) == 1);
    assert(impos.back().dimension(1) == 1);

    Eigen::Tensor<Scalar, 4> LE(1, 1, 1, 1), RE(1, 1, 1, 1); // Left and right environments
    Eigen::Tensor<Scalar, 4> LE_temp, RE_temp;

    LE.setConstant(1.0);
    RE.setConstant(1.0);
    auto &threads = tenx::threads::get();
    for(size_t idx = 0; idx < pos; ++idx) {
        const auto              &mpo         = mpos[idx];
        const auto              &impo        = impos[idx];
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        LE_temp.resize(impo_dagger.dimension(1), mpo_dagger.dimension(1), mpo.dimension(1), impo.dimension(1));
        LE_temp.device(*threads->dev) = LE.contract(impo_dagger, tenx::idx({0}, {0}))
                                            .contract(mpo_dagger, tenx::idx({0, 5}, {0, 2}))
                                            .contract(mpo, tenx::idx({0, 5}, {0, 2}))
                                            .contract(impo, tenx::idx({0, 5, 2}, {0, 2, 3}));
        LE = std::move(LE_temp);
    }
    for(size_t idx = mpos.size() - 1; idx > pos; --idx) {
        const auto              &mpo         = mpos[idx];
        const auto              &impo        = impos[idx];
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        RE_temp.resize(impo_dagger.dimension(0), mpo_dagger.dimension(0), mpo.dimension(0), impo.dimension(0));
        RE_temp.device(*threads->dev) = RE.contract(impo_dagger, tenx::idx({0}, {1}))
                                            .contract(mpo_dagger, tenx::idx({0, 5}, {1, 2}))
                                            .contract(mpo, tenx::idx({0, 5}, {1, 2}))
                                            .contract(impo, tenx::idx({0, 5, 2}, {1, 2, 3}));
        RE = std::move(RE_temp);
    }

    const auto              &mpo         = mpos[pos];
    const auto              &impo        = impos[pos];
    Eigen::Tensor<Scalar, 4> mpo_dagger  = mpo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
    Eigen::Tensor<Scalar, 4> impo_dagger = impo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
    Eigen::Tensor<Scalar, 2> identity    = tenx::TensorIdentity<Scalar>(impo_dagger.dimension(2)); // To set up  2 dummy legs with an identity outer product

    auto rows = LE.dimension(0) * RE.dimension(0) * impo_dagger.dimension(2) * impo_dagger.dimension(3); // Should be the product of impo_dagger dims
    auto cols = LE.dimension(3) * RE.dimension(3) * mpo.dimension(2) * mpo.dimension(3);                 // Should be the product of mpo dims

    Eigen::Tensor<Scalar, 2> M(rows, cols);
    M.device(*threads->dev) = LE.contract(mpo_dagger, tenx::idx({1}, {0}))
                                  .contract(mpo, tenx::idx({1, 5}, {0, 2}))
                                  .contract(RE, tenx::idx({2, 4}, {1, 2}))
                                  .contract(identity, tenx::idx()) // Add two dummy legs
                                  .shuffle(tenx::array8{0, 4, 6, 2, 1, 5, 3, 7})
                                  .reshape(tenx::array2{rows, cols});
    return tenx::MatrixMap(M);
}

template<typename Scalar>
std::pair<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 2>> get_N_envs(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
    assert(mpos.size() == impos.size());
    assert(mpos.front().dimension(0) == 1);
    assert(mpos.back().dimension(1) == 1);
    assert(impos.front().dimension(0) == 1);
    assert(impos.back().dimension(1) == 1);

    Eigen::Tensor<Scalar, 2> NLE(1, 1), NRE(1, 1); // Left and right environments
    Eigen::Tensor<Scalar, 2> NLE_temp, NRE_temp;

    NLE.setConstant(1.0);
    NRE.setConstant(1.0);
    auto &threads = tenx::threads::get();

    for(size_t idx = 0; idx < pos; ++idx) {
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        NLE_temp.resize(impo_dagger.dimension(1), mpo_dagger.dimension(1));
        NLE_temp.device(*threads->dev) = NLE.contract(impo_dagger, tenx::idx({0}, {0})).contract(mpo_dagger, tenx::idx({0, 3, 2}, {0, 2, 3}));
        NLE                            = std::move(NLE_temp);
    }
    for(size_t idx = mpos.size() - 1; idx > pos; --idx) {
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        NRE_temp.resize(impo_dagger.dimension(0), mpo_dagger.dimension(0));
        NRE_temp.device(*threads->dev) = NRE.contract(impo_dagger, tenx::idx({0}, {1})).contract(mpo_dagger, tenx::idx({0, 3, 2}, {1, 2, 3}));
        NRE                            = std::move(NRE_temp);
    }

    return {NLE, NRE};
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> get_N(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const std::vector<Eigen::Tensor<Scalar, 4>> &impos,
                                               size_t pos) {
    assert(mpos.size() == impos.size());
    assert(mpos.front().dimension(0) == 1);
    assert(mpos.back().dimension(1) == 1);
    assert(impos.front().dimension(0) == 1);
    assert(impos.back().dimension(1) == 1);

    Eigen::Tensor<Scalar, 2> LE(1, 1), RE(1, 1); // Left and right environments
    Eigen::Tensor<Scalar, 2> LE_temp, RE_temp;

    LE.setConstant(1.0);
    RE.setConstant(1.0);
    auto &threads = tenx::threads::get();

    for(size_t idx = 0; idx < pos; ++idx) {
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        LE_temp.resize(impo_dagger.dimension(1), mpo_dagger.dimension(1));
        LE_temp.device(*threads->dev) = LE.contract(impo_dagger, tenx::idx({0}, {0})).contract(mpo_dagger, tenx::idx({0, 3, 2}, {0, 2, 3}));
        LE                            = std::move(LE_temp);
    }
    for(size_t idx = mpos.size() - 1; idx > pos; --idx) {
        Eigen::Tensor<Scalar, 4> mpo_dagger  = mpos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        Eigen::Tensor<Scalar, 4> impo_dagger = impos[idx].conjugate().shuffle(tenx::array4{0, 1, 3, 2});
        RE_temp.resize(impo_dagger.dimension(0), mpo_dagger.dimension(0));
        RE_temp.device(*threads->dev) = RE.contract(impo_dagger, tenx::idx({0}, {1})).contract(mpo_dagger, tenx::idx({0, 3, 2}, {1, 2, 3}));
        RE                            = std::move(RE_temp);
    }

    const auto              &mpo         = mpos[pos];
    const auto              &impo        = impos[pos];
    Eigen::Tensor<Scalar, 4> mpo_dagger  = mpo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});
    Eigen::Tensor<Scalar, 4> impo_dagger = impo.conjugate().shuffle(tenx::array4{0, 1, 3, 2});

    auto                     rows = LE.dimension(0) * RE.dimension(0) * mpo_dagger.dimension(2) * mpo_dagger.dimension(3);
    Eigen::Tensor<Scalar, 1> N(rows);
    N.device(*threads->dev) =
        LE.contract(mpo_dagger, tenx::idx({1}, {0})).contract(RE, tenx::idx({1}, {1})).shuffle(tenx::array4{0, 3, 1, 2}).reshape(tenx::array1{rows});
    return tenx::VectorMap(N);
}

template<typename Scalar>
Eigen::Tensor<Scalar, 4> get_N_dagger(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
    assert(mpos.size() == impos.size());
    assert(mpos.front().dimension(0) == 1);
    assert(mpos.back().dimension(1) == 1);
    assert(impos.front().dimension(0) == 1);
    assert(impos.back().dimension(1) == 1);

    Eigen::Tensor<Scalar, 2> LE(1, 1), RE(1, 1); // Left and right environments
    Eigen::Tensor<Scalar, 2> LE_temp, RE_temp;

    LE.setConstant(1.0);
    RE.setConstant(1.0);
    auto &threads = tenx::threads::get();

    for(size_t idx = 0; idx < pos; ++idx) {
        auto &mpo  = mpos[idx];
        auto &impo = impos[idx];
        LE_temp.resize(mpo.dimension(1), impo.dimension(1));
        LE_temp.device(*threads->dev) = LE.contract(mpo, tenx::idx({0}, {0})).contract(impo, tenx::idx({0, 3, 2}, {0, 2, 3}));
        LE                            = std::move(LE_temp);
    }
    for(size_t idx = mpos.size() - 1; idx > pos; --idx) {
        auto &mpo  = mpos[idx];
        auto &impo = impos[idx];
        RE_temp.resize(mpo.dimension(0), impo.dimension(0));
        RE_temp.device(*threads->dev) = RE.contract(mpo, tenx::idx({0}, {1})).contract(impo, tenx::idx({0, 3, 2}, {1, 2, 3}));
        RE                            = std::move(RE_temp);
    }

    const auto &mpo  = mpos[pos];
    const auto &impo = impos[pos];
    // auto                     rows = LE.dimension(1) * RE.dimension(1) * impo.dimension(2) * impo.dimension(3);
    Eigen::Tensor<Scalar, 4> Nd(impo.dimensions());
    Nd.device(*threads->dev) = LE.contract(mpo, tenx::idx({0}, {0})).contract(RE, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 2, 1});
    return Nd;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 1> linear_solve(Eigen::Tensor<Scalar, 2> A, Eigen::Tensor<Scalar, 1> Y) {
    auto  n    = static_cast<int>(A.dimension(0));
    auto  nrhs = 1;
    auto *a    = A.data();
    auto  lda  = static_cast<int>(A.dimension(0));
    auto *y    = Y.data();
    auto  ldy  = static_cast<int>(Y.dimension(0));
    auto  ipiv = std::vector<int>(static_cast<size_t>(A.dimension(0)));
    if constexpr(std::is_same_v<Scalar, fp64>) {
        // int info = LAPACKE_dsysv(LAPACK_COL_MAJOR, 'L', n, nhrs, a, lda, ipiv.data(), y, ldy);
        // int info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, nrhs, a, lda, ipiv.data(), y, ldy );
        int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a, lda, ipiv.data(), y, ldy);
        if(info != 0) fmt::print("LAPACKE_dsysv: failed to converge: {}\n", info);
    } else {
        // int info = LAPACKE_zsysv(LAPACK_COL_MAJOR, 'L', n, nhrs, a, lda, ipiv.data(), y, ldy);
        int info = LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, a, lda, ipiv.data(), y, ldy);
        if(info != 0) fmt::print("LAPACKE_zsysv: failed to converge: {}\n", info);
    }
    return Y;
}

template<typename Scalar>
class mpo_functor {
    public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    mpo_functor(const MatrixType &M_, const VectorType &N_, size_t L_) : M(M_), N(N_), L(L_) {}

    double operator()(const VectorType &B, VectorType &grad) {
        Scalar BMB = B.adjoint() * M * B;
        Scalar BN  = B.dot(N);
        Scalar NB  = N.dot(B);
        double fx  = std::log10(std::abs(BMB - BN - NB + std::pow(2, L)));
        double ifx = (1.0 / std::log(10) / (fx + std::numeric_limits<double>::epsilon()));
        grad       = ifx * ((M + M.transpose()) * B - 2.0 * N);
        return fx;
    }

    private:
    const MatrixType &M;
    const VectorType &N;
    size_t            L;
};

using FunctionXd = cppoptlib::function::Function<double>;
template<typename Scalar>
class mpo_functor2 : public FunctionXd {
    public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    mpo_functor2(const MatrixType &M_, const VectorType &N_, size_t L_) : M(M_), N(N_), L(L_), H(M + M.transpose()) {}

    double operator()(const VectorType &B) const override {
        Scalar BMB = B.adjoint() * M * B;
        Scalar BN  = B.dot(N);
        return BMB - 2 * BN + std::pow(2, L);
    }
    void Gradient(const VectorType &B, VectorType *grad) const override {
        if(grad == nullptr) return;
        *grad = H * B - 2.0 * N;
    }
    void Hessian([[maybe_unused]] const VectorType &B, MatrixType *hessian) const override {
        if(hessian == nullptr) return;
        *hessian = H;
    }

    private:
    const MatrixType &M;
    const VectorType &N;
    size_t            L;
    MatrixType        H;
    mutable double    error = 1;
    mutable double    fx    = 1;
};

template<typename Scalar>
class mpo_functor3 : public ceres::CostFunction {
    public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    mpo_functor3(const MatrixType &M_, const VectorType &N_, const VectorType &Nd_, size_t L_, std::array<long, 4> impo_dims_)
        : M(M_), N(N_), Nd(Nd_), L(L_), impo_dims(impo_dims_) {
        *mutable_parameter_block_sizes() = {static_cast<int>(N.rows())};
        set_num_residuals(1);
    }

    bool Evaluate(Scalar const *const *parameters, Scalar *residuals, Scalar **jacobians) const override {
        // We only expect to optimize one residual
        auto   n     = parameter_block_sizes()[0];
        auto   B     = Eigen::Map<const VectorType>(parameters[0], n);
        auto   Bd    = get_B_dagger(B, impo_dims);
        Scalar BdMB  = Bd.transpose() * M * B;
        Scalar BdN   = Bd.transpose() * N;
        Scalar NdB   = Nd.transpose() * B;
        residuals[0] = std::abs(BdMB - BdN - NdB + std::pow(2, L));
        // fmt::print("BdMB {:10.5f} BdN {:10.5f} NdB {:10.5f}\n", BdMB, BdN, NdB);
        if(jacobians != nullptr) {
            auto grad = Eigen::Map<VectorType>(jacobians[0], n);
            // VectorType MB   = M * B;
            VectorType BdM  = (Bd.transpose() * M).transpose();
            auto       sign = residuals[0] > 0 ? 1.0 : (residuals[0] == 0 ? 0.0 : -1.0);
            // grad      = sign * (MB + BdM - N - Nd);
            grad = 2 * sign * (BdM - Nd);

            // for(long i = 0; i < n; ++i)
            // fmt::print("grad={:10.5f} MB={:10.5f} BdM={:10.5f} B={:10.5f} Bd={:10.5f} N={:10.5f} Nd={:.5f}\n", grad[i], MB[i], BdM[i],
            // B[i], Bd[i], N[i], Nd[i]);
        }
        return true;
    }

    private:
    const MatrixType   &M;
    const VectorType   &N;
    const VectorType   &Nd;
    size_t              L;
    std::array<long, 4> impo_dims;

    mutable double error = 1;
    mutable double fx    = 1;
};

template<typename Scalar>
class mpo_functor4 : public ceres::FirstOrderFunction {
    public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    mpo_functor4(const MatrixType &M_, const VectorType &N_, const VectorType &Nd_, size_t L_, std::array<long, 4> impo_dims_)
        : M(M_), N(N_), Nd(Nd_), L(L_), impo_dims(impo_dims_) {}
    bool Evaluate(const double *Bptr, double *fx, double *gptr) const override {
        t_eval.tic();
        t_cost.tic();
        auto n  = N.rows();
        auto B  = Eigen::Map<const VectorType>(Bptr, n);
        auto Bd = get_B_dagger(B, impo_dims);
        // VectorType BdM  = Bd.transpose() * M;
        VectorType BdM   = get_BdM(Bd, M, impo_dims);
        Scalar     BdMB  = BdM.transpose() * B;
        Scalar     BdN   = Bd.transpose() * N;
        Scalar     NdB   = Nd.transpose() * B;
        double     error = BdMB - BdN - NdB + std::pow(2, L);
        *fx              = std::abs(error);
        // *fx = std::log10(1.0+std::abs(error));
        t_cost.toc();
        // *fx          = std::abs(error);
        if(gptr) {
            t_grad.tic();
            auto grad = Eigen::Map<VectorType>(gptr, n);
            auto sign = *fx > 0 ? 1.0 : (*fx == 0 ? 0.0 : -1.0);
            grad      = 2 * sign * (BdM - Nd);
            // auto factor = sign / std::log(10) / (1.0 + std::abs(error));
            // grad        = 2 * factor * (BdM - Nd);
            t_grad.toc();
        }
        t_eval.toc();
        return true;
    }
    int             NumParameters() const override { return static_cast<int>(N.rows()); }
    mutable tid::ur t_eval;
    mutable tid::ur t_cost;
    mutable tid::ur t_grad;

    private:
    const MatrixType   &M;
    const VectorType   &N;
    const VectorType   &Nd;
    size_t              L;
    std::array<long, 4> impo_dims;
};
template<typename Scalar>
class mpo_functor5 : public ceres::FirstOrderFunction {
    public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    mpo_functor5(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const std::vector<Eigen::Tensor<Scalar, 4>> &impos, size_t pos) {
        std::tie(MLE, MRE) = get_M_envs(mpos, impos, pos);
        std::tie(NLE, NRE) = get_N_envs(mpos, impos, pos);
        L                  = mpos.size();
        A                  = mpos[pos];
        Nd                 = get_N_dagger(mpos, impos, pos);
        impo_dims          = impos[pos].dimensions();
    }
    bool Evaluate(const double *Bptr, double *fx, double *gptr) const override {
        t_eval.tic();
        t_cost.tic();
        auto   n    = Nd.size();
        auto   B    = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(Bptr, impo_dims);
        Scalar BdMB = get_BdMB(MLE, MRE, B, A);
        Scalar BdN  = get_BdN(NLE, NRE, B, A);
        // Scalar BdMB  = BdM.transpose() * B;
        // Scalar BdN   = Bd.transpose() * N;
        Scalar NdB   = BdN; // Nd.transpose() * B; // May have to be computed from scrach? Or just conjugated
        double error = BdMB - BdN - NdB + std::pow(2, L);
        *fx          = std::abs(error);
        // *fx = std::log10(1.0+std::abs(error));
        t_cost.toc();
        // *fx          = std::abs(error);
        if(gptr) {
            t_grad.tic();
            auto BdM    = get_BdM(MLE, MRE, B, A);
            auto grad   = Eigen::Map<VectorType>(gptr, n);
            auto BdMvec = tenx::VectorMap(BdM);
            auto Ndvec  = tenx::VectorMap(Nd);
            auto sign   = *fx > 0 ? 1.0 : (*fx == 0 ? 0.0 : -1.0);
            grad        = 2 * sign * (BdMvec - Ndvec);
            // auto factor = sign / std::log(10) / (1.0 + std::abs(error));
            // grad        = 2 * factor * (BdM - Nd);
            t_grad.toc();
        }
        t_eval.toc();
        return true;
    }
    int             NumParameters() const override { return static_cast<int>(Nd.size()); }
    mutable tid::ur t_eval;
    mutable tid::ur t_cost;
    mutable tid::ur t_grad;

    private:
    Eigen::Tensor<Scalar, 4> MLE, MRE;
    Eigen::Tensor<Scalar, 2> NLE, NRE;
    Eigen::Tensor<Scalar, 4> A;
    Eigen::Tensor<Scalar, 4> Nd;
    size_t                   L;
    std::array<long, 4>      impo_dims;
};

template<typename Scalar>
std::vector<Eigen::Tensor<cx64, 4>> get_inverted_mpos_internal(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos) {
    // using MatrixType         = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType         = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    auto impos               = get_inverted_mpos_initial_guess(mpos, 72);
    auto svd_cfg             = svd::config();
    svd_cfg.truncation_limit = 1e-14;
    auto svd_solver          = svd::solver(svd_cfg);

    // auto solver = Eigen::ColPivHouseholderQR<MatrixType>();
    // auto   solver        = Eigen::FullPivHouseholderQR<MatrixType>();
    // auto solver = Eigen::PartialPivLU<MatrixType>();
    // auto solver = Eigen::FullPivLU<MatrixType>();
    // auto solver = Eigen::BiCGSTAB<MatrixType>();
    ceres::Solver::Options linoptions;
    linoptions.check_gradients                   = false;
    linoptions.max_num_iterations                = 1000;
    linoptions.dense_linear_algebra_library_type = ceres::LAPACK;
    linoptions.minimizer_progress_to_stdout      = true;

    linoptions.linear_solver_type           = ceres::LinearSolverType::ITERATIVE_SCHUR;
    linoptions.preconditioner_type          = ceres::PreconditionerType::SCHUR_POWER_SERIES_EXPANSION;
    linoptions.use_spse_initialization      = true;
    linoptions.max_num_spse_iterations      = 5;
    linoptions.spse_tolerance               = 0.1;
    linoptions.max_linear_solver_iterations = 0;

    linoptions.logging_type                            = ceres::LoggingType::SILENT;
    linoptions.function_tolerance                      = 1e-14;
    linoptions.gradient_tolerance                      = 1e-14;
    linoptions.parameter_tolerance                     = 0;
    linoptions.use_approximate_eigenvalue_bfgs_scaling = true;
    linoptions.min_line_search_step_size               = 1e-16;
    linoptions.num_threads                             = tenx::threads::getNumThreads();

    ceres::GradientProblemSolver::Options gradoptions;
    // options.check_gradients                   = true;
    gradoptions.line_search_direction_type              = ceres::LineSearchDirectionType::LBFGS;
    gradoptions.line_search_type                        = ceres::LineSearchType::WOLFE;
    gradoptions.line_search_interpolation_type          = ceres::LineSearchInterpolationType::CUBIC;
    gradoptions.nonlinear_conjugate_gradient_type       = ceres::NonlinearConjugateGradientType::POLAK_RIBIERE;
    gradoptions.max_num_iterations                      = 20;
    gradoptions.max_lbfgs_rank                          = 20;
    gradoptions.minimizer_progress_to_stdout            = true;
    gradoptions.logging_type                            = ceres::LoggingType::SILENT;
    gradoptions.function_tolerance                      = 1e-14;
    gradoptions.gradient_tolerance                      = 1e-14;
    gradoptions.parameter_tolerance                     = 1e-14;
    gradoptions.use_approximate_eigenvalue_bfgs_scaling = true;
    gradoptions.min_line_search_step_size               = 1e-14; // std::numeric_limits<double>::epsilon();

    double error     = -1.0;
    size_t sweep     = 0;
    size_t pos       = 0;
    int    dir       = 1;
    int    max_iters = gradoptions.max_num_iterations;
    while(sweep < 20) {
        gradoptions.max_num_iterations = max_iters * static_cast<int>(sweep + 1);
        auto M                         = get_M(mpos, impos, pos);
        auto N                         = get_N(mpos, impos, pos);
        auto Ndtensor                  = get_N_dagger(mpos, impos, pos);
        auto Nd                        = tenx::VectorMap(Ndtensor);

        if constexpr(std::is_same_v<Scalar, fp64>) {
            auto B = get_B(impos, pos);
            // int  niter = 0;
            // solver.compute(M);
            // B = solver.solve(N);
            // ceres::Problem        problem;
            // ceres::CostFunction  *cost_function    = new mpo_functor3(M, N, Nd, mpos.size(), impos[pos].dimensions());
            // std::vector<double *> parameter_blocks = {B.data()};
            // problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);
            // ceres::Solver::Summary summary;
            // ceres::Solve(linoptions, &problem, &summary);

            ceres::FirstOrderFunction            *cost_function = new mpo_functor5(mpos, impos, pos);
            ceres::GradientProblem                problem(cost_function);
            ceres::GradientProblemSolver::Summary summary;
            ceres::Solve(gradoptions, problem, B.data(), &summary);

            auto *mpo_func = static_cast<mpo_functor5<Scalar> *>(cost_function);

            fmt::print("pos {} | sweep {} | iters {} | error {:.5e} | {}: {} | {:.3f} func/s ({} total) | {:.3f} grad/s ({} total) \n", pos, sweep,
                       summary.iterations.size(), summary.final_cost, ceres::TerminationTypeToString(summary.termination_type), summary.message.c_str(),
                       1.0 / mpo_func->t_cost.get_time_avg(), mpo_func->t_cost.get_tic_count(), 1.0 / mpo_func->t_grad.get_time_avg(),
                       mpo_func->t_grad.get_tic_count());
            set_B(B, impos, pos);
            if(!isnan(summary.final_cost) and summary.final_cost >= 0) error = summary.final_cost;
        } else {
            auto B = VectorType(svd_solver.pseudo_inverse(M) * N);
            // auto B     = linear_solve(M, N);
            // solver.compute(M_map);
            // B            = solver.solve(N_map);
            VectorType Bd        = get_B_dagger(B, impos[pos].dimensions());
            Scalar     BdMB      = Bd.transpose() * M * B;
            Scalar     BdN       = Bd.transpose() * N;
            Scalar     NdB       = Nd.transpose() * B;
            double     new_error = std::abs(BdMB - BdN - NdB + std::pow(2, mpos.size()));
            if(!std::isnan(new_error) and (new_error < error * 100 or sweep < 5)) {
                set_B(B, impos, pos);
                error = new_error;
            }
            fmt::print("error {:3}: {:.3e} | size {}\n", sweep, error, N.size());
        }
        if(std::abs(error) < 1e-10) break;
        if(std::isnan(error)) break;
        if(pos + 1 == mpos.size()) {
            sweep++;
            dir = -1;
        }
        if(pos == 0) {
            if(dir == -1) sweep++; // Increment each time we return
            dir = 1;
        }
        if(dir > 0) pos++;
        if(dir < 0) pos--;
    }
    if constexpr(std::is_same_v<Scalar, cx64>)
        return impos;
    else {
        std::vector<Eigen::Tensor<cx64, 4>> impos_cplx;
        impos_cplx.reserve(impos.size());
        for(const auto &impo : impos) impos_cplx.emplace_back(impo.template cast<cx64>());
        return impos_cplx;
    }
}

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_inverted_mpos(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos) {
    fmt::print("INIT: num_threads: {} | omp_in_parallel {} \n", tenx::threads::getNumThreads(), omp_in_parallel());
    static bool googleLogginghasInitialized = false;
    if(not googleLogginghasInitialized) {
        googleLogginghasInitialized = true;
#if defined(CERCES_INTERNAL_MINIGLOG_GLOG_LOGGING_H_)
        // google::InitGoogleLogging();
#else
        google::InitGoogleLogging(tools::log->name().c_str());
        google::SetStderrLogging(3);
#endif
    }
    bool isComplex = false;
    for(const auto &mpo : mpos) {
        if(!tenx::isReal(mpo)) {
            isComplex = true;
            break;
        }
    }
    if(isComplex) {
        return get_inverted_mpos_internal<cx64>(mpos);
    } else {
        std::vector<Eigen::Tensor<fp64, 4>> mpos_real;
        mpos_real.reserve(mpos.size());
        for(const auto &mpo : mpos) mpos_real.emplace_back(mpo.real());
        return get_inverted_mpos_internal<fp64>(mpos_real);
    }
}

template std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_inverted_mpos(const std::vector<Eigen::Tensor<cx64, 4>> &mpos);