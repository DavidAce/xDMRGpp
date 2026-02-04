#pragma once
#include "config/settings.h"
#include "expectation_value.h"
#include "hamiltonian.h"
#include "math/num.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/contraction/matvec_policy.h"
#include "tools/common/log.h"
#include <Eigen/Eigenvalues>

namespace settings {
    constexpr bool debug_hamiltonian = true;
}

using tools::finite::measure::RealScalar;

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const Eigen::Tensor<Scalar, 3> &mps, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    auto mpo  = model.get_mpo_active();
    auto env  = edges.get_ene_active();
    auto t_H2 = tid::tic_scope("H", tid::level::highest);

    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    const auto &mps = state.get_mps_active();
    const auto &mpo = model.get_mpo_active();
    const auto &env = edges.get_ene_active();
    auto        t_H = tid::tic_scope("H", tid::level::highest);
    // This only works if mps contains the center (e.g. AC, [AC, B], [A, AC] and so on
    bool has_center = std::count(state.active_sites.begin(), state.active_sites.end(), state.template get_position<long>()) > 0;
    if(has_center) {
        return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
    } else {
        auto mmps = state.template get_multisite_mps<Scalar>();
        return tools::finite::measure::expectation_value<Scalar>(mmps, mmps, mpo, env);
    }
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                  const EdgesFinite<Scalar> &edges) {
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));

    const auto &mps = state.get_mps(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_ene(sites);
    auto        t_H = tid::tic_scope("H", tid::level::highest);

    // This only works if mps contains the center (e.g. AC, [AC, B], [A, AC] and so on
    bool has_center = std::count(state.active_sites.begin(), state.active_sites.end(), state.template get_position<long>()) > 0;
    if(has_center) {
        return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
    } else {
        auto mmps = state.template get_multisite_mps<Scalar>();
        return tools::finite::measure::expectation_value<Scalar>(mmps, mmps, mpo, env);
    }
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                  const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                  const env_pair<const EnvEne<Scalar> &>                           &envs) {
    auto t_H = tid::tic_scope("H", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo_refs, envs);
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<Scalar, 3> &mps, const ModelFinite<Scalar> &model,
                                                          const EdgesFinite<Scalar> &edges) {
    assert(num::all_equal(model.active_sites, edges.active_sites));
    auto mpo = model.get_mpo_active();
    auto env = edges.get_var_active();
    auto t_H = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                          const EdgesFinite<Scalar> &edges) {
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    auto mps  = state.get_mps_active();
    auto mpo  = model.get_mpo_active();
    auto env  = edges.get_var_active();
    auto t_H2 = tid::tic_scope("H2", tid::level::highest);

    // This only works if mps contains the center (e.g. AC, [AC, B], [A, AC] and so on
    bool has_center = std::count(state.active_sites.begin(), state.active_sites.end(), state.template get_position<long>()) > 0;
    if(has_center) {
        return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
    } else {
        auto mmps = state.template get_multisite_mps<Scalar>();
        return tools::finite::measure::expectation_value<Scalar>(mmps, mmps, mpo, env);
    }
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                          const EdgesFinite<Scalar> &edges) {
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    const auto &mps = state.get_mps(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_var(sites);
    auto        t_H = tid::tic_scope("H2", tid::level::highest);
    // This only works if mps contains the center (e.g. AC, [AC, B], [A, AC] and so on
    bool has_center = std::count(state.active_sites.begin(), state.active_sites.end(), state.template get_position<long>()) > 0;
    if(has_center) {
        return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
    } else {
        auto mmps = state.template get_multisite_mps<Scalar>();
        return tools::finite::measure::expectation_value<Scalar>(mmps, mmps, mpo, env);
    }
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                          const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                          const env_pair<const EnvVar<Scalar> &>                           &envs) {
    auto t_H2 = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo_refs, envs);
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const TensorsFinite<Scalar> &tensors) {
    return tools::finite::measure::expval_hamiltonian<Scalar>(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const TensorsFinite<Scalar> &tensors) {
    return tools::finite::measure::expval_hamiltonian_squared<Scalar>(*tensors.state, *tensors.model, *tensors.edges);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::local_operator_norm_estimate(const Eigen::Tensor<Scalar, 4> &mpo, const Eigen::Tensor<Scalar, 3> &envL,
                                                                        const Eigen::Tensor<Scalar, 3> &envR, Eigen::Index maxiter,
                                                                        fp32 reltol // e.g. 1e-3
) {
    using Real    = RealScalar<Scalar>;
    using VecType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VecReal = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    const Eigen::Index size_mps  = envL.dimension(0) * envR.dimension(0) * mpo.dimension(2);
    const auto         shape_mps = std::array{mpo.dimension(2), envL.dimension(0), envR.dimension(0)};

    Eigen::Tensor<Scalar, 3> vt(shape_mps);
    Eigen::Tensor<Scalar, 3> wt(shape_mps);
    auto                     v_map = Eigen::Map<VecType>(vt.data(), size_mps);
    auto                     w_map = Eigen::Map<VecType>(wt.data(), size_mps);

    VecType v = VecType::Random(size_mps).normalized();

    auto mvopts = MatVecRaiiOptions(MatVecBackend::TBLIS);

    Real         lambda    = Real{0};
    Eigen::Index krylovdim = 3;
    for(Eigen::Index iter = 0; iter < maxiter; ++iter) {
        const Eigen::Index p = std::max<Eigen::Index>(2, krylovdim);

        // Build Krylov matrix K = [v, Av, A^2 v, ...]
        MatType K(size_mps, p);
        K.col(0) = v;

        for(Eigen::Index j = 1; j < p; ++j) {
            v_map = K.col(j - 1);
            tools::common::contraction::matrix_vector_product(wt, vt, mpo, envL, envR);
            K.col(j) = w_map;
        }

        // Orthonormalize K
        Eigen::HouseholderQR<MatType> qr(K);

        // Estimate effective subspace size from |R(j,j)|
        // (R is p x p upper triangular inside matrixQR)
        const MatType QR = qr.matrixQR().topLeftCorner(p, p);
        VecReal       diag_abs(p);
        for(Eigen::Index j = 0; j < p; ++j) diag_abs(j) = std::abs(QR(j, j));

        const Real diag0    = (p > 0) ? diag_abs(0) : Real{0};
        const Real drop_tol = std::max(Real{1e-20f}, Real{1e-12f} * diag0);

        Eigen::Index k_eff = 0;
        for(Eigen::Index j = 0; j < p; ++j) {
            if(diag_abs(j) > drop_tol)
                ++k_eff;
            else
                break;
        }
        k_eff = std::max<Eigen::Index>(k_eff, 1);

        // Form thin Q explicitly: Q = qr.householderQ() * I(:, 0:k_eff-1)
        // This is n x k_eff, k_eff is small so the explicit form is usually fine.
        MatType Q = qr.householderQ() * MatType::Identity(size_mps, k_eff).eval();

        // W = A Q (compute each column with your matvec)
        MatType W(size_mps, k_eff);
        for(Eigen::Index j = 0; j < k_eff; ++j) {
            v_map = Q.col(j);
            tools::common::contraction::matrix_vector_product(wt, vt, mpo, envL, envR);
            W.col(j) = w_map;
        }

        // Projected operator T = Q^* (A Q) = Q^* W
        MatType T = (Q.adjoint() * W).eval();
        T         = ((T + T.adjoint()) / Real{2}).eval(); // suppress tiny non-Hermitian noise

        Eigen::SelfAdjointEigenSolver<MatType> es(T);
        if(es.info() != Eigen::Success) break;

        const auto &evals = es.eigenvalues();
        const auto &evecs = es.eigenvectors();

        Eigen::Index idx = 0;
        evals.cwiseAbs().maxCoeff(&idx);

        const Scalar theta      = Scalar(evals(idx));
        const Real   lambda_new = std::abs(evals(idx));

        // Ritz vector y = Q x
        VecType    y      = (Q * evecs.col(idx)).eval();
        const Real y_norm = y.norm();
        if(y_norm > Real{0}) y /= y_norm;

        // Ritz residual r = W x - theta (Q x)
        VecType    r      = (W * evecs.col(idx) - (Q * evecs.col(idx)) * theta).eval();
        const Real r_norm = r.norm();

        v      = std::move(y);
        lambda = lambda_new;
        tools::log->debug("iter {:<2}: lambda = {:.8e}", iter, fp(lambda));

        if(lambda > Real{0} && r_norm < static_cast<Real>(reltol) * lambda) break;
    }

    return lambda;
}

template<typename Scalar> RealScalar<Scalar> tools::finite::measure::local_hamiltonian_norm(const TensorsFinite<Scalar> &tensors, Eigen::Index maxiter,
                                                                                            fp32 reltol) {
    return local_hamiltonian_norm(tensors.get_model(), tensors.get_edges(), maxiter, reltol);
}
template<typename Scalar> RealScalar<Scalar> tools::finite::measure::local_hamiltonian_norm(const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                                                            Eigen::Index maxiter, fp32 reltol) {
    return local_hamiltonian_norm(model.get_mpo_active(), edges.get_ene_active(), maxiter, reltol);
}
template<typename Scalar> RealScalar<Scalar>
    tools::finite::measure::local_hamiltonian_norm(const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                   const env_pair<const EnvEne<Scalar> &> &envs, Eigen::Index maxiter, fp32 reltol) {
    using Scalar32 = typename std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, cx32, fp32>;
    if(mpo_refs.size() != 1) { throw except::runtime_error("norm_hamiltonian_squared: Expected 1-site mpo. Got {} mpo sites.", mpo_refs.size()); }
    Eigen::Tensor<Scalar32, 3> envL = envs.L.template get_block_as<Scalar32>();
    Eigen::Tensor<Scalar32, 3> envR = envs.R.template get_block_as<Scalar32>();
    Eigen::Tensor<Scalar32, 4> mpo  = mpo_refs.front().get().template MPO_as<Scalar32>();
    return static_cast<RealScalar<Scalar>>(local_operator_norm_estimate(mpo, envL, envR, maxiter, reltol));
}

template<typename Scalar> RealScalar<Scalar> tools::finite::measure::local_hamiltonian_squared_norm(const TensorsFinite<Scalar> &tensors, Eigen::Index maxiter,
                                                                                                    fp32 reltol) {
    return local_hamiltonian_squared_norm(tensors.get_model(), tensors.get_edges(), maxiter, reltol);
}
template<typename Scalar> RealScalar<Scalar> tools::finite::measure::local_hamiltonian_squared_norm(const ModelFinite<Scalar> &model,
                                                                                                    const EdgesFinite<Scalar> &edges, Eigen::Index maxiter,
                                                                                                    fp32 reltol) {
    return local_hamiltonian_squared_norm(model.get_mpo_active(), edges.get_var_active(), maxiter, reltol);
}

template<typename Scalar> RealScalar<Scalar>
    tools::finite::measure::local_hamiltonian_squared_norm(const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                           const env_pair<const EnvVar<Scalar> &> &envs, Eigen::Index maxiter, fp32 reltol) {
    using Scalar32 = typename std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex == 1, cx32, fp32>;
    if(mpo_refs.size() != 1) { throw except::runtime_error("norm_hamiltonian_squared: Expected 1-site mpo. Got {} mpo sites.", mpo_refs.size()); }
    Eigen::Tensor<Scalar32, 3> envL = envs.L.template get_block_as<Scalar32>();
    Eigen::Tensor<Scalar32, 3> envR = envs.R.template get_block_as<Scalar32>();
    Eigen::Tensor<Scalar32, 4> mpo2 = mpo_refs.front().get().template MPO2_as<Scalar32>();
    return static_cast<RealScalar<Scalar>>(local_operator_norm_estimate(mpo2, envL, envR, maxiter, reltol));
}

template<typename Scalar> RealScalar<Scalar> tools::finite::measure::global_hamiltonian_trace(const TensorsFinite<Scalar> &tensors) {
    return global_hamiltonian_trace(tensors.get_model(), tensors.get_edges());
}
template<typename Scalar> RealScalar<Scalar> tools::finite::measure::global_hamiltonian_trace(const ModelFinite<Scalar> &model,
                                                                                              const EdgesFinite<Scalar> &edges) {
    auto numsites = model.MPO.size();
    auto sites    = num::range<size_t>(0, numsites);
    auto mpos     = model.get_mpo(sites);
    auto envs     = edges.get_multisite_env_ene(sites);
    return global_hamiltonian_trace(mpos, envs);
}

template<typename Scalar> RealScalar<Scalar>
    tools::finite::measure::global_hamiltonian_trace(const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                     const env_pair<const EnvEne<Scalar> &>                           &envs) {
    std::vector<size_t> sites;
    for(const auto &mpo : mpo_refs) { sites.emplace_back(mpo.get().get_position()); }

    if(sites.empty()) throw std::runtime_error("global_hamiltonian_trace: No sites on which to trace a hamiltonian");
    if(envs.L.get_position() != sites.front())
        throw except::runtime_error("global_hamiltonian_trace: Position mismatch: envs.L and sites.front():  {} != {}", envs.L.get_position(), sites.front());
    if(envs.R.get_position() != sites.back())
        throw except::runtime_error("global_hamiltonian_trace: Position mismatch: envs.R and sites.front():  {} != {}", envs.R.get_position(), sites.front());
    if(mpo_refs.front().get().MPO().dimension(0) != envs.L.get_block().dimension(2))
        throw except::runtime_error("mpo and env.L virtual bond dimension mismatch {} != {}", mpo_refs.front().get().MPO().dimension(0),
                                    envs.L.get_block().dimension(2));
    if(mpo_refs.back().get().MPO().dimension(1) != envs.R.get_block().dimension(2))
        throw except::runtime_error("mpo and env.R virtual bond dimension mismatch {} != {}", mpo_refs.back().get().MPO().dimension(1),
                                    envs.R.get_block().dimension(2));

    auto  t_mpo   = tid::tic_scope("global_hamiltonian_trace", tid::level::highest);
    auto &threads = tenx::threads::get();

    Eigen::Tensor<Scalar, 2> multisite_mpo, mpoL, mpoR;

    tools::log->trace("Contracting multisite mpo tensor with sites {} ", sites);

    /* We prepend envL to the first mpo.
     *    |---0
     *    |
     *   [L]--2   --[trace(0,1) and reshape]-->  0---[L]---1   *    (0)---[M]---1   =  0---[LM]---1
     *    |
     *    |___1
     */

    {
        const auto                    &envL        = envs.L.get_block();
        const auto                    &dimL        = envL.dimensions();
        const Eigen::Tensor<Scalar, 2> envL_traced = envL.trace(std::array{0, 1}).reshape(std::array{1l, dimL[2]});
        mpoL                                       = mpo_refs.front().get().MPO().trace(std::array{2, 3});
        multisite_mpo.resize(std::array{envL_traced.dimension(0), mpoL.dimension(1)});
        multisite_mpo.device(*threads->dev) = envL_traced.contract(mpoL, tenx::idx({1}, {0}));
    }

    for(size_t idx = 1; idx < sites.size(); ++idx) {
        const auto &mpo = mpo_refs[idx].get();
        const auto  pos = mpo.get_position();
        if constexpr(settings::debug_hamiltonian) tools::log->trace("contracting position {} at idx {}", pos, idx);

        mpoL = multisite_mpo;
        mpoR = mpo.MPO().trace(std::array{2, 3});

        // Append mpoR to the chain
        multisite_mpo.resize(std::array{mpoL.dimension(0), mpoR.dimension(1)});
        multisite_mpo.device(*threads->dev) = mpoL.contract(mpoR, tenx::idx({1}, {0}));
    }
    /* We append envR to the chain.
     *    0---|
     *        |
     *    2--[R]   --[trace(0,1) and reshape]-->     0---[M]---1 *  0---[R]---1    =  0---[LM]---1
     *        |
     *    1---|
     */
    {
        const auto                    &envR        = envs.R.get_block();
        const auto                    &dimR        = envR.dimensions();
        const Eigen::Tensor<Scalar, 2> envR_traced = envR.trace(std::array{0, 1}).reshape(std::array{dimR[2], 1l});
        mpoR.resize(std::array{multisite_mpo.dimension(0), envR_traced.dimension(1)});
        mpoR.device(*threads->dev) = multisite_mpo.contract(envR_traced, tenx::idx({1}, {0}));
    }

    assert(mpoR.size() == 1);
    return std::real(mpoR.coeff(0));
}

template<typename Scalar> RealScalar<Scalar> tools::finite::measure::global_hamiltonian_squared_trace(const TensorsFinite<Scalar> &tensors) {
    return global_hamiltonian_squared_trace(tensors.get_model(), tensors.get_edges());
}
template<typename Scalar> RealScalar<Scalar> tools::finite::measure::global_hamiltonian_squared_trace(const ModelFinite<Scalar> &model,
                                                                                                      const EdgesFinite<Scalar> &edges) {
    auto numsites = model.MPO.size();
    auto sites    = num::range<size_t>(0, numsites);
    auto mpos     = model.get_mpo(sites);
    auto envs     = edges.get_multisite_env_var(sites);
    return global_hamiltonian_squared_trace(mpos, envs);
}

template<typename Scalar> RealScalar<Scalar>
    tools::finite::measure::global_hamiltonian_squared_trace(const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                             const env_pair<const EnvVar<Scalar> &>                           &envs) {
    std::vector<size_t> sites;
    for(const auto &mpo : mpo_refs) { sites.emplace_back(mpo.get().get_position()); }

    if(sites.empty()) throw std::runtime_error("global_hamiltonian_trace: No sites on which to trace a hamiltonian");
    if(envs.L.get_position() != sites.front())
        throw except::runtime_error("global_hamiltonian_trace: Position mismatch: envs.L and sites.front():  {} != {}", envs.L.get_position(), sites.front());
    if(envs.R.get_position() != sites.back())
        throw except::runtime_error("global_hamiltonian_trace: Position mismatch: envs.R and sites.front():  {} != {}", envs.R.get_position(), sites.front());
    if(mpo_refs.front().get().MPO2().dimension(0) != envs.L.get_block().dimension(2))
        throw except::runtime_error("mpo2 and env.L virtual bond dimension mismatch {} != {}", mpo_refs.front().get().MPO2().dimension(0),
                                    envs.L.get_block().dimension(2));
    if(mpo_refs.back().get().MPO2().dimension(1) != envs.R.get_block().dimension(2))
        throw except::runtime_error("mpo2 and env.R virtual bond dimension mismatch {} != {}", mpo_refs.back().get().MPO2().dimension(1),
                                    envs.R.get_block().dimension(2));

    auto  t_mpo   = tid::tic_scope("global_hamiltonian_trace", tid::level::highest);
    auto &threads = tenx::threads::get();

    Eigen::Tensor<Scalar, 2> multisite_mpo, mpoL, mpoR;

    tools::log->trace("Contracting multisite mpo tensor with sites {} ", sites);

    /* We prepend envL to the first mpo.
     *    |---0
     *    |
     *   [L]--2   --[trace(0,1) and reshape]-->  0---[L]---1   *    (0)---[M]---1   =  0---[LM]---1
     *    |
     *    |___1
     */

    {
        const auto                    &envL        = envs.L.get_block();
        const auto                    &dimL        = envL.dimensions();
        const Eigen::Tensor<Scalar, 2> envL_traced = envL.trace(std::array{0, 1}).reshape(std::array{1l, dimL[2]});
        mpoL                                       = mpo_refs.front().get().MPO2().trace(std::array{2, 3});
        multisite_mpo.resize(std::array{envL_traced.dimension(0), mpoL.dimension(1)});
        multisite_mpo.device(*threads->dev) = envL_traced.contract(mpoL, tenx::idx({1}, {0}));
    }

    for(size_t idx = 1; idx < sites.size(); ++idx) {
        const auto &mpo = mpo_refs[idx].get();
        const auto  pos = mpo.get_position();
        if constexpr(settings::debug_hamiltonian) tools::log->trace("contracting position {} at idx {}", pos, idx);

        mpoL = multisite_mpo;
        mpoR = mpo.MPO2().trace(std::array{2, 3});

        // Append mpoR to the chain
        multisite_mpo.resize(std::array{mpoL.dimension(0), mpoR.dimension(1)});
        multisite_mpo.device(*threads->dev) = mpoL.contract(mpoR, tenx::idx({1}, {0}));
    }
    /* We append envR to the chain.
     *    0---|
     *        |
     *    2--[R]   --[trace(0,1) and reshape]-->     0---[M]---1 *  0---[R]---1    =  0---[LM]---1
     *        |
     *    1---|
     */
    {
        const auto                    &envR        = envs.R.get_block();
        const auto                    &dimR        = envR.dimensions();
        const Eigen::Tensor<Scalar, 2> envR_traced = envR.trace(std::array{0, 1}).reshape(std::array{dimR[2], 1l});
        mpoR.resize(std::array{multisite_mpo.dimension(0), mpoR.dimension(1)});
        mpoR.device(*threads->dev) = multisite_mpo.contract(envR_traced, tenx::idx({1}, {0}));
    }

    assert(mpoR.size() == 1);
    return std::real(mpoR.coeff(0));
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                     const EdgesFinite<Scalar> &edges, MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy_minus_energy_shift) {
        if constexpr(!settings::debug_hamiltonian) {
            // Return the cache hit when not debugging. Otherwise, check that it is correct!
            // tools::log->trace("energy_minus_energy_shift: cache hit: {:.16f}", measurements->energy_minus_energy_shift.value());
            return measurements->energy_minus_energy_shift.value();
        }
    }
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    auto t_ene = tid::tic_scope("ene", tid::level::highest);
    if constexpr(settings::debug) tools::log->trace("Measuring energy: sites {}", state.active_sites);
    auto e_minus_ered = expval_hamiltonian<Scalar>(state, model, edges);
    if constexpr(settings::debug_hamiltonian) {
        [[maybe_unused]] constexpr auto tol           = static_cast<RealScalar<Scalar>>(1e-12);
        const auto                     &multisite_mps = state.template get_multisite_mps<Scalar>();
        const auto                     &multisite_mpo = model.template get_multisite_mpo<Scalar>();
        const auto                     &multisite_env = edges.get_multisite_env_ene_blk();
        auto                            edbg = tools::common::contraction::expectation_value(multisite_mps, multisite_mpo, multisite_env.L, multisite_env.R);
        tools::log->trace("e_minus_ered: {:.16f}", fp(e_minus_ered));
        tools::log->trace("e_minus_edbg: {:.16f}", fp(edbg));
        if(measurements != nullptr and measurements->energy_minus_energy_shift) {
            tools::log->trace("e_minus_ehit: {:.16f}", fp(measurements->energy_minus_energy_shift.value()));
            assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < tol);
        }
        assert(std::abs(e_minus_ered - edbg) < tol);
    }

    assert(std::abs(std::imag(e_minus_ered)) < static_cast<RealScalar<Scalar>>(1e-10));
    if(measurements != nullptr) measurements->energy_minus_energy_shift = std::real(e_minus_ered);
    return std::real(e_minus_ered);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                                     const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg,
                                                                     MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy_minus_energy_shift) {
        if constexpr(!settings::debug_hamiltonian) {
            // Return the cache hit when not debugging. Otherwise, check that it is correct!
            // tools::log->trace("energy_minus_energy_shift: cache hit: {:.16f}", measurements->energy_minus_energy_shift.value());
            return measurements->energy_minus_energy_shift.value();
        }
    }
    auto t_ene = tid::tic_scope("ene", tid::level::highest);
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    assert(num::all_equal(model.active_sites, edges.active_sites));
    // Check if we can contract directly or if we need to use the split method
    // Normally it's only worth splitting the multisite mps when it has more than 3 sites
    constexpr auto nan          = std::numeric_limits<RealScalar<Scalar>>::quiet_NaN();
    auto           e_minus_ered = Scalar{nan};
    if(model.active_sites.size() <= 3) {
        // Contract directly
        const auto &mpo = model.template get_multisite_mpo<Scalar>();
        const auto &env = edges.get_multisite_env_ene_blk();
        if constexpr(settings::debug_hamiltonian)
            tools::log->trace("Measuring energy: multisite_mps dims {} | model sites {} dims {} | edges sites {} dims [L{} R{}]", multisite_mps.dimensions(),
                              model.active_sites, mpo.dimensions(), edges.active_sites, env.L.dimensions(), env.R.dimensions());
        e_minus_ered = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
        if constexpr(settings::debug_hamiltonian) {
            // Split the multisite mps first
            const auto mpos = model.get_mpo_active();
            const auto envs = edges.get_ene_active();
            const auto edbg = tools::finite::measure::expectation_value<Scalar>(multisite_mps, mpos, envs, svd_cfg);
            tools::log->trace("e_minus_ered: {:.16f}", fp(e_minus_ered));
            tools::log->trace("e_minus_edbg: {:.16f}", fp(edbg));
            if(measurements != nullptr and measurements->energy_minus_energy_shift) {
                tools::log->trace("e_minus_ehit: {:.16f}", fp(measurements->energy_minus_energy_shift.value()));
                assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < RealScalar<Scalar>{1e-14f});
            }
            assert(std::abs(e_minus_ered - edbg) < RealScalar<Scalar>{1e-14f});
        }
    } else {
        model.clear_cache();
        // Split the multisite mps first
        const auto mpos = model.get_mpo_active();
        const auto envs = edges.get_ene_active();
        tools::log->trace("Measuring energy: multisite_mps dims {} | sites {} | eshift {:.16f} | norm {:.16f}", multisite_mps.dimensions(), model.active_sites,
                          fp(model.get_energy_shift_mpo()), fp(tenx::norm(multisite_mps)));
        e_minus_ered = tools::finite::measure::expectation_value<Scalar>(multisite_mps, mpos, envs, svd_cfg);
        if constexpr(settings::debug_hamiltonian) {
            [[maybe_unused]] constexpr auto tol  = static_cast<RealScalar<Scalar>>(1e-14);
            const auto                     &mpo  = model.template get_multisite_mpo<Scalar>();
            const auto                     &env  = edges.get_multisite_env_ene_blk();
            const auto                      edbg = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
            tools::log->trace("e_minus_ered: {:.16f}", fp(e_minus_ered));
            tools::log->trace("e_minus_edbg: {:.16f}", fp(edbg));
            if(measurements != nullptr and measurements->energy_minus_energy_shift) {
                tools::log->trace("e_minus_ehit: {:.16f}", fp(measurements->energy_minus_energy_shift.value()));
                assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < tol);
            }
            assert(std::abs(e_minus_ered - edbg) < tol);
        }
    }
    assert(std::abs(std::imag(e_minus_ered)) < static_cast<RealScalar<Scalar>>(1e-10));
    if(measurements != nullptr) measurements->energy_minus_energy_shift = std::real(e_minus_ered);
    return std::real(e_minus_ered);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                  MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy) return measurements->energy.value();
    // This measures the actual energy of the system regardless of the energy shift in the MPO's
    // If they are shifted, then
    //      "Actual energy" = (E - E_shift) + E_shift = (~0) + E_shift = E
    // Else
    //      "Actual energy" = (E - E_shift) + E_shift = E  + 0 = E
    auto e_minus_eshift = tools::finite::measure::energy_minus_energy_shift(state, model, edges, measurements);
    auto eshift         = std::real(model.get_energy_shift_mpo());
    auto energy         = e_minus_eshift + eshift;
    if(measurements != nullptr) measurements->energy = energy;
    return energy;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                  const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg,
                                                  MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy) return measurements->energy.value();
    // This measures the actual energy of the system regardless of the energy shift in the MPO's
    // If they are shifted, then
    //      "Actual energy" = (E - E_shift) + E_shift = (~0) + E_shift = E
    // Else
    //      "Actual energy" = (E - E_shift) + E_shift = E  + 0 = E
    auto e_minus_eshift = tools::finite::measure::energy_minus_energy_shift(multisite_mps, model, edges, svd_cfg, measurements);
    auto eshift         = std::real(model.get_energy_shift_mpo());
    auto energy         = e_minus_eshift + eshift;
    if(measurements != nullptr) measurements->energy = energy;
    return energy;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                           MeasurementsTensorsFinite<Scalar> *measurements) {
    // Here we show that the variance calculated with energy-shifted mpo²'s is equivalent to the usual way.
    // If mpo's are shifted in the mpo²:
    //      Var H = <(H-E_shf)²> - <H-E_shf>²     = <H²>  - 2<H>E_shf + E_shf² - (<H> - E_shf)²
    //                                            = H²    - 2*E*E_shf + E_shf² - E² + 2*E*E_shf - E_shf²
    //                                            = H²    - E²
    //      Note that in the last line, H²-E² is a subtraction of two large numbers --> catastrophic cancellation --> loss of precision.
    //      On the other hand Var H = <(H-E_shf)²> - energy_minus_energy_shift² = <(H-E_red)²> - ~dE², where both terms are always  << 1.
    //      The first term is computed from a double-layer of shifted mpo's.
    //      In the second term dE is usually very small, in fact identically zero immediately after an energy-reduction operation,
    //      but may grow if the optimization steps make significant progress refining E. Thus wethe first term is a good approximation to
    //      the variance by itself.
    //
    // Else, if E_shf = 0 (i.e. not shifted) we get the usual formula:
    //      Var H = <(H - 0)²> - <H - 0>² = H² - E²
    if(measurements != nullptr and measurements->energy_variance) return measurements->energy_variance.value();
    assert(not state.active_sites.empty());
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    if constexpr(settings::debug_hamiltonian) tools::log->trace("Measuring energy variance: sites {}", state.active_sites);
    auto E  = expval_hamiltonian<Scalar>(state, model, edges);
    auto E2 = E * E;
    auto H2 = expval_hamiltonian_squared<Scalar>(state, model, edges);
    assert(std::abs(std::imag(H2)) < static_cast<RealScalar<Scalar>>(1e-10));
    RealScalar<Scalar> var = std::real(H2 - E2);
    if constexpr(settings::debug_hamiltonian) tools::log->trace("Variance |H2-E2| = |{:.16f} - {:.16f}| = {:.16f}", fp(std::real(H2)), fp(E2), fp(var));
    if(measurements != nullptr) measurements->energy_variance = var;
    return var;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                           const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg,
                                                           MeasurementsTensorsFinite<Scalar> *measurements) {
    // Here we show that the variance calculated with energy-shifted mpo's is equivalent to the usual way.
    // If mpo's are shifted:
    //      Var H = <(H-E_shf)²> - <H-E_shf>²     = <H²>  - 2<H>E_shf + E_shf² - (<H> - E_shf)²
    //                                            = H²    - 2*E*E_shf + E_shf² - E² + 2*E*E_shf - E_shf²
    //                                            = H²    - E²
    //      Note that in the last line, H²-E² is a subtraction of two large numbers --> catastrophic cancellation --> loss of precision.
    //      On the other hand Var H = <(H-E_shf)²> - energy_minus_energy_shift² = <(H-E_red)²> - ~dE², where both terms are always  << 1.
    //      The first term is computed from a double-layer of shifted mpo's.
    //      In the second term dE is usually very small, in fact identically zero immediately after an energy-reduction operation,
    //      but may grow if the optimization steps make significant progress refining E. Thus wethe first term is a good approximation to
    //      the variance by itself.
    //
    // Else, if E_shf = 0 (i.e. not shifted) we get the usual formula:
    //      Var H = <(H - 0)²> - <H - 0>² = H² - E²
    if(measurements != nullptr and measurements->energy_variance) return measurements->energy_variance.value();
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    if(not num::all_equal(model.active_sites, edges.active_sites))
        throw std::runtime_error(
            fmt::format("Could not compute energy variance: active sites are not equal: model {} | edges {}", model.active_sites, edges.active_sites));
    RealScalar<Scalar> energy = tools::finite::measure::energy_minus_energy_shift(multisite_mps, model, edges, svd_cfg, measurements);
    RealScalar<Scalar> E2     = energy * energy;

    auto t_var = tid::tic_scope("var", tid::level::highest);

    // Check if we can contract directly or if we need to use the split method
    // Normally it's only worth splitting the multisite mps when it has more than 3 sites
    constexpr auto nan = std::numeric_limits<RealScalar<Scalar>>::quiet_NaN();
    Scalar         H2  = Scalar{nan};
    if(model.active_sites.size() <= 3) {
        // Direct contraction
        const auto &mpo2 = model.template get_multisite_mpo_squared<Scalar>();
        const auto &env2 = edges.get_multisite_env_var_blk();
        if constexpr(settings::debug)
            tools::log->trace("Measuring energy variance: state dims {} | model sites {} dims {} | edges sites {} dims [L{} R{}]", multisite_mps.dimensions(),
                              model.active_sites, mpo2.dimensions(), edges.active_sites, env2.L.dimensions(), env2.R.dimensions());

        if(multisite_mps.dimension(0) != mpo2.dimension(2))
            throw std::runtime_error(fmt::format("State and model have incompatible physical dimension: state dim {} | model dim {}",
                                                 multisite_mps.dimension(0), mpo2.dimension(2)));
        H2 = tools::common::contraction::expectation_value(multisite_mps, mpo2, env2.L, env2.R);
    } else {
        // Split the multisite mps first
        const auto mpos = model.get_mpo_active();
        const auto envs = edges.get_var_active();
        if constexpr(settings::debug_hamiltonian)
            tools::log->trace("Measuring energy variance: state dims {} | sites {}", multisite_mps.dimensions(), model.active_sites);
        H2 = tools::finite::measure::expectation_value<Scalar>(multisite_mps, multisite_mps, mpos, envs, svd_cfg);
        if constexpr(settings::debug_hamiltonian) {
            const auto &mpo   = model.template get_multisite_mpo_squared<Scalar>();
            const auto &env   = edges.get_multisite_env_var_blk();
            const auto  H2dbg = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
            tools::log->trace("H2   : {:.16f}", fp(H2));
            tools::log->trace("H2dbg: {:.16f}", fp(H2dbg));
            assert(std::abs(H2 - H2dbg) < static_cast<RealScalar<Scalar>>(1e-14));
        }
    }
    assert(std::abs(std::imag(H2)) < static_cast<RealScalar<Scalar>>(1e-10));
    RealScalar<Scalar> var = std::real(H2 - E2);
    if constexpr(settings::debug_hamiltonian)
        tools::log->debug("energy_variance: Var H = H² - E² = {:.16f} - {:.16f} = {:.16f} | sites {}", fp(std::real(H2)), fp(E2), fp(var), model.active_sites);
    if(measurements != nullptr) measurements->energy_variance = var;
    return var;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_shift(const TensorsFinite<Scalar> &tensors) {
    return std::real(tensors.model->get_energy_shift_mpo());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const TensorsFinite<Scalar> &tensors) {
    tensors.assert_edges_ene();
    return energy_minus_energy_shift(*tensors.state, tensors.get_model(), tensors.get_edges(), &tensors.measurements);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const TensorsFinite<Scalar> &tensors) {
    if(not tensors.measurements.energy) {
        tensors.assert_edges_ene();
        tensors.measurements.energy = tools::finite::measure::energy(tensors.get_state(), tensors.get_model(), tensors.get_edges(), &tensors.measurements);
    }
    return tensors.measurements.energy.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const TensorsFinite<Scalar> &tensors) {
    if(not tensors.measurements.energy_variance) {
        tensors.assert_edges_var();
        tensors.measurements.energy_variance = tools::finite::measure::energy_variance(*tensors.state, *tensors.model, *tensors.edges, &tensors.measurements);
    }
    return tensors.measurements.energy_variance.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors,
                                                                     MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy_minus_energy_shift(state, tensors.get_model(), tensors.get_edges(), measurements);
}
template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors,
                                                  MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy(state, tensors.get_model(), tensors.get_edges(), measurements);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors,
                                                           MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy_variance(state, tensors.get_model(), tensors.get_edges(), measurements);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors,
                                                                     std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy_minus_energy_shift(mps, tensors.get_model(), tensors.get_edges(), svd_cfg, measurements);
}
template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg,
                                                  MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy(mps, tensors.get_model(), tensors.get_edges(), svd_cfg, measurements);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors,
                                                           std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy_variance(mps, tensors.get_model(), tensors.get_edges(), svd_cfg, measurements);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_normalized(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                             const EdgesFinite<Scalar> &edges, RealScalar<Scalar> energy_min, RealScalar<Scalar> energy_max,
                                                             MeasurementsTensorsFinite<Scalar> *measurements) {
    return (tools::finite::measure::energy(state, model, edges, measurements) - energy_min) / (energy_max - energy_min);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_normalized(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                             const EdgesFinite<Scalar> &edges, RealScalar<Scalar> energy_min, RealScalar<Scalar> energy_max,
                                                             std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements) {
    return (tools::finite::measure::energy(multisite_mps, model, edges, svd_cfg, measurements) - energy_min) / (energy_max - energy_min);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_normalized(const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> emin, RealScalar<Scalar> emax) {
    tensors.assert_edges_ene();
    return energy_normalized(*tensors.state, tensors.get_model(), tensors.get_edges(), emin, emax);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_normalized(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> emin,
                                                             RealScalar<Scalar> emax, MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy_normalized(state, tensors.get_model(), tensors.get_edges(), emin, emax, measurements);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_normalized(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> emin,
                                                             RealScalar<Scalar> emax, std::optional<svd::config> svd_cfg,
                                                             MeasurementsTensorsFinite<Scalar> *measurements) {
    return energy_normalized(mps, tensors.get_model(), tensors.get_edges(), emin, emax, svd_cfg, measurements);
}
