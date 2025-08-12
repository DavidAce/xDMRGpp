#define ANKERL_NANOBENCH_IMPLEMENT
#include "config/settings.h"
#include "config/threading.h"
#include "env/environment.h"
#include "math/tenx.h"
#include "nanobench.h"
#include "tools/common/log.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <string_view>
#include <tid/tid.h>
#include <tools/common/contraction.h>
#include <unsupported/Eigen/CXX11/Tensor>

#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis/tblis.h>
    #include <tblis/util/thread.h>
    #include <tci/tci_config.h>
#endif

using namespace tools::common::contraction;

#if defined(DMRG_ENABLE_TBLIS)
template<typename ea_type, typename eb_type, typename ec_type>
void contract_tblis(const TensorRead<ea_type> &ea, const TensorRead<eb_type> &eb, TensorWrite<ec_type> &ec, const tblis::label_vector &la,
                    const tblis::label_vector &lb, const tblis::label_vector &lc, const tblis::tblis_config_s *tblis_cntx) {
    const auto &ea_ref = static_cast<const ea_type &>(ea);
    const auto &eb_ref = static_cast<const eb_type &>(eb);
    auto       &ec_ref = static_cast<ec_type &>(ec);

    tblis::len_vector da, db, dc;
    da.assign(ea_ref.dimensions().begin(), ea_ref.dimensions().end());
    db.assign(eb_ref.dimensions().begin(), eb_ref.dimensions().end());
    dc.assign(ec_ref.dimensions().begin(), ec_ref.dimensions().end());

    auto                     ta    = tblis::varray_view<const typename ea_type::Scalar>(da, ea_ref.data(), tblis::COLUMN_MAJOR);
    auto                     tb    = tblis::varray_view<const typename eb_type::Scalar>(db, eb_ref.data(), tblis::COLUMN_MAJOR);
    auto                     tc    = tblis::varray_view<typename ec_type::Scalar>(dc, ec_ref.data(), tblis::COLUMN_MAJOR);
    typename ea_type::Scalar alpha = 1.0;
    typename ec_type::Scalar beta  = 0.0;

    tblis::tblis_tensor A_s(alpha, ta);
    tblis::tblis_tensor B_s(tb);
    tblis::tblis_tensor C_s(beta, tc);

    // tblis_tensor_mult(nullptr, tblis_cntx, &A_s, la.c_str(), &B_s, lb.c_str(), &C_s, lc.c_str());
    tblis_tensor_mult(nullptr, tblis_cntx, &A_s, la.c_str(), &B_s, lb.c_str(), &C_s, lc.c_str());
}

template<typename res_type, typename mps_type, typename env_type>
void mps_enL_tblis(TensorWrite<res_type> &res_, const TensorRead<mps_type> &mps_, const TensorRead<env_type> &enL_, std::string arch) {
    auto                         res          = tenx::asEval(res_);
    auto                         mps          = tenx::asEval(mps_);
    auto                         enL          = tenx::asEval(enL_);
    const tblis::tblis_config_s *tblis_config = tblis::tblis_get_config(arch.data());
    #if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
    tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads()));
    #endif
    if(mps.dimension(1) == enL.dimension(0)) {
        contract_tblis(mps, enL, res_, "afb", "fcd", "abcd", tblis_config);
    } else {
        contract_tblis(mps, enL, res_, "fab", "fcd", "abcd", tblis_config);
    }
}

template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
void matrix_vector_product_tblis(TensorWrite<res_type> &res_, const TensorRead<mps_type> &mps_, const TensorRead<mpo_type> &mpo_,
                                 const TensorRead<env_type> &enL_, const TensorRead<env_type> &enR_, std::string arch) {
    using T  = typename mps_type::Scalar;
    // using MT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    auto res = tenx::asEval(res_);
    auto mps = tenx::asEval(mps_);
    auto mpo = tenx::asEval(mpo_);
    auto enL = tenx::asEval(enL_);
    auto enR = tenx::asEval(enR_);

    const tblis::tblis_config_s *tblis_config = tblis::tblis_get_config(arch.data());
    #if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
    tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads()));
    #endif

    Eigen::Tensor<T, 4> mpsenvL(mps.dimension(0), mps.dimension(2), enL.dimension(1), enL.dimension(2));
    // Eigen::Tensor<T, 4> mpsenvLmpo(mps.dimension(2), enL.dimension(1), mpo.dimension(1), mpo.dimension(3));
    Eigen::Tensor<T, 4> mpsenvLmpo_alt(mpo.dimension(1), mpo.dimension(3), mps.dimension(2), enL.dimension(1));
    contract_tblis(mps, enL, mpsenvL, "afb", "fcd", "abcd", tblis_config);
    // contract_tblis(mpsenvL, mpo, mpsenvLmpo, "qijr", "rkql", "ijkl", tblis_config);
    // contract_tblis(mpsenvLmpo, enR, res, "qjri", "qkr", "ijk", tblis_config);
    contract_tblis(mpo, mpsenvL, mpsenvLmpo_alt, "qhri", "rgjq", "higj", tblis_config);
    contract_tblis(mpsenvLmpo_alt, enR, res, "higj", "gkh", "ijk", tblis_config);
}
#endif
#if defined(DMRG_ENABLE_TBLIS)
std::string get_arch() {
    #if defined(__GNUC__)
    if(__builtin_cpu_supports("x86-64-v4")) return "skx";
    if(__builtin_cpu_is("znver3") or __builtin_cpu_is("znver2") or __builtin_cpu_is("znver1")) return "zen";
    if(__builtin_cpu_supports("x86-64-v3")) return "haswell";
    #endif
    return "haswell";
}
#endif

template<typename Scalar>
void matrix_vector_product(Scalar *res_ptr, const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr,
                           std::array<long, 4> mpo_dims, const Scalar *const envL_ptr, std::array<long, 3> envL_dims, const Scalar *const envR_ptr,
                           std::array<long, 3> envR_dims) {
    //    auto t_matvec = tid::tic_token("matrix_vector_product", tid::level::extra);

    // This applies the mpo's with corresponding environments to local multisite mps
    // This is usually the operation H|psi>  or H²|psi>
    auto res  = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, mps_dims);
    auto mps  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto mpo  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
    auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envL_ptr, envL_dims);
    auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envR_ptr, envR_dims);

    if(mps.dimension(1) != envL.dimension(0)) throw except::runtime_error("Dimension mismatch mps {} and envL {}", mps.dimensions(), envL.dimensions());
    if(mps.dimension(2) != envR.dimension(0)) throw except::runtime_error("Dimension mismatch mps {} and envR {}", mps.dimensions(), envR.dimensions());
    if(mps.dimension(0) != mpo.dimension(2)) throw except::runtime_error("Dimension mismatch mps {} and mpo {}", mps.dimensions(), mpo.dimensions());
    if(envL.dimension(2) != mpo.dimension(0)) throw except::runtime_error("Dimension mismatch envL {} and mpo {}", envL.dimensions(), mpo.dimensions());
    if(envR.dimension(2) != mpo.dimension(1)) throw except::runtime_error("Dimension mismatch envR {} and mpo {}", envR.dimensions(), mpo.dimensions());

#if defined(DMRG_ENABLE_TBLIS)
    if constexpr(std::is_same_v<Scalar, fp64>) {
        // auto                         arch         = get_arch();
        const tblis::tblis_config_s *tblis_config = nullptr; // tblis::tblis_get_config(arch.data());
        // #if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
        // tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads()));
        // #endif
        if(mps.dimension(1) + 1 >= mps.dimension(2)) {
            Eigen::Tensor<Scalar, 4> mpsenvL(mps.dimension(0), mps.dimension(2), envL.dimension(1), envL.dimension(2));
            // Eigen::Tensor<Scalar, 4> mpsenvLmpo(mps.dimension(2), envL.dimension(1), mpo.dimension(1), mpo.dimension(3));
            Eigen::Tensor<Scalar, 4> mpsenvLmpo_alt(mpo.dimension(1), mpo.dimension(3), mps.dimension(2), envL.dimension(1));
            contract_tblis(mps, envL, mpsenvL, "afb", "fcd", "abcd", tblis_config);
            // contract_tblis(mpsenvL, mpo, mpsenvLmpo, "qijr", "rkql", "ijkl", tblis_config);
            // contract_tblis(mpsenvLmpo, envR, res, "qjri", "qkr", "ijk", tblis_config);
            contract_tblis(mpo, mpsenvL, mpsenvLmpo_alt, "qhri", "rgjq", "higj", tblis_config);
            contract_tblis(mpsenvLmpo_alt, envR, res, "higj", "gkh", "ijk", tblis_config);
            // contract_tblis(mpo, mpsenvL, mpsenvLmpo, "rkql", "qijr", "lkij", tblis_config);
            // contract_tblis(mpsenvLmpo, envR, res, "qirj", "rkq", "ijk", tblis_config);
        } else {
            Eigen::Tensor<Scalar, 4> mpsenvR(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
            Eigen::Tensor<Scalar, 4> mpsenvRmpo(mps.dimension(1), envR.dimension(1), mpo.dimension(0), mpo.dimension(3));
            contract_tblis(mps, envR, mpsenvR, "abf", "fcd", "abcd", tblis_config);
            contract_tblis(mpsenvR, mpo, mpsenvRmpo, "qijk", "rkql", "ijrl", tblis_config);
            contract_tblis(mpsenvRmpo, envL, res, "qkri", "qjr", "ijk", tblis_config);
            // contract_tblis(mpo, mpsenvR, mpsenvRmpo_alt, "qklr", "qijr", "lkij", tblis_config);
            // contract_tblis(mpsenvRmpo_alt, envL, res, "qirk", "rjq", "ijk", tblis_config);
        }
    } else {
        auto &threads = tenx::threads::get();
        if(mps.dimension(1) >= mps.dimension(2)) {
            res.device(*threads->dev) = mps.contract(envL, tenx::idx({1}, {0}))
                                            .contract(mpo, tenx::idx({3, 0}, {0, 2}))
                                            .contract(envR, tenx::idx({0, 2}, {0, 2}))
                                            .shuffle(tenx::array3{1, 0, 2});
        } else {
            res.device(*threads->dev) = mps.contract(envR, tenx::idx({2}, {0}))
                                            .contract(mpo, tenx::idx({3, 0}, {1, 2}))
                                            .contract(envL, tenx::idx({0, 2}, {0, 2}))
                                            .shuffle(tenx::array3{1, 2, 0});
        }
    }
#else
    auto &threads = tenx::threads::get();
    if(mps.dimension(1) >= mps.dimension(2)) {
        res.device(*threads->dev) = mps.contract(envL, tenx::idx({1}, {0}))
                                        .contract(mpo, tenx::idx({3, 0}, {0, 2}))
                                        .contract(envR, tenx::idx({0, 2}, {0, 2}))
                                        .shuffle(tenx::array3{1, 0, 2});
    } else {
        res.device(*threads->dev) = mps.contract(envR, tenx::idx({2}, {0}))
                                        .contract(mpo, tenx::idx({3, 0}, {1, 2}))
                                        .contract(envL, tenx::idx({0, 2}, {0, 2}))
                                        .shuffle(tenx::array3{1, 2, 0});
    }
#endif
}

template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
void matrix_vector_product(TensorWrite<res_type> &res, const TensorRead<mps_type> &mps, TensorRead<mpo_type> &mpo, const TensorRead<env_type> &envL,
                           const TensorRead<env_type> &envR) {
    static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3 passed to calculation of matrix_vector_product");
    static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of matrix_vector_product");
    static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of matrix_vector_product");
    auto &res_ref   = static_cast<res_type &>(res);
    auto  mps_eval  = tenx::asEval(mps);
    auto  mpo_eval  = tenx::asEval(mpo);
    auto  envL_eval = tenx::asEval(envL);
    auto  envR_eval = tenx::asEval(envR);
    matrix_vector_produc(res_ref.data(), mps_eval.data(), mps_eval.dimensions(), mpo_eval.data(), envL_eval.data(), envL_eval.dimensions(), envR_eval.data(),
                         envR_eval.dimensions());
}

template<typename Scalar, typename mpo_type>
void matrix_vector_product_custom(Scalar *res_ptr, const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const std::vector<mpo_type> &mpos_shf,
                                  const Scalar *const envL_ptr, std::array<long, 3> envL_dims, const Scalar *const envR_ptr, std::array<long, 3> envR_dims) {
    // Make sure the mpos are pre-shuffled. If not, shuffle and call this function again
    bool is_shuffled = mpos_shf.front().dimension(2) == envL_dims[2] and mpos_shf.back().dimension(3) == envR_dims[2];
    if(not is_shuffled) {
        // mpos_shf are not actually shuffled. Let's shuffle.
        std::vector<Eigen::Tensor<Scalar, 4>> mpos_really_shuffled;
        for(const auto &mpo : mpos_shf) { mpos_really_shuffled.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1})); }
        return matrix_vector_product_custom(res_ptr, mps_ptr, mps_dims, mpos_really_shuffled, envL_ptr, envL_dims, envR_ptr, envR_dims);
    }

    auto &threads = tenx::threads::get();
    auto  mps_out = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, mps_dims);
    auto  mps_in  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto  envL    = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envL_ptr, envL_dims);
    auto  envR    = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envR_ptr, envR_dims);

    if(mps_in.dimension(1) != envL.dimension(0)) throw except::runtime_error("Dimension mismatch mps {} and envL {}", mps_in.dimensions(), envL.dimensions());
    if(mps_in.dimension(2) != envR.dimension(0)) throw except::runtime_error("Dimension mismatch mps {} and envR {}", mps_in.dimensions(), envR.dimensions());

    auto L = mpos_shf.size();

    auto mpodimprod = [&](size_t fr, size_t to) -> long {
        long prod = 1;
        if(fr == -1ul) fr = 0;
        if(to == 0 or to == -1ul) return prod;
        for(size_t idx = fr; idx < to; ++idx) {
            if(idx >= mpos_shf.size()) break;
            prod *= mpos_shf[idx].dimension(1);
        }
        return prod;
    };

// At best, the number of operations for contracting left-to-right and right-to-left are equal.
// Since the site indices are contracted left to right, we do not need any shuffles in this direction.

// Contract left to right
#if defined(DMRG_ENABLE_TBLIS)
    const tblis::tblis_config_s *tblis_config = tblis::tblis_get_config(get_arch().data());
    #if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
    tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads()));
    #endif
#endif
    auto d0       = mpodimprod(0, 1); // Split 0 --> 0,1
    auto d1       = mpodimprod(1, L); // Split 0 --> 0,1
    auto d2       = mps_in.dimension(2);
    auto d3       = envL.dimension(1);
    auto d4       = envL.dimension(2);
    auto d5       = 1l; // A new dummy index
    auto new_shp6 = tenx::array6{d0, d1, d2, d3, d4, d5};
    auto mps_tmp1 = Eigen::Tensor<Scalar, 6>();
    auto mps_tmp2 = Eigen::Tensor<Scalar, 6>();
    // auto t_mv     = tid::tic_token("mv");
    // t_mv->start_lap();
    mps_tmp1.resize(tenx::array6{d0, d1, d2, d3, d5, d4});
    // tools::log->info("mps_tmp1.resize({} = {}) | t = {:.3e} | tot = {:.3e}", new_shp6, Eigen::internal::array_prod(new_shp6), t_mv->restart_lap(),
    // t_mv->get_last_interval());
#if defined(DMRG_ENABLE_TBLIS)
    if constexpr(std::is_same_v<Scalar, fp64>) {
        auto mps_tmp1_map4 = Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>(mps_tmp1.data(), std::array{d0 * d1, d2, d3, d4 * d5});
        contract_tblis(mps_in, envL, mps_tmp1_map4, "afb", "fcd", "abcd", tblis_config);
    } else
#endif
    {
        mps_tmp1.device(*threads->dev) = mps_in.contract(envL, tenx::idx({1}, {0})).reshape(new_shp6).shuffle(tenx::array6{0, 1, 2, 3, 5, 4});
    }
    // tools::log->info("prepended envL | t = {:.3e} | tot = {:.3e}", t_mv->restart_lap(), t_mv->get_last_interval());
    for(size_t idx = 0; idx < L; ++idx) {
        const auto &mpo = mpos_shf[idx];
        // Set up the dimensions for the reshape after the contraction
        d0 = mpodimprod(idx + 1, idx + 2); // if idx == k, this has the mpo at idx == k+1
        d1 = mpodimprod(idx + 2, L);       // if idx == 0,  this has the mpos at idx == k+2...L-1
        d2 = mps_tmp1.dimension(2);
        d3 = mps_tmp1.dimension(3);
        d4 = mpodimprod(0, idx + 1); // if idx == 0, this has the mpos at idx == 0...k (i.e. including the one from the current iteration)
        d5 = mpo.dimension(3);       // The virtual bond of the current mpo
#if defined(DMRG_ENABLE_TBLIS)
        if constexpr(std::is_same_v<Scalar, fp64>) {
            auto md  = mps_tmp1.dimensions();
            new_shp6 = tenx::array6{d0, d1, d2, d3, d4, d5};
            mps_tmp2.resize(new_shp6);
            auto map_shp6     = tenx::array6{md[1], md[2], md[3], md[4], mpo.dimension(1), mpo.dimension(3)};
            auto mps_tmp2_map = Eigen::TensorMap<Eigen::Tensor<Scalar, 6>>(mps_tmp2.data(), map_shp6);
            contract_tblis(mps_tmp1, mpo, mps_tmp2_map, "qbcder", "qfrg", "bcdefg", tblis_config);
            // tools::log->info("contracted {} | t = {:.3e} | tot = {:.3e} ", idx, t_mv->restart_lap(), t_mv->get_last_interval());
            mps_tmp1 = std::move(mps_tmp2);
        } else
#endif
        {
            new_shp6 = tenx::array6{d0, d1, d2, d3, d4, d5};
            mps_tmp2.resize(new_shp6);
            mps_tmp2.device(*threads->dev) = mps_tmp1.contract(mpo, tenx::idx({0, 5}, {0, 2})).reshape(new_shp6);
            mps_tmp1                       = std::move(mps_tmp2);
        }
    }
    d0 = mps_tmp1.dimension(0) * mps_tmp1.dimension(1) * mps_tmp1.dimension(2); // idx 0 and 1 should have dim == 1
    d1 = mps_tmp1.dimension(3);
    d2 = mps_tmp1.dimension(4);
    d3 = mps_tmp1.dimension(5);
#if defined(DMRG_ENABLE_TBLIS)
    if constexpr(std::is_same_v<Scalar, fp64>) {
        auto mps_tmp1_map4 = Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>(mps_tmp1.data(), std::array{d0, d1, d2, d3});
        contract_tblis(mps_tmp1_map4, envR, mps_out, "qjir", "qkr", "ijk", tblis_config);
    } else
#endif
    {
        mps_out.device(*threads->dev) = mps_tmp1.reshape(tenx::array4{d0, d1, d2, d3}).contract(envR, tenx::idx({0, 3}, {0, 2})).shuffle(tenx::array3{1, 0, 2});
    }
    // tools::log->info("appended envR | t = {:.3e} | tot = {:.3e} ", t_mv->restart_lap(), t_mv->get_last_interval());
}

template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
void matrix_vector_product_custom(TensorWrite<res_type> &res, const TensorRead<mps_type> &mps, const std::vector<mpo_type> &mpos_shf,
                                  const TensorRead<env_type> &envL, const TensorRead<env_type> &envR) {
    static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3 passed to calculation of matrix_vector_product");
    static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of matrix_vector_product");
    static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of matrix_vector_product");
    auto &res_ref   = static_cast<res_type &>(res);
    auto  mps_eval  = tenx::asEval(mps);
    auto  envL_eval = tenx::asEval(envL);
    auto  envR_eval = tenx::asEval(envR);
    matrix_vector_product_custom(res_ref.data(), mps_eval.data(), mps_eval.dimensions(), mpos_shf, envL_eval.data(), envL_eval.dimensions(), envR_eval.data(),
                                 envR_eval.dimensions());
}

int main() {
    tools::log = tools::Logger::setLogger("matvec", 2, true);
    fmt::print("Compiler flags {}\n", env::build::compiler_flags);
    settings::threading::num_threads = 8;
    settings::configure_threads();
    using fp64 = double;
    // using cplx = std::complex<double>;
    long sites = 2;
    long d     = 2 << (sites - 1);
    // long d     = 8 << (sites - 1);
    long m    = 12;
    long chiL = 1024;
    long chiR = 512;

    tools::log->info("__builtin_cpu_supports(avx512f)    : {}", __builtin_cpu_supports("avx512f"));
    tools::log->info("__builtin_cpu_supports(x86-64-v4)  : {}", __builtin_cpu_supports("x86-64-v4"));
    tools::log->info("__builtin_cpu_supports(x86-64-v3)  : {}", __builtin_cpu_supports("x86-64-v3"));
    tools::log->info("__builtin_cpu_supports(x86-64-v2)  : {}", __builtin_cpu_supports("x86-64-v2"));
    tools::log->info("__builtin_cpu_supports(x86-64)     : {}", __builtin_cpu_supports("x86-64"));
    tools::log->info("__builtin_cpu_is(znver1)           : {}", __builtin_cpu_is("znver1"));
    tools::log->info("__builtin_cpu_is(znver2)           : {}", __builtin_cpu_is("znver2"));
    tools::log->info("__builtin_cpu_is(znver3)           : {}", __builtin_cpu_is("znver3"));
    tools::log->info("__builtin_cpu_is(znver4)           : {}", __builtin_cpu_is("znver4"));
    // tools::log->info("__builtin_cpu_is(znver5)            : {}", __builtin_cpu_is("znver5"));

    auto mps_enL = Eigen::Tensor<fp64, 4>(d, chiR, chiR, m);
    auto res     = Eigen::Tensor<fp64, 3>(d, chiL, chiR);
    auto mps     = Eigen::Tensor<fp64, 3>(d, chiL, chiR);
    auto mps_shf = Eigen::Tensor<fp64, 3>(chiR, d, chiL);
    auto mpo     = Eigen::Tensor<fp64, 4>(m, m, d, d);
    auto enL     = Eigen::Tensor<fp64, 3>(chiL, chiL, m);
    auto enR     = Eigen::Tensor<fp64, 3>(chiR, chiR, m);
    auto mpo_shf = Eigen::Tensor<fp64, 4>(2, 2, m, m);
    // auto mpo_shf = Eigen::Tensor<fp64, 4>(d, d, m, m);
    mps.setRandom();
    mps_shf.setRandom();
    // mpo.setRandom();
    enL.setRandom();
    enR.setRandom();
    mpo_shf.setRandom();
    tools::log->info("mps dims: {} | size: {}", mps.dimensions(), mps.size());
    // tools::log->info("mpo dims: {}", mpo.dimensions());
    auto mpos_shf = std::vector<Eigen::Tensor<fp64, 4>>(safe_cast<size_t>(sites), mpo_shf);

    {
        ankerl::nanobench::Bench().warmup(4).epochs(4).minEpochIterations(10).run("tools::common::contraction::matrix_vector_product", [&] {
            auto val = tools::common::contraction::expectation_value(mps, mpo, enL, enR);
            ankerl::nanobench::doNotOptimizeAway(val);
        });
        // ankerl::nanobench::Bench().warmup(4).epochs(4).minEpochIterations(10).run("matrix_vector_product_tblis", [&] {
        //     matrix_vector_product_tblis(res, mps, mpo, enL, enR, get_arch());
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
        // ankerl::nanobench::Bench().warmup(4).epochs(4).minEpochIterations(10).run("matrix_vector_product", [&] {
        //     matrix_vector_product_tblis(res, mps, mpo, enL, enR, get_arch());
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
        // ankerl::nanobench::Bench().warmup(4).epochs(4).minEpochIterations(10).run("matrix_vector_product_custom", [&] {
        //     matrix_vector_product_custom(res, mps, mpos_shf, enL, enR);
        //     // matrix_vector_product_custom(res, mps, mpos_shf, enL, enR);
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
        // ankerl::nanobench::Bench().warmup(5).epochs(5).minEpochIterations(10).run("mps enL custom", [&] {
        //     mps_enL_custom(mps_enL, mps, enL);
        //     ankerl::nanobench::doNotOptimizeAway(mps_enL);
        // });
        // ankerl::nanobench::Bench().warmup(5).epochs(5).minEpochIterations(10).run("mps enL eigen", [&] {
        //     mps_enL_eigen(mps_enL, mps, enL);
        //     ankerl::nanobench::doNotOptimizeAway(mps_enL);
        // });

#if defined(DMRG_ENABLE_TBLIS)
        // ankerl::nanobench::Bench().warmup(5).epochs(20).minEpochIterations(10).run("tblis zen", [&] {
        //     matrix_vector_product_tblis(res, mps, mpo, enL, enR, "zen");
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
        // ankerl::nanobench::Bench().warmup(5).epochs(5).minEpochIterations(100).run("tblis auto", [&] {
        //     tools::common::contraction::matrix_vector_product(res, mps, mpo, enL, enR);
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
        // ankerl::nanobench::Bench().warmup(5).epochs(5).minEpochIterations(10).run("mps enL tblis haswell", [&] {
        //     mps_enL_tblis(mps_enL, mps, enL, "haswell");
        //     ankerl::nanobench::doNotOptimizeAway(mps_enL);
        // });
        // ankerl::nanobench::Bench().warmup(5).epochs(5).minEpochIterations(10).run("mps enL tblis zen", [&] {
        //     mps_enL_tblis(mps_enL, mps, enL, "zen");
        //     ankerl::nanobench::doNotOptimizeAway(mps_enL);
        // });
        // ankerl::nanobench::Bench().warmup(5).epochs(5).minEpochIterations(10).run("mps enL tblis zen", [&] {
        //     mps_enL_tblis(mps_enL, mps_shf, enL, "zen");
        //     ankerl::nanobench::doNotOptimizeAway(mps_enL);
        // });

        // ankerl::nanobench::Bench().warmup(5).epochs(40).run("tblis haswell", [&] {
        //     matrix_vector_product_tblis(res, mps, mpo, enL, enR, "haswell");
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });

        // ankerl::nanobench::Bench().warmup(5).epochs(5).minEpochIterations(10).run("tblis skx1", [&] {
        //     matrix_vector_product_tblis(res, mps, mpo, enL, enR, "intel");
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
        // ankerl::nanobench::Bench().warmup(5).epochs(40).run("tblis skx2", [&] {
        //     matrix_vector_product_tblis(res, mps, mpo, enL, enR, "skx2");
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
#endif
    }

    {
        // auto res = Eigen::Tensor<fp64, 3>(chiL, d, chiR);
        // auto mps = Eigen::Tensor<fp64, 3>(chiL, d, chiR);
        // auto mpo = Eigen::Tensor<fp64, 4>(m, m, d, d);
        // auto enL = Eigen::Tensor<fp64, 3>(chiL, m, chiL);
        // auto enR = Eigen::Tensor<fp64, 3>(chiR, m, chiR);
        //
        // mps.setRandom();
        // mpo.setRandom();
        // enL.setRandom();
        // enR.setRandom();

        // ankerl::nanobench::Bench().warmup(5).epochs(40).run("matvec eigen2", [&] {
        //     matrix_vector_product_eigen2(res, mps, mpo, enL, enR);
        //     ankerl::nanobench::doNotOptimizeAway(res);
        //     ankerl::nanobench::doNotOptimizeAway(mps);
        // });
    }
}
