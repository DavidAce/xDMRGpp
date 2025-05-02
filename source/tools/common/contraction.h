#pragma once

#include "math/tenx/fwd_decl.h"
// Eigen goes first
#include "math/float.h"
#include "math/tenx/eval.h"
#include "math/tenx/threads.h"
#include <complex>
#include "contraction/InvMatVecCfg.h"

template<typename Scalar> struct InvMatVecCfg;

namespace tools::common::contraction {
    template<typename T>
    using TensorWrite = Eigen::TensorBase<T, Eigen::WriteAccessors>;
    template<typename T>
    using TensorRead = Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>;

    class dimlist : public std::vector<long> {
        public:
        using base_t = std::vector<long>;
        using base_t::base_t;
        template<auto rank>
        dimlist(std::array<long, rank> arr) : base_t(arr.begin(), arr.end()) {}
    };

    std::string get_tblis_arch();

    template<typename Scalar>
    void contract_tblis(const Scalar *aptr, dimlist adim,         //
                        const Scalar *bptr, dimlist bdim,         //
                        Scalar *cptr, dimlist cdim,               //
                        std::string_view la, std::string_view lb, //
                        std::string_view lc, const void *tblis_config_ptr);

    // template<typename A_t, typename B_t, typename C_t>
    // void contract_tblis(const TensorRead<A_t> &ea, const TensorRead<B_t> &eb, TensorWrite<C_t> &ec, std::string_view la, std::string_view lb,
    //                     std::string_view lc, const void *tblis_config_ptr);

    /* clang-format off */
    template<typename Scalar>
    Scalar expectation_value(const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                             const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                             const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                             const Scalar * const envR_ptr, std::array<long,3> envR_dims);

    template<typename Scalar>
    Scalar expectation_value(const Scalar * const bra_ptr, std::array<long,3> bra_dims,
                             const Scalar * const ket_ptr, std::array<long,3> ket_dims,
                             const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                             const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                             const Scalar * const envR_ptr, std::array<long,3> envR_dims);

    template<typename Scalar>
    void matrix_vector_product(Scalar * res_ptr,
                               const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                               const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                               const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                               const Scalar * const envR_ptr, std::array<long,3> envR_dims);


    template<typename Scalar, typename mpo_type>
    void matrix_vector_product(Scalar * res_ptr,
                               const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                               const std::vector<mpo_type> & mpos_shf,
                               const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                               const Scalar * const envR_ptr, std::array<long,3> envR_dims);


    template<typename Scalar>
    void matrix_inverse_vector_product(      Scalar * res_ptr,
                                       const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                                       const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                                       const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                                       const Scalar * const envR_ptr, std::array<long,3> envR_dims,
                                       InvMatVecCfg<Scalar> cfg = InvMatVecCfg<Scalar>());

    template<typename mps_type, typename mpo_type, typename env_type>
    auto expectation_value(const TensorRead<mps_type> & mps,
                           const TensorRead<mpo_type> & mpo,
                           const TensorRead<env_type> & envL,
                           const TensorRead<env_type> & envR){
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of expectation_value");
        static_assert(mpo_type::NumIndices == 4 and "Wrong mpo tensor rank != 4 passed to calculation of expectation_value");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of expectation_value");
        auto  mps_eval = tenx::asEval(mps);
        auto  mpo_eval = tenx::asEval(mpo);
        auto  envL_eval = tenx::asEval(envL);
        auto  envR_eval = tenx::asEval(envR);


        return expectation_value(
            mps_eval.data(), mps_eval.dimensions(),
            mpo_eval.data(), mpo_eval.dimensions(),
            envL_eval.data(), envL_eval.dimensions(),
            envR_eval.data(), envR_eval.dimensions());
    }

    template<typename bra_type,typename ket_type,  typename mpo_type, typename env_type>
    auto expectation_value(const TensorRead<bra_type> & bra,
                           const TensorRead<ket_type> & ket,
                           const TensorRead<mpo_type> & mpo,
                           const TensorRead<env_type> & envL,
                           const TensorRead<env_type> & envR){
        static_assert(bra_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of expectation_value");
        static_assert(ket_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of expectation_value");
        static_assert(mpo_type::NumIndices == 4 and "Wrong mpo tensor rank != 4 passed to calculation of expectation_value");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of expectation_value");
        auto  bra_eval = tenx::asEval(bra);
        auto  ket_eval = tenx::asEval(ket);
        auto  mpo_eval = tenx::asEval(mpo);
        auto  envL_eval = tenx::asEval(envL);
        auto  envR_eval = tenx::asEval(envR);


        return expectation_value(
            bra_eval.data(), bra_eval.dimensions(),
            ket_eval.data(), ket_eval.dimensions(),
            mpo_eval.data(), mpo_eval.dimensions(),
            envL_eval.data(), envL_eval.dimensions(),
            envR_eval.data(), envR_eval.dimensions());
    }


    template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
    void matrix_vector_product(TensorWrite<res_type> &res,
                               const TensorRead<mps_type> & mps,
                               const TensorRead<mpo_type> & mpo,
                               const TensorRead<env_type> & envL,
                               const TensorRead<env_type> & envR){
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mpo_type::NumIndices == 4 and "Wrong mpo tensor rank != 4 passed to calculation of matrix_vector_product");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of matrix_vector_product");
        auto & res_ref = static_cast<res_type&>(res);
        auto  mps_eval = tenx::asEval(mps);
        auto  mpo_eval = tenx::asEval(mpo);
        auto  envL_eval = tenx::asEval(envL);
        auto  envR_eval = tenx::asEval(envR);
        matrix_vector_product(
            res_ref.data(),
            mps_eval.data(), mps_eval.dimensions(),
            mpo_eval.data(), mpo_eval.dimensions(),
            envL_eval.data(), envL_eval.dimensions(),
            envR_eval.data(), envR_eval.dimensions());
    }

    template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
    void matrix_vector_product(TensorWrite<res_type> &res,
                               const TensorRead<mps_type> & mps,
                               const std::vector<mpo_type> & mpos_shf,
                               const TensorRead<env_type> & envL,
                               const TensorRead<env_type> & envR){
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of matrix_vector_product");
        auto & res_ref = static_cast<res_type&>(res);
        auto  mps_eval = tenx::asEval(mps);
        // auto  mpo_eval = tenx::asEval(mpo);
        auto  envL_eval = tenx::asEval(envL);
        auto  envR_eval = tenx::asEval(envR);
        matrix_vector_product(
            res_ref.data(),
            mps_eval.data(), mps_eval.dimensions(),
            mpos_shf,
            envL_eval.data(), envL_eval.dimensions(),
            envR_eval.data(), envR_eval.dimensions());
    }


    template<typename mps_type, typename mpo_type, typename env_type>
    mps_type matrix_vector_product(const mps_type &mps, const mpo_type &mpo, const env_type &envL, const env_type &envR) {
        using Scalar = typename mps_type::Scalar;
        Eigen::Tensor<Scalar,3> result(mps.dimensions());
        matrix_vector_product(result, mps, mpo, envL, envR);
        return result;
    }


    template<typename res_type, typename mps_type, typename mpo_type, typename env_type>
    void matrix_inverse_vector_product(
        TensorWrite<res_type> &res,
        const TensorRead<mps_type> & mps,
        const TensorRead<mpo_type> & mpo,
        const TensorRead<env_type> & envL,
        const TensorRead<env_type> & envR,
        InvMatVecCfg<typename res_type::Scalar> cfg = InvMatVecCfg<typename res_type::Scalar>()){
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3 passed to calculation of matrix_vector_product");
        static_assert(mpo_type::NumIndices == 4 and "Wrong mpo tensor rank != 4 passed to calculation of matrix_vector_product");
        static_assert(env_type::NumIndices == 3 and "Wrong env tensor rank != 3 passed to calculation of matrix_vector_product");
        auto & res_ref = static_cast<res_type&>(res);
        auto  mps_eval = tenx::asEval(mps);
        auto  mpo_eval = tenx::asEval(mpo);
        auto  envL_eval = tenx::asEval(envL);
        auto  envR_eval = tenx::asEval(envR);
        matrix_inverse_vector_product(
            res_ref.data(),
            mps_eval.data(), mps_eval.dimensions(),
            mpo_eval.data(), mpo_eval.dimensions(),
            envL_eval.data(), envL_eval.dimensions(),
            envR_eval.data(), envR_eval.dimensions(),
            cfg);
    }

    /* clang-format on */

    template<typename Scalar>
    void contract_mps_bnd(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const bnd_ptr,
                          std::array<long, 1> bnd_dims);

    template<typename Scalar>
    void contract_bnd_mps(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const bnd_ptr, std::array<long, 1> bnd_dims, const Scalar *const mps_ptr,
                          std::array<long, 3> mps_dims);

    template<typename Scalar>
    void contract_mps_mps(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const mpsL_ptr, std::array<long, 3> mpsL_dims,
                          const Scalar *const mpsR_ptr, std::array<long, 3> mpsR_dims);

    template<typename Scalar>
    void contract_env_mps_mpo(Scalar *res_ptr, std::array<long, 2> res_dims, const Scalar *const env_ptr, std::array<long, 2> env_dims,
                              const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr, std::array<long, 2> mpo_dims);
    template<typename Scalar>
    void contract_env_mps_mpo(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const env_ptr, std::array<long, 3> env_dims,
                              const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims);

    template<typename Scalar>
    void contract_mps_mpo_env(Scalar *res_ptr, std::array<long, 2> res_dims, const Scalar *const env_ptr, std::array<long, 2> env_dims,
                              const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr, std::array<long, 2> mpo_dims);
    template<typename Scalar>
    void contract_mps_mpo_env(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *const env_ptr, std::array<long, 3> env_dims,
                              const Scalar *const mps_ptr, std::array<long, 3> mps_dims, const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims);

    template<typename res_type, typename mps_type, typename bnd_type>
    void contract_mps_bnd(TensorWrite<res_type> &res, const TensorRead<mps_type> &mps, const TensorRead<bnd_type> &bnd) {
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(bnd_type::NumIndices == 1 and "Wrong bnd tensor rank != 1");
        auto &res_ref  = static_cast<res_type &>(res);
        auto  mps_eval = tenx::asEval(mps);
        auto  bnd_eval = tenx::asEval(bnd);

        res_ref.resize(mps_eval.dimensions());
        contract_mps_bnd(res_ref.data(), res_ref.dimensions(), mps_eval.data(), mps_eval.dimensions(), bnd_eval.data(), bnd_eval.dimensions());
    }

    template<typename res_type, typename bnd_type, typename mps_type>
    void contract_bnd_mps(TensorWrite<res_type> &res, const TensorRead<bnd_type> &bnd, const TensorRead<mps_type> &mps) {
        static_assert(res_type::NumIndices == 3 and "Wrong res tensor rank != 3");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(bnd_type::NumIndices == 1 and "Wrong bnd tensor rank != 1");
        auto &res_ref  = static_cast<res_type &>(res);
        auto  mps_eval = tenx::asEval(mps);
        auto  bnd_eval = tenx::asEval(bnd);
        res_ref.resize(mps_eval.dimensions());
        contract_bnd_mps(res_ref.data(), res_ref.dimensions(), bnd_eval.data(), bnd_eval.dimensions(), mps_eval.data(), mps_eval.dimensions());
    }

    template<typename mps_type, typename bnd_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 3> contract_mps_bnd(const TensorRead<mps_type> &mps, const TensorRead<bnd_type> &bnd) {
        Eigen::Tensor<typename mps_type::Scalar, 3> res;
        contract_mps_bnd(res, mps, bnd);
        return res;
    }

    template<typename bnd_type, typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 3> contract_bnd_mps(const TensorRead<bnd_type> &bnd, const TensorRead<mps_type> &mps) {
        Eigen::Tensor<typename mps_type::Scalar, 3> res;
        contract_bnd_mps(res, bnd, mps);
        return res;
    }

    template<typename mps_type>
    void contract_mps_mps(TensorWrite<mps_type> &res, const TensorRead<mps_type> &mpsL, const TensorRead<mps_type> &mpsR) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        auto               &res_ref   = static_cast<mps_type &>(res);
        auto                mpsL_eval = tenx::asEval(mpsL);
        auto                mpsR_eval = tenx::asEval(mpsR);
        long                d0        = mpsL_eval.dimension(0) * mpsR_eval.dimension(0);
        long                d1        = mpsL_eval.dimension(1);
        long                d2        = mpsR_eval.dimension(2);
        std::array<long, 3> res_dims  = {d0, d1, d2};
        res_ref.resize(res_dims);
        contract_mps_mps(res_ref.data(), res_ref.dimensions(), mpsL_eval.data(), mpsL_eval.dimensions(), mpsR_eval.data(), mpsR_eval.dimensions());
    }

    template<typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 3> contract_mps_mps(const TensorRead<mps_type> &mpsL, const TensorRead<mps_type> &mpsR) {
        Eigen::Tensor<typename mps_type::Scalar, 3> res;
        contract_mps_mps(res, mpsL, mpsR);
        return res;
    }

    template<typename Scalar>
    Scalar contract_mps_mps_overlap(const Scalar *const mps1_ptr, std::array<long, 3> mps1_dims, const Scalar *const mps2_ptr, std::array<long, 3> mps2_dims);

    template<typename mps_type1, typename mps_type2>
    auto contract_mps_overlap(const TensorRead<mps_type1> &mps1, const TensorRead<mps_type2> &mps2) {
        static_assert(mps_type1::NumIndices == 3 and "Wrong mps1 tensor rank != 3");
        static_assert(mps_type2::NumIndices == 3 and "Wrong mps2 tensor rank != 3");
        auto mps1_eval = tenx::asEval(mps1);
        auto mps2_eval = tenx::asEval(mps2);
        return contract_mps_mps_overlap(mps1_eval.data(), mps1_eval.dimensions(), mps2_eval.data(), mps2_eval.dimensions());
    }

    template<typename mps_type>
    auto contract_mps_norm(const TensorRead<mps_type> &mps) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        auto mps_eval = tenx::asEval(mps);
        return contract_mps_mps_overlap(mps_eval.data(), mps_eval.dimensions(), mps_eval.data(), mps_eval.dimensions());
    }

    template<typename Scalar>
    void contract_mps_mps_partial(Scalar *res_ptr, std::array<long, 2> res_dims,               //
                                  const Scalar *const mps1_ptr, std::array<long, 3> mps1_dims, //
                                  const Scalar *const mps2_ptr, std::array<long, 3> mps2_dims, //
                                  std::array<long, 2> idx);

    template<std::array<long, 2> idx, typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 2> contract_mps_mps_partial(const TensorRead<mps_type> &mps1, const TensorRead<mps_type> &mps2) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(idx == std::array{0l, 1l} or idx == std::array{0l, 2l} or idx == std::array{1l, 2l});
        auto                mps1_eval = tenx::asEval(mps1);
        auto                mps2_eval = tenx::asEval(mps2);
        std::array<long, 2> dims      = {};
        if constexpr(idx == std::array{0l, 1l})
            dims = {mps1_eval.dimension(2), mps2_eval.dimension(2)};
        else if constexpr(idx == std::array{0l, 2l})
            dims = {mps1_eval.dimension(1), mps2_eval.dimension(1)};
        else if constexpr(idx == std::array{1l, 2l})
            dims = {mps1_eval.dimension(0), mps2_eval.dimension(0)};

        Eigen::Tensor<typename mps_type::Scalar, 2> res(dims);
        contract_mps_mps_partial(res.data(), res.dimensions(), mps1_eval.data(), mps1_eval.dimensions(), mps2_eval.data(), mps2_eval.dimensions(), idx);
        return res;
    }

    template<std::array<long, 2> idx, typename mps_type>
    [[nodiscard]] Eigen::Tensor<typename mps_type::Scalar, 2> contract_mps_partial(const TensorRead<mps_type> &mps) {
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        static_assert(idx == std::array{0l, 1l} or idx == std::array{0l, 2l} or idx == std::array{1l, 2l});
        auto                mps_eval = tenx::asEval(mps);
        std::array<long, 2> dims     = {};
        if constexpr(idx == std::array{0l, 1l})
            dims = {mps_eval.dimension(2), mps_eval.dimension(2)};
        else if constexpr(idx == std::array{0l, 2l})
            dims = {mps_eval.dimension(1), mps_eval.dimension(1)};
        else if constexpr(idx == std::array{1l, 2l})
            dims = {mps_eval.dimension(0), mps_eval.dimension(0)};
        Eigen::Tensor<typename mps_type::Scalar, 2> res(dims);
        contract_mps_mps_partial(res.data(), res.dimensions(), mps_eval.data(), mps_eval.dimensions(), mps_eval.data(), mps_eval.dimensions(), idx);
        return res;
    }

    template<typename res_type, typename env_type, typename mps_type, typename mpo_type>
    void contract_env_mps_mpo(TensorWrite<res_type> &res, const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo) {
        static_assert((res_type::NumIndices == 2 or res_type::NumIndices == 3) and "Wrong res tensor rank != 2 or 3");
        static_assert((env_type::NumIndices == 2 or env_type::NumIndices == 3) and "Wrong env tensor rank != 2 or 3");
        static_assert((mpo_type::NumIndices == 2 or mpo_type::NumIndices == 4) and "Wrong mpo tensor rank != 2 or 4");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        /* clang-format off */
        auto &res_ref = static_cast<res_type &>(res);
        auto env_eval = tenx::asEval(env);
        auto mps_eval = tenx::asEval(mps);
        auto mpo_eval = tenx::asEval(mpo);
        if constexpr(env_type::NumIndices == 2){
            res_ref.resize(mps_eval.dimension(2), mps_eval.dimension(2));
            contract_env_mps_mpo(res_ref.data(), res_ref.dimensions(),
                                 env_eval.data(), env_eval.dimensions(),
                                 mps_eval.data(), mps_eval.dimensions(),
                                 mpo_eval.data(), mpo_eval.dimensions());
        }
        else {
            res_ref.resize(mps_eval.dimension(2), mps_eval.dimension(2), mpo_eval.dimension(1));
            contract_env_mps_mpo(res_ref.data(), res_ref.dimensions(),
                                 env_eval.data(), env_eval.dimensions(),
                                 mps_eval.data(), mps_eval.dimensions(),
                                 mpo_eval.data(), mpo_eval.dimensions());
        }
        /* clang-format on */
    }
    template<typename res_type, typename env_type, typename mps_type, typename mpo_type>
    void contract_mps_mpo_env(TensorWrite<res_type> &res, const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo) {
        static_assert((res_type::NumIndices == 2 or res_type::NumIndices == 3) and "Wrong res tensor rank != 2 or 3");
        static_assert((env_type::NumIndices == 2 or env_type::NumIndices == 3) and "Wrong env tensor rank != 2 or 3");
        static_assert((mpo_type::NumIndices == 2 or mpo_type::NumIndices == 4) and "Wrong mpo tensor rank != 2 or 4");
        static_assert(mps_type::NumIndices == 3 and "Wrong mps tensor rank != 3");
        /* clang-format off */
        auto &res_ref = static_cast<res_type &>(res);
        auto env_eval = tenx::asEval(env);
        auto mps_eval = tenx::asEval(mps);
        auto mpo_eval = tenx::asEval(mpo);


        if constexpr(env_type::NumIndices == 2){
            res_ref.resize(mps_eval.dimension(1), mps_eval.dimension(1));
            contract_mps_mpo_env(res_ref.data(), res_ref.dimensions(),
                                 env_eval.data(), env_eval.dimensions(),
                                 mps_eval.data(), mps_eval.dimensions(),
                                 mpo_eval.data(), mpo_eval.dimensions());
        }
        else {
            res_ref.resize(mps_eval.dimension(1), mps_eval.dimension(1), mpo_eval.dimension(0));
            contract_mps_mpo_env(res_ref.data(), res_ref.dimensions(),
                                 env_eval.data(), env_eval.dimensions(),
                                 mps_eval.data(), mps_eval.dimensions(),
                                 mpo_eval.data(), mpo_eval.dimensions());
        }
        /* clang-format on */
    }

    template<typename env_type, typename mps_type, typename mpo_type>
    [[nodiscard]] auto contract_env_mps_mpo(const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo) {
        Eigen::Tensor<typename env_type::Scalar, env_type::NumIndices> res;
        contract_env_mps_mpo(res, env, mps, mpo);
        return res;
    }
    template<typename env_type, typename mps_type, typename mpo_type>
    [[nodiscard]] auto contract_mps_mpo_env(const TensorRead<env_type> &env, const TensorRead<mps_type> &mps, const TensorRead<mpo_type> &mpo) {
        Eigen::Tensor<typename env_type::Scalar, env_type::NumIndices> res;
        contract_mps_mpo_env(res, env, mps, mpo);
        return res;
    }
    template<typename mps_type, typename mpo_type, typename env_type>
    auto expectation_value(const mps_type &mps, const mpo_type &mpo, const env_type &envL, const env_type &envR) {
        mps_type result;
        result.resize(mps.dimensions());
        matrix_vector_product(result, mps, mpo, envL, envR);
        return contract_mps_overlap(mps, result);
    }
}