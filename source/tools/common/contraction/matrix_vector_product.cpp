#include "../contraction.h"
#include "math/tenx.h"
#include "tid/tid.h"
#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis/util/configs.h>
    #include <tblis/util/thread.h>
    #include <tci/tci_config.h>
    #if defined(TCI_USE_OPENMP_THREADS)
        #include <omp.h>
    #endif
#endif

using namespace tools::common::contraction;

/* clang-format off */
template<typename Scalar>
void tools::common::contraction::matrix_vector_product(      Scalar * res_ptr,
                                                       const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                                                       const Scalar * const mpo_ptr, std::array<long,4> mpo_dims,
                                                       const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                                                       const Scalar * const envR_ptr, std::array<long,3> envR_dims){

//    auto t_matvec = tid::tic_token("matrix_vector_product", tid::level::extra);

    // This applies the mpo's with corresponding environments to local multisite mps
    // This is usually the operation H|psi>  or H²|psi>
    auto res = Eigen::TensorMap<Eigen::Tensor<Scalar,3>>(res_ptr,mps_dims);
    auto mps = Eigen::TensorMap<const  Eigen::Tensor<Scalar,3>>(mps_ptr,mps_dims);
    auto mpo = Eigen::TensorMap<const  Eigen::Tensor<Scalar,4>>(mpo_ptr,mpo_dims);
    auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envL_ptr,envL_dims);
    auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envR_ptr,envR_dims);

    assert(mps.dimension(1) == envL.dimension(0));
    assert(mps.dimension(2) == envR.dimension(0));
    assert(mps.dimension(0) == mpo.dimension(2));
    assert(envL.dimension(2) == mpo.dimension(0));
    assert(envR.dimension(2) == mpo.dimension(1));

#if defined(DMRG_ENABLE_TBLIS)
    if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>){
        static const tblis::tblis_config_s *tblis_config = tblis::tblis_get_config(get_tblis_arch().data());
        #if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
        tblis_set_num_threads(static_cast<unsigned int>(omp_get_max_threads()));
        #endif
        if (mps.dimension(1) >= mps.dimension(2)){
            Eigen::Tensor<Scalar, 4> mpsenvL(mps.dimension(0), mps.dimension(2), envL.dimension(1), envL.dimension(2));
            // Eigen::Tensor<Scalar, 4> mpsenvLmpo(mps.dimension(2), envL.dimension(1), mpo.dimension(1), mpo.dimension(3));
            Eigen::Tensor<Scalar, 4> mpsenvLmpo_alt(mpo.dimension(1), mpo.dimension(3), mps.dimension(2), envL.dimension(1));

            // contract_tblis(mps, envL, mpsenvL, "afb", "fcd", "abcd", tblis_config);
            // contract_tblis(mpo, mpsenvL, mpsenvLmpo_alt, "qhri", "rgjq", "higj", tblis_config);
            // contract_tblis(mpsenvLmpo_alt, envR, res, "higj", "gkh", "ijk", tblis_config);
            contract_tblis(mps.data(), mps.dimensions(),            //
                           envL.data(), envL.dimensions(),          //
                           mpsenvL.data(), mpsenvL.dimensions(),    //
                           "afb", "fcd", "abcd", tblis_config);
            contract_tblis(mpo.data(), mpo.dimensions(),                       //
                           mpsenvL.data(), mpsenvL.dimensions(),               //
                           mpsenvLmpo_alt.data(), mpsenvLmpo_alt.dimensions(), //
                           "qhri", "rgjq", "higj", tblis_config);              //
            contract_tblis(mpsenvLmpo_alt.data(), mpsenvLmpo_alt.dimensions(), //
                            envR.data(), envR.dimensions(),                     //
                            res.data(), res.dimensions(),                      //
                            "higj", "gkh", "ijk", tblis_config);
        }
        else{
            Eigen::Tensor<Scalar, 4> mpsenvR(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
            Eigen::Tensor<Scalar, 4> mpsenvRmpo(mps.dimension(1), envR.dimension(1), mpo.dimension(0), mpo.dimension(3));
            // contract_tblis(mps, envR, mpsenvR, "abf", "fcd", "abcd", tblis_config);
            // contract_tblis(mpsenvR, mpo, mpsenvRmpo, "qijk", "rkql", "ijrl", tblis_config);
            // contract_tblis(mpsenvRmpo, envL, res, "qkri", "qjr", "ijk", tblis_config);
            contract_tblis(mps.data(), mps.dimensions(),           //
                           envR.data(), envR.dimensions(),         //
                           mpsenvR.data(), mpsenvR.dimensions(),   //
                           "abf", "fcd", "abcd", tblis_config);
            contract_tblis(mpsenvR.data(), mpsenvR.dimensions(),       //
                           mpo.data(), mpo.dimensions(),               //
                           mpsenvRmpo.data(), mpsenvRmpo.dimensions(), //
                           "qijk", "rkql", "ijrl", tblis_config);
            contract_tblis(mpsenvRmpo.data(), mpsenvRmpo.dimensions(),  //
                           envL.data(), envL.dimensions(),              //
                           res.data(), res.dimensions(),                //
                           "qkri", "qjr", "ijk", tblis_config);
        }
    }else{
        auto &threads = tenx::threads::get();
        if (mps.dimension(1) >= mps.dimension(2)){
            res.device(*threads->dev) = mps
                                     .contract(envL, tenx::idx({1}, {0}))
                                     .contract(mpo,  tenx::idx({3, 0}, {0, 2}))
                                     .contract(envR, tenx::idx({0, 2}, {0, 2}))
                                     .shuffle(tenx::array3{1, 0, 2});
        }else{
            res.device(*threads->dev) = mps
                                     .contract(envR, tenx::idx({2}, {0}))
                                     .contract(mpo,  tenx::idx({3, 0}, {1, 2}))
                                     .contract(envL, tenx::idx({0, 2}, {0, 2}))
                                     .shuffle(tenx::array3{1, 2, 0});
        }
    }
    #else
    auto &threads = tenx::threads::get();
    if (mps.dimension(1) >= mps.dimension(2)){
        res.device(*threads->dev) = mps
                                 .contract(envL, tenx::idx({1}, {0}))
                                 .contract(mpo,  tenx::idx({3, 0}, {0, 2}))
                                 .contract(envR, tenx::idx({0, 2}, {0, 2}))
                                 .shuffle(tenx::array3{1, 0, 2});
    }else{
        res.device(*threads->dev) = mps
                                 .contract(envR, tenx::idx({2}, {0}))
                                 .contract(mpo,  tenx::idx({3, 0}, {1, 2}))
                                 .contract(envL, tenx::idx({0, 2}, {0, 2}))
                                 .shuffle(tenx::array3{1, 2, 0});
    }
    #endif
}
using namespace tools::common::contraction;
template void tools::common::contraction::matrix_vector_product(      fp32 *       res_ptr,
                                                                const fp32 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const fp32 * const mpo_ptr, std::array<long,4> mpo_dims,
                                                                const fp32 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const fp32 * const envR_ptr, std::array<long,3> envR_dims);
template void tools::common::contraction::matrix_vector_product(      fp64 *       res_ptr,
                                                                const fp64 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const fp64 * const mpo_ptr, std::array<long,4> mpo_dims,
                                                                const fp64 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const fp64 * const envR_ptr, std::array<long,3> envR_dims);
template void tools::common::contraction::matrix_vector_product(      cx32 *       res_ptr,
                                                                const cx32 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const cx32 * const mpo_ptr, std::array<long,4> mpo_dims,
                                                                const cx32 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const cx32 * const envR_ptr, std::array<long,3> envR_dims);
template void tools::common::contraction::matrix_vector_product(      cx64 *       res_ptr,
                                                                const cx64 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const cx64 * const mpo_ptr, std::array<long,4> mpo_dims,
                                                                const cx64 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const cx64 * const envR_ptr, std::array<long,3> envR_dims);



template<typename Scalar, typename mpo_type>
void tools::common::contraction::matrix_vector_product(Scalar * res_ptr,
                           const Scalar * const mps_ptr, std::array<long,3> mps_dims,
                           const std::vector<mpo_type> & mpos_shf,
                           const Scalar * const envL_ptr, std::array<long,3> envL_dims,
                           const Scalar * const envR_ptr, std::array<long,3> envR_dims) {
    // Make sure the mpos are pre-shuffled. If not, shuffle and call this function again
    bool is_shuffled = mpos_shf.front().dimension(2) == envL_dims[2] and mpos_shf.back().dimension(3) == envR_dims[2];
    if(not is_shuffled){
        // mpos_shf are not actually shuffled. Let's shuffle.
        std::vector<Eigen::Tensor<Scalar, 4>> mpos_really_shuffled;
        for (const auto & mpo : mpos_shf) {
            mpos_really_shuffled.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1}));
        }
        return matrix_vector_product(res_ptr, mps_ptr, mps_dims, mpos_really_shuffled, envL_ptr, envL_dims, envR_ptr, envR_dims);
    }


    auto &threads  = tenx::threads::get();
    auto mps_out = Eigen::TensorMap<Eigen::Tensor<Scalar,3>>(res_ptr,mps_dims);
    auto mps_in  = Eigen::TensorMap<const  Eigen::Tensor<Scalar,3>>(mps_ptr,mps_dims);
    auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envL_ptr,envL_dims);
    auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(envR_ptr,envR_dims);

    assert(mps_in.dimension(1) == envL.dimension(0));
    assert(mps_in.dimension(2) == envR.dimension(0));


    auto  L        = mpos_shf.size();

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
    static const tblis::tblis_config_s *tblis_config = tblis::tblis_get_config(get_tblis_arch().data());
    #if defined(TCI_USE_OPENMP_THREADS)
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
    auto  mps_tmp1 = Eigen::Tensor<Scalar, 6>();
    auto  mps_tmp2 = Eigen::Tensor<Scalar, 6>();
    mps_tmp1.resize(tenx::array6{d0, d1, d2, d3, d5, d4});
    #if defined(DMRG_ENABLE_TBLIS)
    if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>) {
        auto mps_tmp1_map4 = Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>(mps_tmp1.data(), std::array{d0 * d1, d2, d3, d4 * d5});
        // contract_tblis(mps_in, envL, mps_tmp1_map4, "afb", "fcd", "abcd", tblis_config);
        contract_tblis(mps_in.data(),mps_in.dimensions(),                 //
                       envL.data(), envL.dimensions(),                    //
                       mps_tmp1_map4.data(), mps_tmp1_map4.dimensions(),  //
                       "afb", "fcd", "abcd", tblis_config);
    } else
    #endif
    {
        mps_tmp1.device(*threads->dev) = mps_in.contract(envL, tenx::idx({1}, {0})).reshape(new_shp6).shuffle(tenx::array6{0, 1, 2, 3, 5, 4});
    }
    for(size_t idx = 0; idx < L; ++idx) {
        const auto &mpo = mpos_shf[idx];
        // Set up the dimensions for the reshape after the contraction
        d0       = mpodimprod(idx + 1, idx + 2); // if idx == k, this has the mpo at idx == k+1
        d1       = mpodimprod(idx + 2, L);       // if idx == 0,  this has the mpos at idx == k+2...L-1
        d2       = mps_tmp1.dimension(2);
        d3       = mps_tmp1.dimension(3);
        d4       = mpodimprod(0, idx + 1); // if idx == 0, this has the mpos at idx == 0...k (i.e. including the one from the current iteration)
        d5       = mpo.dimension(3);       // The virtual bond of the current mpo
        #if defined(DMRG_ENABLE_TBLIS)
        if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>) {
            auto md  = mps_tmp1.dimensions();
            new_shp6 = tenx::array6{d0, d1, d2, d3, d4, d5};
            mps_tmp2.resize(new_shp6);
            auto map_shp6 = tenx::array6{md[1], md[2], md[3], md[4], mpo.dimension(1), mpo.dimension(3)};
            auto mps_tmp2_map = Eigen::TensorMap<Eigen::Tensor<Scalar,6>>(mps_tmp2.data(), map_shp6);
            // contract_tblis(mps_tmp1, mpo, mps_tmp2_map, "qbcder", "qfrg", "bcdefg", tblis_config);
            contract_tblis(mps_tmp1.data(),mps_tmp1.dimensions(),          //
                           mpo.data(), mpo.dimensions(),                   //
                           mps_tmp2_map.data(), mps_tmp2_map.dimensions(), //
                           "qbcder", "qfrg", "bcdefg", tblis_config);
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
    if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>) {
        auto mps_tmp1_map4 = Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>(mps_tmp1.data(), std::array{d0, d1, d2, d3});
        // contract_tblis(mps_tmp1_map4, envR, mps_out, "qjir", "qkr", "ijk", tblis_config);
        contract_tblis(mps_tmp1_map4.data(), mps_tmp1_map4.dimensions(),
                       envR.data()         , envR.dimensions(),
                       mps_out.data()      , mps_out.dimensions(),
                       "qjir", "qkr", "ijk", tblis_config);
    } else
    #endif
    {
        mps_out.device(*threads->dev) = mps_tmp1.reshape(tenx::array4{d0, d1, d2, d3}).contract(envR, tenx::idx({0, 3}, {0, 2})).shuffle(tenx::array3{1, 0, 2});
    }
}

using namespace tools::common::contraction;
template void tools::common::contraction::matrix_vector_product(      fp32 *       res_ptr,
                                                                const fp32 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const std::vector<Eigen::Tensor<fp32, 4>> & mpos_shf,
                                                                const fp32 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const fp32 * const envR_ptr, std::array<long,3> envR_dims);
template void tools::common::contraction::matrix_vector_product(      fp64 *       res_ptr,
                                                                const fp64 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const std::vector<Eigen::Tensor<fp64, 4>> & mpos_shf,
                                                                const fp64 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const fp64 * const envR_ptr, std::array<long,3> envR_dims);
template void tools::common::contraction::matrix_vector_product(      cx64 *       res_ptr,
                                                                const cx64 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const std::vector<Eigen::Tensor<cx64, 4>> & mpos_shf,
                                                                const cx64 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const cx64 * const envR_ptr, std::array<long,3> envR_dims);
template void tools::common::contraction::matrix_vector_product(      cx32 *       res_ptr,
                                                                const cx32 * const mps_ptr, std::array<long,3> mps_dims,
                                                                const std::vector<Eigen::Tensor<cx32, 4>> & mpos_shf,
                                                                const cx32 * const envL_ptr, std::array<long,3> envL_dims,
                                                                const cx32 * const envR_ptr, std::array<long,3> envR_dims);


