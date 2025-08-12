#include "../contraction.h"
#include "math/tenx.h"
#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis/tblis.h>

std::string tools::common::contraction::get_tblis_arch() {
    #if defined(__GNUC__)
    if(__builtin_cpu_supports("x86-64-v4")) return "skx";
    if(__builtin_cpu_is("znver3") or __builtin_cpu_is("znver2") or __builtin_cpu_is("znver1")) return "zen";
    if(__builtin_cpu_supports("x86-64-v3")) return "haswell";
    #endif
    return "haswell";
}

template<typename Scalar>
void tools::common::contraction::contract_tblis(const Scalar *aptr, dimlist adim,                              //
                                                const Scalar *bptr, dimlist bdim,                              //
                                                Scalar *cptr, dimlist cdim,                                    //
                                                std::string_view la, std::string_view lb, std::string_view lc, //
                                                const void *cntx_ptr) {
    auto  *cntx  = static_cast<const tblis::tblis_config *>(cntx_ptr);
    auto   ta    = MArray::marray_view<const Scalar>(adim, aptr, MArray::COLUMN_MAJOR);
    auto   tb    = MArray::marray_view<const Scalar>(bdim, bptr, MArray::COLUMN_MAJOR);
    auto   tc    = MArray::marray_view<Scalar>(cdim, cptr, MArray::COLUMN_MAJOR);
    Scalar alpha = 1.0;
    Scalar beta  = 0.0;

    tblis::tblis_tensor A_s(alpha, ta.data(), ta.dimension(), ta.lengths().data(), ta.strides().data());
    tblis::tblis_tensor B_s(tb.data(), tb.dimension(), tb.lengths().data(), tb.strides().data());
    tblis::tblis_tensor C_s(beta, tc.data(), tc.dimension(), tc.lengths().data(), tc.strides().data());

    tblis_tensor_mult(nullptr, cntx, &A_s, la.data(), &B_s, lb.data(), &C_s, lc.data());
}
template void tools::common::contraction::contract_tblis(const fp32 *aptr, dimlist adim,                                //
                                                         const fp32 *bptr, dimlist bdim,                                //
                                                         fp32 *cptr, dimlist cdim,                                      //
                                                         std::string_view la, std::string_view lb, std::string_view lc, //
                                                         const void *tblis_config_ptr);
template void tools::common::contraction::contract_tblis(const fp64 *aptr, dimlist adim,                                //
                                                         const fp64 *bptr, dimlist bdim,                                //
                                                         fp64 *cptr, dimlist cdim,                                      //
                                                         std::string_view la, std::string_view lb, std::string_view lc, //
                                                         const void *tblis_config_ptr);

// template<typename ea_type, typename eb_type, typename ec_type>
// void tools::common::contraction::contract_tblis(const TensorRead<ea_type> &ea, const TensorRead<eb_type> &eb, TensorWrite<ec_type> &ec, std::string_view la,
//                                                 std::string_view lb, std::string_view lc, const void *tblis_config_ptr) {
//     auto       *tblis_config = static_cast<const tblis::tblis_config_s *>(tblis_config_ptr);
//     const auto &ea_ref       = static_cast<const ea_type &>(ea);
//     const auto &eb_ref       = static_cast<const eb_type &>(eb);
//     auto       &ec_ref       = static_cast<ec_type &>(ec);
//
//     tblis::len_vector da, db, dc;
//     da.assign(ea_ref.dimensions().begin(), ea_ref.dimensions().end());
//     db.assign(eb_ref.dimensions().begin(), eb_ref.dimensions().end());
//     dc.assign(ec_ref.dimensions().begin(), ec_ref.dimensions().end());
//
//     auto                     ta    = tblis::varray_view<const typename ea_type::Scalar>(da, ea_ref.data(), tblis::COLUMN_MAJOR);
//     auto                     tb    = tblis::varray_view<const typename eb_type::Scalar>(db, eb_ref.data(), tblis::COLUMN_MAJOR);
//     auto                     tc    = tblis::varray_view<typename ec_type::Scalar>(dc, ec_ref.data(), tblis::COLUMN_MAJOR);
//     typename ea_type::Scalar alpha = 1.0;
//     typename ec_type::Scalar beta  = 0.0;
//
//     tblis::tblis_tensor A_s(alpha, ta);
//     tblis::tblis_tensor B_s(tb);
//     tblis::tblis_tensor C_s(beta, tc);
//
//     tblis_tensor_mult(nullptr, tblis_config, &A_s, la.data(), &B_s, lb.data(), &C_s, lc.data());
// }
//
// /* clang-format off */
// template<typename Scalar, auto rank>
// using rmap = Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>>;
// template<typename Scalar, auto rank>
// using wmap = Eigen::TensorMap<Eigen::Tensor<Scalar, rank>>;
// template<typename Scalar, auto rank>
// using rmap = Eigen::TensorMap<const Eigen::Tensor<Scalar, rank>>;
// template<typename Scalar, auto rank>
// using wmap = Eigen::TensorMap<Eigen::Tensor<Scalar, rank>>;
//
//
// template void tools::common::contraction::contract_tblis(
//     const TensorRead<rmap<fp32, 3>>  &ea,
//     const TensorRead<rmap<fp32, 3>>  &eb,
//          TensorWrite<wmap<fp32, 4>>  &ec,
//     std::string_view la, std::string_view lb, std::string_view lc, const void *tblis_config_ptr);
//
// template void tools::common::contraction::contract_tblis(
//     const TensorRead<rmap<fp32, 3>>  &ea,
//     const TensorRead<rmap<fp32, 3>>  &eb,
//          TensorWrite<wmap<fp32, 4>>  &ec,
//     std::string_view la, std::string_view lb, std::string_view lc, const void *tblis_config_ptr);
//
//
// template void tools::common::contraction::contract_tblis(
//     const TensorRead<fp64>  &ea,
//     const TensorRead<fp64>  &eb,
//          TensorWrite<fp64>  &ec,
//     std::string_view la, std::string_view lb, std::string_view lc, const void *tblis_config_ptr);
/* clang-format on */

#endif
