#include "../contraction.h"
#include "math/tenx.h"
#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis/tblis.h>

template<typename Scalar>
void tools::common::contraction::contract_tblis(const Scalar *aptr, dimlist adim,                              //
                                                const Scalar *bptr, dimlist bdim,                              //
                                                Scalar *cptr, dimlist cdim,                                    //
                                                std::string_view la, std::string_view lb, std::string_view lc, //
                                                [[maybe_unused]] const void *cntx_ptr) {
    auto                  ta    = MArray::marray_view<const Scalar>(adim, aptr, MArray::COLUMN_MAJOR);
    auto                  tb    = MArray::marray_view<const Scalar>(bdim, bptr, MArray::COLUMN_MAJOR);
    auto                  tc    = MArray::marray_view<Scalar>(cdim, cptr, MArray::COLUMN_MAJOR);
    Scalar                alpha = 1.0;
    Scalar                beta  = 0.0;
    tblis::tensor_wrapper A_s(ta);
    tblis::tensor_wrapper B_s(tb);
    tblis::tensor_wrapper C_s(tc);

    tblis::mult(alpha, A_s, la.data(), B_s, lb.data(), beta, C_s, lc.data());
}

template void tools::common::contraction::contract_tblis(const fp32 *aptr, dimlist adim,                                //
                                                         const fp32 *bptr, dimlist bdim,                                //
                                                         fp32 *cptr, dimlist cdim,                                      //
                                                         std::string_view la, std::string_view lb, std::string_view lc, //
                                                         const void *tblis_config_ptr);
template void tools::common::contraction::contract_tblis(const cx32 *aptr, dimlist adim,                                //
                                                         const cx32 *bptr, dimlist bdim,                                //
                                                         cx32 *cptr, dimlist cdim,                                      //
                                                         std::string_view la, std::string_view lb, std::string_view lc, //
                                                         const void *tblis_config_ptr);
template void tools::common::contraction::contract_tblis(const fp64 *aptr, dimlist adim,                                //
                                                         const fp64 *bptr, dimlist bdim,                                //
                                                         fp64 *cptr, dimlist cdim,                                      //
                                                         std::string_view la, std::string_view lb, std::string_view lc, //
                                                         const void *tblis_config_ptr);
template void tools::common::contraction::contract_tblis(const cx64 *aptr, dimlist adim,                                //
                                                         const cx64 *bptr, dimlist bdim,                                //
                                                         cx64 *cptr, dimlist cdim,                                      //
                                                         std::string_view la, std::string_view lb, std::string_view lc, //
                                                         const void *tblis_config_ptr);

#endif
