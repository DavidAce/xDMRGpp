#pragma once
#include <array>
#include <string_view>
#include <vector>
#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis/tblis_config.h>
#endif

namespace settings {
#if defined(DMRG_ENABLE_TBLIS)
    inline constexpr bool tblis_enabled = true;
#else
    inline constexpr bool tblis_enabled = false;
#endif

#if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
    inline constexpr bool tblis_use_openmp = true;
#else
    inline constexpr bool tblis_use_openmp = false;
#endif
}

namespace tools::common::contraction {
    class dimlist : public std::vector<long> {
        public:
        using base_t = std::vector<long>;
        using base_t::base_t;
        template<auto rank>
        dimlist(std::array<long, rank> arr) : base_t(arr.begin(), arr.end()) {}
    };

    template<typename Scalar>
    void contract_tblis(const Scalar *aptr, dimlist adim,         //
                        const Scalar *bptr, dimlist bdim,         //
                        Scalar *cptr, dimlist cdim,               //
                        std::string_view la, std::string_view lb, //
                        std::string_view lc, const void *tblis_config_ptr);
}