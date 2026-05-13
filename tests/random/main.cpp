
#include "io/fmt_f128_t.h"
#include "math/float.h"
#include "math/rnd.h"
#include <fmt/format.h>
#include <fmt/std.h>
#include <stdexcept>



int main() {
    rnd::seed(1);
    for (size_t i=0; i < 10; i++) {
        auto num =  f128_t(rnd::random<fp128>(rnd::dist::uniform,0,10));
        fmt::print("rnd1:  [{:.36f}]\n", num);
        if (std::abs(num) > 5) throw std::runtime_error("rnd1 > 5");
        auto str = fmt::format("[{:.36f}]", num);
        fmt::print("str1:  {} | size {}\n", str, str.size());
#if defined(DMRG_USE_QUADMATH) || defined(DMRG_USE_FLOAT128)
        if(i == 0 and str != "[0.438341628573894261816701310882243285]")
            throw std::runtime_error(fmt::format("unexpected fp128 random value: {}", str));
#endif
    }
    rnd::seed(1);
    for (size_t i=0; i < 10; i++) fmt::print("rnd2:  [{:.36f}]\n", f128_t(rnd::random<fp128>(rnd::dist::uniform,0,10)));
    rnd::seed(1);
    for (size_t i=0; i < 10; i++) fmt::print("rnd3:  [{}]\n", rnd::random<long double>(rnd::dist::uniform,0,10));
    rnd::seed(1);
    for (size_t i=0; i < 10; i++) fmt::print("rnd4:  [{}]\n", rnd::random<double>(rnd::dist::uniform,0,10));
    return 0;
}
