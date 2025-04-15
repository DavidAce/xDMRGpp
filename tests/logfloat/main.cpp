
#include "io/fmt_custom.h"
#include "math/float.h"
#include "tools/common/log.h"
#include <cassert>
#include <charconv>
#include <fmt/format.h>
#include <fmt/std.h>
#include <stdfloat>

int main() {
    // Example quad-precision value.
    std::float128_t               a = -1234567890123456789012345678901234.1234567890123456789012345678901234f128;
    std::complex<std::float128_t> c(a, -a);
    std::complex<std::float128_t> c0(a, 0.0);
    // This now uses our custom formatter, letting you specify precision and format:
    double d = -4.01234567890123456789012345678901234;
    fmt::print("Double  precision value: {:>15.10f}\n", fp(d));
    fmt::print("Quad    precision value: {:.20f}\n", fp(a));
    fmt::print("Quad cx precision value: {:+.20f}\n", fp(c));
    fmt::print("Quad cx precision value: {}\n", fp(c0));

    return 0;
}
