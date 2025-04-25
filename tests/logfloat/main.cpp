#include "io/fmt_custom.h"
#include "math/float.h"
#include "tools/common/log.h"
#include <cassert>
#include <charconv>
#include <Eigen/Core>
#include <fmt/format.h>
#include <fmt/std.h>
#include <stdfloat>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
    // Example quad-precision value.
    std::float128_t               a = -1234567890123456789012345678901234.1234567890123456789012345678901234f128;
    std::complex<std::float128_t> c(a, -a);
    std::complex<std::float128_t> c0(a, 0.0);
    std::array                    arr  = {a, a, a};
    std::array                    arr2 = {0.0, 1.0, 2.0};
    std::vector                   vec  = {0.0, 1.0, 2.0};
    Eigen::Tensor<double, 1>      t(10);
    std::vector<fp<double>>       vec2 = {1.0, 2.0, 3.0};
    t.setConstant(1.0);
    // This now uses our custom formatter, letting you specify precision and format:
    double d = -4.01234567890123456789012345678901234;
    auto   e = fv(arr);
    auto   f = fv(arr2);
    auto   g = fv(vec);
    auto   h = fv(t);
    auto   i = fpv(vec2);
    fmt::print("Double  precision value: {:>15.10f}\n", fp(d));
    fmt::print("Quad    precision value: {:.20f}\n", fp(a));
    fmt::print("Quad cx precision value: {:+.20f}\n", fp(c));
    fmt::print("Quad cx precision value: {}\n", fp(c0));
    fmt::print("Quad cx precision value: {::+.20f}\n", fv(arr));
    fmt::print("e                      : {::+.20f}\n", e);
    fmt::print("f                      : {::+.20f}\n", f);
    fmt::print("g                      : {::.20f}\n", g);
    fmt::print("h                      : {::.20f}\n", h);
    fmt::print("i                      : {::.20f}\n", i);



    return 0;
}
