#include "io/fmt_custom.h"
#include "math/float.h"
#include "math/tenx.h"
#include "tools/common/log.h"
#include <cassert>
#include <charconv>
#include <Eigen/Core>
#include <fmt/format.h>
#include <fmt/std.h>
#include <stdfloat>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename Scalar>
struct MpoSite {
    Eigen::Tensor<Scalar, 1> tensor = {};
    MpoSite() {
        tensor.resize(1000000);
        tensor.setZero();
    }
    Eigen::Tensor<Scalar,1> get_tensor(const Eigen::Tensor<Scalar,1> &t ) {
        Eigen::Tensor<Scalar, 1> tensor_internal = t;
        tensor(0)                                = 1.0;
        tensor(50000)                            = 1.0;
        return t;
    }
    template<typename T>
    decltype(auto) get_tensor() {
        return tenx::asScalarType<T>(get_tensor(tensor));
    }
};

int main() {
    auto mpo    = MpoSite<cx64>();
    auto tensor = Eigen::Tensor<double, 1>(1000000);
    decltype(auto) tensor_ref = tenx::asScalarType<fp64>(tensor);
    static_assert(std::is_lvalue_reference_v<decltype(tensor_ref)>);
    assert(&tensor_ref == &tensor);

    auto tensor_copy = tenx::asScalarType<fp64>(Eigen::Tensor<double, 1>(1000000));
    static_assert(not std::is_reference_v<decltype(tensor_copy)>);
    assert(tensor_copy.size() == 1000000);

    fmt::print("{} : {}\n", sfinae::type_name<decltype(mpo.get_tensor<fp64>())>(), fp(mpo.get_tensor<double>().coeff(0)));
    fmt::print("{} : {}\n", sfinae::type_name<decltype(mpo.get_tensor<cx64>())>(), fp(mpo.get_tensor<cx64>().coeff(0)));

    auto tensor_fp64 = mpo.get_tensor<fp64>();
    fmt::print("{} : {}\n", sfinae::type_name<decltype(tensor_fp64)>(), fp(tensor_fp64.coeff(0)));

    auto tensor_cx64 = mpo.get_tensor<cx64>();
    fmt::print("{} : {}\n", sfinae::type_name<decltype(tensor_cx64)>(), fp(tensor_cx64.coeff(0)));

    return 0;
}
