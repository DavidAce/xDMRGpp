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

template<typename Scalar, typename T>
requires tenx::sfinae::is_eigen_tensor_v<std::remove_cvref_t<T>>
decltype(auto) asScalarType2(T &&tensor) {
    static_assert(sfinae::is_any_v<Scalar, fp32, fp64, fp128, cx32, cx64, cx128>);
    using EigenType = std::remove_cvref_t<decltype(tensor)>;
    using OldScalar = typename EigenType::Scalar;
    fmt::print("T      {}\n", sfinae::type_name<T>());
    fmt::print("tensor {}\n", sfinae::type_name<decltype(tensor)>());
    if constexpr(std::is_same_v<OldScalar, Scalar>) {
        return (tensor); // Returns the exact same object as a reference
    } else if constexpr(sfinae::is_std_complex_v<OldScalar> and !sfinae::is_std_complex_v<Scalar>) {
        // Complex to Real
        using DimType       = typename EigenType::Dimensions;
        constexpr auto rank = Eigen::internal::array_size<DimType>::value;
        return Eigen::Tensor<Scalar, rank>(tensor.real().template cast<Scalar>());
    } else {
        // Cast between different precisions, e.g. fp64 to fp32
        using DimType       = typename EigenType::Dimensions;
        constexpr auto rank = Eigen::internal::array_size<DimType>::value;
        return Eigen::Tensor<Scalar, rank>(tensor.template cast<Scalar>());
    }
}

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
        return asScalarType2<T>(get_tensor(tensor));
    }
};

int main() {
    auto mpo    = MpoSite<cx64>();
    auto tensor = Eigen::Tensor<double, 1>(1000000);
    // fmt::print("{} : {}\n", sfinae::type_name<decltype(asScalarType2<fp64>(tensor))>(), fp(asScalarType2<fp64>(tensor).coeff(0)));
    fmt::print("{} : {}\n", sfinae::type_name<decltype(mpo.get_tensor<fp64>())>(), fp(mpo.get_tensor<double>().coeff(0)));
    fmt::print("{} : {}\n", sfinae::type_name<decltype(mpo.get_tensor<cx64>())>(), fp(mpo.get_tensor<cx64>().coeff(0)));

    auto tensor_fp64 = mpo.get_tensor<fp64>();
    fmt::print("{} : {}\n", sfinae::type_name<decltype(tensor_fp64)>(), fp(tensor_fp64.coeff(0)));

    auto tensor_cx64 = mpo.get_tensor<cx64>();
    fmt::print("{} : {}\n", sfinae::type_name<decltype(tensor_cx64)>(), fp(tensor_cx64.coeff(0)));

    return 0;
}
