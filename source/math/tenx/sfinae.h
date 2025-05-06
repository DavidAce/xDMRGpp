#pragma once
#include "fwd_decl.h"
#include <array>
#include <complex>
#include <utility>

/*!
 * \brief A collection of type-detection and type-analysis utilities using SFINAE on Eigen types
 */
namespace tenx::sfinae {

    template<typename...>
    struct print_type_and_exit_compile_time;
    template<typename T>
    [[deprecated]] inline constexpr void print_type_and_continue_compile_time([[maybe_unused]] const char *msg = nullptr) {}

    template<typename T1, typename T2>
    concept type_is = std::same_as<std::remove_cvref_t<T1>, T2>;

    template<typename T, typename... Ts>
    concept is_any_v = (type_is<T, Ts> || ...);

    template<typename T, typename... Ts>
    concept are_same_v = (type_is<T, Ts> && ...);

    template<typename T>
    concept has_value_type_v = requires { typename T::value_type; };

    template<typename T>
    concept has_vlen_type_v = requires { typename T::vlen_type; };

    template<template<class...> class Template, class... Args>
    void is_specialization_impl(const Template<Args...> &);
    template<class T, template<class...> class Template>
    concept is_specialization_v = requires(const T &t) { is_specialization_impl<Template>(t); };

    template<template<typename InnerT, auto N> class Template, typename InnerT, auto N>
    void is_specialization_with_one_parameter_impl(const Template<InnerT, N> &);
    template<class T, template<typename InnerT, auto N> class Template>
    concept is_specialization_with_one_parameter_v = requires(const T &t) { is_specialization_with_one_parameter_impl<Template>(t); };
    //    static_assert(is_specialization_with_one_parameter_v<std::array<long,3>, std::array>);

    template<typename T>
    concept is_std_complex_v = is_specialization_v<T, std::complex>;

    template<typename T>
    concept is_std_array_v = type_is<T, std::array<typename std::remove_cvref_t<T>::value_type, std::tuple_size<T>::value>>;
    static_assert(is_std_array_v<std::array<long, 3>>);

    template<typename T>
    concept is_eigen_dsizes_v =
        type_is<T, Eigen::DSizes<typename std::remove_cvref_t<T>::Base::value_type, std::tuple_size_v<typename std::remove_cvref_t<T>::Base>>>;
    //    static_assert(is_eigen_dsizes_v<Eigen::DSizes<long,3>>);

    template<typename T>
    concept is_pointer_type = std::is_pointer_v<T>;
    template<typename T>
    concept has_data_v = requires(T m) {
        { m.data() } -> is_pointer_type;
    };
    template<typename T>
    concept has_expression_v = requires(T m) {
        { m.expression() };
    };
    template<typename T>
    concept has_dimensions_v = requires(T m) {
        { m.dimensions() } -> is_eigen_dsizes_v;
    };

    template<typename T>
    concept has_self_v = std::is_same_v<T, typename T::Self>;

    template<typename T>
    concept is_eigen_tensor_v = std::is_base_of_v<Eigen::TensorBase<std::decay_t<T>, Eigen::ReadOnlyAccessors>, std::decay_t<T>>;
    //    static_assert(is_eigen_tensor_v<decltype(Eigen::Tensor<double, 3>().shuffle(std::array<long, 3>{0}))>);
    //    static_assert(is_eigen_tensor_v<Eigen::Tensor<double, 3>>);
    //    static_assert(is_eigen_tensor_v<Eigen::TensorMap<Eigen::Tensor<double, 3>>>);
    //    static_assert(!is_eigen_tensor_v<Eigen::TensorBase<Eigen::Tensor<double, 3>, Eigen::AccessorLevels::ReadOnlyAccessors>>);

    template<typename T>
    concept is_eigen_tensorbase_v = requires(T m) { requires is_specialization_with_one_parameter_v<T, Eigen::TensorBase>; };
    //    static_assert(!is_eigen_tensorbase_v<Eigen::Tensor<double, 3>>);
    //    static_assert(is_eigen_tensorbase_v<Eigen::TensorBase<Eigen::Tensor<double, 3>, Eigen::AccessorLevels::ReadOnlyAccessors>>);

    template<typename T>
    concept is_plain_tensor_v = is_eigen_tensor_v<T> and has_data_v<T> and has_dimensions_v<T> and !has_expression_v<T>;
    static_assert(is_plain_tensor_v<Eigen::Tensor<double, 3>>);
    static_assert(is_plain_tensor_v<Eigen::TensorMap<Eigen::Tensor<double, 3>>>);
    static_assert(!is_plain_tensor_v<decltype(Eigen::Tensor<double, 3>().shuffle(std::array<long, 3>{0}))>);
    static_assert(!is_plain_tensor_v<Eigen::TensorBase<Eigen::Tensor<double, 3>, Eigen::AccessorLevels::WriteAccessors>>);

    template<typename T>
    concept is_eigen_tensormap_v = requires(T m) {
        requires is_plain_tensor_v<T>;
        requires is_pointer_type<typename T::PointerType>; // Only TensorMap seems to have this
    };
    //    static_assert(!is_eigen_tensormap_v<Eigen::Tensor<double, 3>>);
    //    static_assert(!is_eigen_tensormap_v<decltype(Eigen::Tensor<double, 3>().shuffle(std::array<long, 3>{0}))>);
    //    static_assert(is_eigen_tensormap_v<Eigen::TensorMap<Eigen::Tensor<double, 3>>>);

    template<typename T, auto rank>
    concept is_eigen_tensorN = std::is_base_of_v<Eigen::TensorBase<std::remove_cvref_t<T>, Eigen::ReadOnlyAccessors>, std::remove_cvref_t<T>> && //
                               std::remove_cvref_t<T>::NumDimensions == rank;

    template<typename T> concept is_eigen_tensor1 = is_eigen_tensorN<T, 1>;
    template<typename T> concept is_eigen_tensor2 = is_eigen_tensorN<T, 2>;
    template<typename T> concept is_eigen_tensor3 = is_eigen_tensorN<T, 3>;

    template<typename T, typename = std::void_t<>>
    struct has_NumIndices : public std::false_type {};
    template<typename T>
    struct has_NumIndices<T, std::void_t<decltype(std::declval<T>().NumIndices)>> : public std::true_type {};
    template<typename T>
    inline constexpr bool has_NumIndices_v = has_NumIndices<T>::value;
    template<typename T>
    using is_eigen_matrix = std::is_base_of<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>;
    template<typename T>
    inline constexpr bool is_eigen_matrix_v = is_eigen_matrix<T>::value;
    template<typename T>
    using is_eigen_array = std::is_base_of<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>>;
    template<typename T>
    inline constexpr bool is_eigen_array_v = is_eigen_array<T>::value;
    template<typename T>
    using is_eigen_dense = std::is_base_of<Eigen::DenseBase<std::decay_t<T>>, std::decay_t<T>>;
    template<typename T>
    inline constexpr bool is_eigen_dense_v = is_eigen_dense<T>::value;
    template<typename T>
    using is_eigen_map = std::is_base_of<Eigen::MapBase<std::decay_t<T>, Eigen::ReadOnlyAccessors>, std::decay_t<T>>;
    template<typename T>
    inline constexpr bool is_eigen_map_v = is_eigen_map<T>::value;
    template<typename T>
    using is_eigen_plain = std::is_base_of<Eigen::PlainObjectBase<std::decay_t<T>>, std::decay_t<T>>;
    template<typename T>
    inline constexpr bool is_eigen_plain_v = is_eigen_plain<T>::value;
    template<typename T>
    using is_eigen_base = std::is_base_of<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>>;
    template<typename T>
    inline constexpr bool is_eigen_base_v = is_eigen_base<T>::value;
    template<typename T>
    struct is_eigen_core : public std::false_type {};
    template<typename T, int rows, int cols, int StorageOrder>
    struct is_eigen_core<Eigen::Matrix<T, rows, cols, StorageOrder>> : public std::true_type {};
    template<typename T, int rows, int cols, int StorageOrder>
    struct is_eigen_core<Eigen::Array<T, rows, cols, StorageOrder>> : public std::true_type {};
    template<typename T>
    inline constexpr bool is_eigen_core_v = is_eigen_core<T>::value;
    template<typename T>
    struct is_eigen_any {
        static constexpr bool value = is_eigen_base<T>::value or is_eigen_tensor_v<T>;
    };
    template<typename T>
    inline constexpr bool is_eigen_any_v = is_eigen_any<T>::value;
    template<typename T>
    struct is_eigen_contiguous {
        static constexpr bool value = is_eigen_any<T>::value and has_data_v<T>;
    };
    template<typename T>
    inline constexpr bool is_eigen_contiguous_v = is_eigen_contiguous<T>::value;
    template<typename T>
    class is_eigen_1d {
        private:
        template<typename U>
        static constexpr auto test() {
            if constexpr(is_eigen_map<U>::value) return test<typename U::PlainObject>();
            if constexpr(is_eigen_dense<U>::value) return U::RowsAtCompileTime == 1 or U::ColsAtCompileTime == 1;
            if constexpr(is_eigen_tensor_v<U> and has_NumIndices<U>::value)
                return U::NumIndices == 1;
            else
                return false;
        }

        public:
        static constexpr bool value = test<T>();
    };
    template<typename T>
    inline constexpr bool is_eigen_1d_v = is_eigen_1d<T>::value;

    template<typename T>
    class is_eigen_colmajor {
        template<typename U>
        static constexpr bool test() {
            if constexpr(is_eigen_base<U>::value) return not U::IsRowMajor;
            if constexpr(is_eigen_tensor_v<U>)
                return Eigen::ColMajor == static_cast<Eigen::StorageOptions>(U::Layout);
            else
                return false;
        }

        public:
        static constexpr bool value = test<T>();
    };
    template<typename T>
    inline constexpr bool is_eigen_colmajor_v = is_eigen_colmajor<T>::value;

    template<typename T>
    class is_eigen_rowmajor {
        template<typename U>
        static constexpr bool test() {
            if constexpr(is_eigen_base<U>::value) return U::IsRowMajor;
            if constexpr(is_eigen_tensor_v<U>)
                return Eigen::RowMajor == static_cast<Eigen::StorageOptions>(U::Layout);
            else
                return false;
        }

        public:
        static constexpr bool value = test<T>();
    };
    template<typename T>
    inline constexpr bool is_eigen_rowmajor_v = is_eigen_rowmajor<T>::value;

    template<typename T>
    concept is_single_prec_v = type_is<T, fp32> or type_is<T, cx32>;

    template<typename T>
    concept is_double_prec_v = type_is<T, fp64> or type_is<T, cx64>;

    template<typename T>
    concept is_long_double_prec_v = type_is<T, long double> or type_is<T, std::complex<long double>>;

    template<typename T>
    concept is_quadruple_prec_v = type_is<T, fp128> or type_is<T, cx128>;

}
