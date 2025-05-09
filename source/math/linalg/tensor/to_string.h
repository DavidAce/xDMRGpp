#pragma once
#include "../common.h"
#include "math/cast.h"
#include <array>
#include <fmt/core.h>
#include <io/fmt_custom.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace linalg::tensor {
    template<typename T>
    std::string formatted_number(T number, int prec, int min_width, int min_width_real, int min_width_imag) {
        if constexpr(linalg::is_std_complex_v<T>) {
            if constexpr(std::is_floating_point_v<typename T::value_type>) {
                std::string real = fmt::format("({0:.{1}f}", fp(number.real()), prec);
                std::string imag = fmt::format("{0:.{1}f})", fp(number.imag()), prec);
                std::string cplx = fmt::format("{:>},{:<}", real, imag);
                return fmt::format("{0:>{1}}", cplx, min_width_real + min_width_imag + 3); // Two doubles, comma, and parentheses

            } else if constexpr(std::is_integral_v<typename T::value_type>) {
                std::string real = fmt::format("({}", fp(number.real()));
                std::string imag = fmt::format("{})", fp(number.imag()));
                std::string cplx = fmt::format("{:>},{:<}", real, imag);
                return fmt::format("{0:>{1}}", cplx, min_width_real + min_width_imag + 3); // Two doubles, comma, and parentheses
            }
        } else if constexpr(std::is_floating_point_v<T>)
            return fmt::format("{0:>{1}.{2}f}", fp(number), min_width, prec);
        else if constexpr(std::is_integral_v<T>)
            return fmt::format("{0:>{1}}", fp(number), min_width);
    }

    template<typename T>
    std::string to_string(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr, int prec = 1, int width = 2, std::string_view sep = ", ") {
        using Evaluator = Eigen::TensorEvaluator<const Eigen::TensorForcedEvalOp<const T>, Eigen::DefaultDevice>;
        using Scalar    = typename Eigen::internal::remove_const<typename Evaluator::Scalar>::type;

        // Evaluate the expression if needed
        Eigen::TensorForcedEvalOp<const T> eval = expr.eval();
        Evaluator                          tensor(eval, Eigen::DefaultDevice());
        tensor.evalSubExprsIfNeeded(nullptr);
        Eigen::Index total_size = Eigen::internal::array_prod(tensor.dimensions());
        if(total_size == 1) { return fmt::format("[{}]", formatted_number(tensor.data()[0], prec, width, width, width)); }
        if(total_size > 0 and tensor.dimensions().size() > 0) {
            Eigen::Index first_dim = tensor.dimensions()[0];
            if constexpr(T::NumDimensions == 4) first_dim = tensor.dimensions()[0] * tensor.dimensions()[1];
            if constexpr(T::NumDimensions == 6) first_dim = tensor.dimensions()[0] * tensor.dimensions()[1] * tensor.dimensions()[2];
            if constexpr(T::NumDimensions == 8) first_dim = tensor.dimensions()[0] * tensor.dimensions()[1] * tensor.dimensions()[2] * tensor.dimensions()[3];
            Eigen::Index other_dim = total_size / first_dim;
            auto matrix = Eigen::Map<Eigen::Array<typename T::Scalar, Eigen::Dynamic, Eigen::Dynamic, Evaluator::Layout>>(tensor.data(), first_dim, other_dim);
            std::string str;

            int comma = 1;
            if constexpr(std::is_integral_v<Scalar>) {
                comma = 0;
                prec  = 0;
            }
            if constexpr(linalg::is_std_complex_v<Scalar>)
                if constexpr(std::is_integral_v<typename Scalar::value_type>) {
                    comma = 0;
                    prec  = 0;
                }

            auto max_val   = static_cast<double>(matrix.cwiseAbs().maxCoeff());
            int  min_width = std::max(width, safe_cast<int>(1 + std::max(0.0, std::log10(max_val))) + comma + prec);

            int min_width_real = min_width;
            int min_width_imag = min_width;
            if constexpr(linalg::is_std_complex_v<Scalar>) {
                auto max_val_real = static_cast<double>(matrix.real().cwiseAbs().maxCoeff());
                auto max_val_imag = static_cast<double>(matrix.imag().cwiseAbs().maxCoeff());
                min_width_real    = std::max(width, safe_cast<int>(1 + std::max(0.0, std::log10(max_val_real))) + comma + prec);
                min_width_imag    = std::max(width, safe_cast<int>(1 + std::max(0.0, std::log10(max_val_imag))) + comma + prec);
                if(matrix.real().minCoeff() < 0) min_width_real += 1;
                if(matrix.imag().minCoeff() < 0) min_width_imag += 1;
            } else {
                if(matrix.minCoeff() < 0) min_width += 1;
            }

            for(long i = 0; i < first_dim; i++) {
                str += fmt::format("[");
                for(long j = 0; j < other_dim; j++) {
                    str += formatted_number(matrix(i, j), prec, min_width, min_width_real, min_width_imag);
                    if(j < other_dim - 1) str += sep;
                }
                if(i < first_dim - 1)
                    str += fmt::format("]\n");
                else
                    str += fmt::format("]");
            }
            tensor.cleanup();
            return str;
        } else {
            tensor.cleanup();
            return "[]";
        }
    }

}