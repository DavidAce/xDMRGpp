#pragma once

#include <unsupported/Eigen/CXX11/Tensor> // Include before sfinae
//
#include "sfinae.h"
#include "threads.h"

namespace tenx {
    template<typename Derived, typename Device = Eigen::DefaultDevice>
    class selfCleaningEvaluator {
        private:
        using Evaluator = Eigen::TensorEvaluator<const Eigen::TensorForcedEvalOp<const Derived>, Device>;
        Evaluator m_eval;

        public:
        using Dimensions                    = typename Evaluator::Dimensions;
        using Scalar                        = typename Eigen::internal::remove_const<typename decltype(m_eval)::Scalar>::type;
        static constexpr auto NumDimensions = Eigen::internal::array_size<Dimensions>::value;

        selfCleaningEvaluator(const Evaluator &eval) : m_eval(eval) {}
        template<int AccessLevel>
        selfCleaningEvaluator(const Eigen::TensorBase<Derived, AccessLevel> &expr, const Device &device = Device()) : m_eval(Evaluator(expr.eval(), device)) {
            m_eval.evalSubExprsIfNeeded(nullptr);
        }

        ~selfCleaningEvaluator() {
            // The whole point of this object is to call cleanup automatically on destruct.
            // If there are pending operations to evaluate, m_eval will allocate a buffer to hold a result,
            // which needs to be deallocated.
            m_eval.cleanup();
        }

        //        constexpr auto rank() { return DimType{}.size(); }
        [[nodiscard]] consteval auto rank() const { return NumDimensions; }
        [[nodiscard]] constexpr auto dimensions() const { return m_eval.dimensions(); }
        [[nodiscard]] constexpr auto size() const { return Eigen::internal::array_prod(m_eval.dimensions()); } // Some ops don't have .size()?
        [[nodiscard]] constexpr auto data() const { return m_eval.data(); }
        [[nodiscard]] constexpr auto dimension(unsigned long n) const { return m_eval.dimensions()[n]; }

        const Evaluator *operator->() const { return &m_eval; }
        Evaluator       *operator->() { return &m_eval; }
        [[nodiscard]] constexpr auto   map() const { return Eigen::TensorMap<const Eigen::Tensor<Scalar, NumDimensions>>(m_eval.data(), m_eval.dimensions()); }
    };

    // Evaluates expressions if needed
    template<typename T, int AccessLevel>
    requires(!tenx::sfinae::is_plain_tensor_v<T>)
    auto asEval(const Eigen::TensorBase<T, AccessLevel> &expr) {
        auto &threads = threads::get();
        return selfCleaningEvaluator(expr, *threads->dev);
    }
    template<typename T, int AccessLevel>
    requires tenx::sfinae::is_plain_tensor_v<T>
    auto asEval(const Eigen::TensorBase<T, AccessLevel> &expr) {
        // Unwraps the derived type const T (same as the protected .derived() member in Eigen::TensorBase)
        if constexpr(tenx::sfinae::is_eigen_tensormap_v<T>)
            return *static_cast<const T *>(&expr); // Avoids a copy by returning a tensormap
        else {
            const auto &ref = *static_cast<const T *>(&expr);
            return Eigen::TensorMap<const T>(ref.data(), ref.dimensions());
        }
    }
    template<typename T, int AccessLevel>
    requires tenx::sfinae::is_plain_tensor_v<T>
    auto asEval(Eigen::TensorBase<T, AccessLevel> &expr) {
        // Unwraps the derived type T (same as the protected .derived() member in Eigen::TensorBase)
        if constexpr(tenx::sfinae::is_eigen_tensormap_v<T>)
            return *static_cast<T *>(&expr); // Avoids a copy by returning a tensormap
        else {
            auto &ref = *static_cast<T *>(&expr);
            return Eigen::TensorMap<T>(ref.data(), ref.dimensions());
        }
    }
}