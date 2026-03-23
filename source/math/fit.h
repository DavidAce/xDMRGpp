#pragma once
#include "math/cast.h"
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <vector>

namespace fit {
    namespace internal {

        // Generic functor
        template<typename Scalar_, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
        struct Functor {
            using Scalar = Scalar_;
            enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
            using InputType    = Eigen::Matrix<Scalar, InputsAtCompileTime, 1>;
            using ValueType    = Eigen::Matrix<Scalar, ValuesAtCompileTime, 1>;
            using JacobianType = Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>;

            int m_inputs, m_values;

            Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
            Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
            virtual int inputs() const { return m_inputs; }
            virtual int values() const { return m_values; }
        };
        struct log_stretched_functor : Functor<double> {
            Eigen::VectorXd X, Y;
            template<typename TX, typename TY>
            log_stretched_functor(const TX &xvec, const TY &yvec) : Functor(3, safe_cast<int>(xvec.size())) {
                if(static_cast<size_t>(xvec.size()) != static_cast<size_t>(yvec.size()))
                    throw std::runtime_error("fit::log_stretched: xvec and yvec size mismatch");
                X = Eigen::Map<const Eigen::VectorXd>(xvec.data(), safe_cast<Eigen::Index>(xvec.size()));
                Y = Eigen::Map<const Eigen::VectorXd>(yvec.data(), safe_cast<Eigen::Index>(yvec.size()));
            }
            int operator()(const Eigen::VectorXd &C, Eigen::VectorXd &F) const {
                // Fit xy-data to the log of a stretched exponential y(x) = log(a) - (x/b)  ** c
                // Here, "a, b, c"  correspond to C(0), C(1) and C(2).
                // In the operator() we provide the difference between the datapoints and function f = y - y(x)
                // where y and x are the given datapoints. Then, the LevenbergMarquardt minimizes sum_x f(x)².
                for(Eigen::Index i = 0; i < F.size(); ++i) { F[i] = Y[i] - (std::log(C(0)) - std::pow(this->X[i] / C(1), C(2))); }
                return 0;
            }
        };
        struct exp_stretched_functor : Functor<double> {
            Eigen::VectorXd X, Y;
            template<typename TX, typename TY>
            exp_stretched_functor(const TX &xvec, const TY &yvec) : Functor(3, safe_cast<int>(xvec.size())) {
                if(xvec.size() != yvec.size()) throw std::runtime_error("fit::log_stretched: xvec and yvec size mismatch");
                X = Eigen::Map<const Eigen::VectorXd>(xvec.data(), static_cast<Eigen::Index>(xvec.size()));
                Y = Eigen::Map<const Eigen::VectorXd>(yvec.data(), static_cast<Eigen::Index>(yvec.size()));
            }
            int operator()(const Eigen::VectorXd &C, Eigen::VectorXd &D) const {
                // Fit xy-data to a stretched exponential f(x) = a * exp(-(x/b)  ** c)
                // Here, "a, b, c"  correspond to C(0), C(1) and C(2).
                // In the operator() we provide the difference between the datapoints and function f = y - y(x)
                // where y and x are the given datapoints. Then, the LevenbergMarquardt minimizes sum_x f(x)².
                for(Eigen::Index i = 0; i < D.size(); ++i) { D[i] = Y[i] - (C(0) * std::exp(-std::pow(this->X[i] / C(1), C(2)))); }
                return 0;
            }
        };
        struct exp_functor : Functor<double> {
            Eigen::VectorXd X, Y;
            template<typename TX, typename TY>
            exp_functor(const TX &xvec, const TY &yvec) : Functor(2, safe_cast<int>(xvec.size())) {
                if(xvec.size() != yvec.size()) throw std::runtime_error("fit::log_stretched: xvec and yvec size mismatch");
                X = Eigen::Map<const Eigen::VectorXd>(xvec.data(), static_cast<Eigen::Index>(xvec.size()));
                Y = Eigen::Map<const Eigen::VectorXd>(yvec.data(), static_cast<Eigen::Index>(yvec.size()));
            }
            int operator()(const Eigen::VectorXd &C, Eigen::VectorXd &D) const {
                // Fit xy-data to an exponential f(x) = a * exp(-x/b)
                // Here, "a, b, c"  correspond to C(0), C(1) and C(2).
                // In the operator() we provide the difference between the datapoints and function f = y - y(x)
                // where y and x are the given datapoints. Then, the LevenbergMarquardt minimizes sum_x f(x)².
                for(Eigen::Index i = 0; i < D.size(); ++i) { D[i] = Y[i] - (C(0) * std::exp(-(this->X[i] / C(1)))); }
                return 0;
            }
        };
        struct log_functor : Functor<double> {
            Eigen::VectorXd X, Y;
            template<typename TX, typename TY>
            log_functor(const TX &xvec, const TY &yvec) : Functor(2, safe_cast<int>(xvec.size())) {
                if(xvec.size() != yvec.size()) throw std::runtime_error("fit::log_stretched: xvec and yvec size mismatch");
                X = Eigen::Map<const Eigen::VectorXd>(xvec.data(), static_cast<Eigen::Index>(xvec.size()));
                Y = Eigen::Map<const Eigen::VectorXd>(yvec.data(), static_cast<Eigen::Index>(yvec.size()));
            }
            int operator()(const Eigen::VectorXd &C, Eigen::VectorXd &D) const {
                // Fit xy-data to a log f(x) = log(a) - x/b
                // Here, "a, b, c"  correspond to C(0), C(1) and C(2).
                // In the operator() we provide the difference between the datapoints and function f = y - y(x)
                // where y and x are the given datapoints. Then, the LevenbergMarquardt minimizes sum_x f(x)².
                for(Eigen::Index i = 0; i < D.size(); ++i) { D[i] = Y[i] - (std::log(C(0)) - this->X[i] / C(1)); }
                return 0;
            }
        };
    }
    template<typename T>
    struct Result {
        std::vector<T>                         coeffs;
        Eigen::LevenbergMarquardtSpace::Status status;
    };
    template<typename TX, typename TY>
    Result<double> log_stretched(const TX &X, const TY &Y, const std::vector<double> &guess) {
        if(guess.size() != 3) throw std::runtime_error("coeffs must have 3 elements");
        auto functor              = Eigen::NumericalDiff<internal::log_stretched_functor>(X, Y);
        auto levmarq              = Eigen::LevenbergMarquardt<Eigen::NumericalDiff<internal::log_stretched_functor>>(functor);
        levmarq.parameters.maxfev = 1000;
        auto mapstdv              = Eigen::Map<const Eigen::VectorXd>(guess.data(), static_cast<Eigen::Index>(guess.size()));
        auto coeffsv              = Eigen::VectorXd(mapstdv);
        auto status               = levmarq.minimize(coeffsv);
        return {std::vector<double>(std::begin(coeffsv), std::end(coeffsv)), status};
    }

    template<typename TX, typename TY>
    Result<double> exp_stretched(const TX &X, const TY &Y, const std::vector<double> &guess) {
        if(guess.size() != 3) throw std::runtime_error("coeffs must have 3 elements");
        auto functor = Eigen::NumericalDiff<internal::exp_stretched_functor>(X, Y);
        auto levmarq = Eigen::LevenbergMarquardt<Eigen::NumericalDiff<internal::exp_stretched_functor>>(functor);
        auto mapstdv = Eigen::Map<const Eigen::VectorXd>(guess.data(), static_cast<Eigen::Index>(guess.size()));
        auto coeffsv = Eigen::VectorXd(mapstdv);
        auto status  = levmarq.minimize(coeffsv);
        return {std::vector<double>(std::begin(coeffsv), std::end(coeffsv)), status};
    }
    template<typename TX, typename TY>
    Result<double> exp(const TX &X, const TY &Y, const std::vector<double> &guess) {
        if(guess.size() != 2) throw std::runtime_error("coeffs must have 2 elements");
        auto functor = Eigen::NumericalDiff<internal::exp_functor>(X, Y);
        auto levmarq = Eigen::LevenbergMarquardt<Eigen::NumericalDiff<internal::exp_functor>>(functor);
        auto mapstdv = Eigen::Map<const Eigen::VectorXd>(guess.data(), static_cast<Eigen::Index>(guess.size()));
        auto coeffsv = Eigen::VectorXd(mapstdv);
        auto status  = levmarq.minimize(coeffsv);
        return {std::vector<double>(std::begin(coeffsv), std::end(coeffsv)), status};
    }
    template<typename TX, typename TY>
    Result<double> log(const TX &X, const TY &Y, const std::vector<double> &guess) {
        if(guess.size() != 2) throw std::runtime_error("coeffs must have 2 elements");
        auto functor = Eigen::NumericalDiff<internal::log_functor>(X, Y);
        auto levmarq = Eigen::LevenbergMarquardt<Eigen::NumericalDiff<internal::log_functor>>(functor);
        auto mapstdv = Eigen::Map<const Eigen::VectorXd>(guess.data(), static_cast<Eigen::Index>(guess.size()));
        auto coeffsv = Eigen::VectorXd(mapstdv);
        auto status  = levmarq.minimize(coeffsv);
        return {std::vector<double>(std::begin(coeffsv), std::end(coeffsv)), status};
    }

    std::vector<double> polyfit(const std::vector<double> &x, const std::vector<double> &y, size_t order);
}
