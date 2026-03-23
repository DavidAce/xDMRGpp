#include "math/cast.h"
#include <Eigen/QR>
#include <cmath>
#include <string_view>
#include <vector>

namespace fit {
    std::vector<double> polyfit(const std::vector<double> &x, const std::vector<double> &y, size_t order) {
        // check to make sure inputs are correct
        if(x.size() != y.size()) throw std::runtime_error("x.size() " + std::to_string(x.size()) + " != y.size() " + std::to_string(x.size()));
        if(x.size() < order + 1) throw std::runtime_error("x.size() " + std::to_string(x.size()) + " < order + 1 (order = " + std::to_string(order) + ")");

        // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
        Eigen::VectorXd Y = Eigen::VectorXd::Map(y.data(), safe_cast<long>(y.size()));
        Eigen::MatrixXd X(x.size(), order + 1);

        // Populate the X matrix
        for(Eigen::Index row = 0; row < X.rows(); ++row) {
            double xp = 1.0;
            for(Eigen::Index col = 0; col < X.cols(); ++col) {
                X(row, col) = xp;
                xp *= x[safe_cast<size_t>(row)];
            }
        }

        // Allocate for the results
        std::vector<double> coeff(order + 1, 0);
        auto                coeff_map = Eigen::VectorXd::Map(coeff.data(), safe_cast<long>(coeff.size()));
        // Solve for linear least square fit
        coeff_map = X.householderQr().solve(Y);
        return coeff;
    }
}
