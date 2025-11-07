
#include "math/float_eigen.h"

#include <Eigen/QR>
int main() {
    using MatrixXcq = Eigen::Matrix<cx128, Eigen::Dynamic, Eigen::Dynamic>;
    MatrixXcq matrix1(10, 10);
    matrix1.setRandom();
    auto cpqr = Eigen::ColPivHouseholderQR<MatrixXcq>(matrix1);

    MatrixXcq matrix2 = cpqr.householderQ().setLength(10) * MatrixXcq::Identity(10, 10);

    return 0;
}
