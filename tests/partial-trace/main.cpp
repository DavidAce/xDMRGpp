#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "math/linalg.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include <io/spdlog.h>

template<typename A, typename B>
bool equal(const A &a, const B &b, double prec = 1e-8) {
    if(a.size() != b.size()) return false;
    for(size_t i = 0; i < static_cast<size_t>(a.size()); i++) {
        //        fmt::print("{}: {} =? {}\n",i, a.data()[i] , b.data()[i]);
        if(std::abs(a.data()[i] - b.data()[i]) > prec) return false;
    }
    return true;
}

TEST_CASE("Compare matrix and tensor kronecker products", "[kronecker]") {
    SECTION("Rank 4") {
        Eigen::MatrixXcd id = qm::spin::half::id;
        Eigen::MatrixXcd m(2, 2);
        for(long i = 0; i < m.size(); i++) m(i) = static_cast<double>(i + 1);
        m(1, 0) = -2;
        m(0, 1) = std::complex<double>(-2.3, -10);

        auto matrix_m_id  = linalg::matrix::kronecker(m, id);
        auto matrix_id_m  = linalg::matrix::kronecker(id, m);
        auto tensor4_m_id = linalg::tensor::kronecker(m, id);
        auto tensor4_id_m = linalg::tensor::kronecker(id, m);
        //        fmt::print("matrix_m_id\n{}\n", linalg::matrix::to_string(matrix_m_id));
        //        fmt::print("matrix_id_m\n{}\n", linalg::matrix::to_string(matrix_id_m));
        //        fmt::print("tensor4_m_id\n{}\n", linalg::tensor::to_string(tensor4_m_id));
        //        fmt::print("tensor4_id_m\n{}\n", linalg::tensor::to_string(tensor4_id_m));
        REQUIRE(equal(matrix_m_id, tensor4_id_m));
        REQUIRE(equal(matrix_id_m, tensor4_m_id));
    }
    SECTION("Rank 6") {
        Eigen::MatrixXcd id = qm::spin::half::id;
        Eigen::MatrixXcd m(2, 2);
        for(long i = 0; i < m.size(); i++) m(i) = static_cast<double>(i + 1);
        m(1, 0) = -2;
        m(0, 1) = std::complex<double>(-2.3, -10);

        auto matrix_m_id_id  = linalg::matrix::kronecker(linalg::matrix::kronecker(m, id), id);
        auto matrix_id_id_m  = linalg::matrix::kronecker(linalg::matrix::kronecker(id, id), m);
        auto tensor6_m_id_id = linalg::tensor::kronecker(linalg::tensor::kronecker(m, id), id);
        auto tensor6_id_id_m = linalg::tensor::kronecker(linalg::tensor::kronecker(id, id), m);
        //        fmt::print("matrix_m_id_id\n{}\n", linalg::matrix::to_string(matrix_m_id_id));
        //        fmt::print("matrix_id_id_m\n{}\n", linalg::matrix::to_string(matrix_id_id_m));
        //        fmt::print("tensor6_m_id_id\n{}\n", linalg::tensor::to_string(tensor6_m_id_id));
        //        fmt::print("tensor6_id_id_m\n{}\n", linalg::tensor::to_string(tensor6_id_id_m));
        REQUIRE(equal(matrix_m_id_id, tensor6_id_id_m));
        REQUIRE(equal(matrix_id_id_m, tensor6_m_id_id));
    }
}

TEST_CASE("Test partial trace of tensors", "[partial trace]") {
    SECTION("Rank 4") {
        Eigen::MatrixXcd id = qm::spin::half::id;
        Eigen::MatrixXcd m(2, 2);
        for(long i = 0; i < m.size(); i++) m(i) = static_cast<double>(i + 1);
        m(1, 0) = -2;
        m(0, 1) = std::complex<double>(-2.3, -10);

        // Compute the traces
        auto tr_id = id.trace();
        auto tr_m  = m.trace();
        // Define the expected results
        Eigen::MatrixXcd matrix_m  = m * tr_id;
        Eigen::MatrixXcd matrix_id = id * tr_m;
        // Define the tensors to trace
        auto tensor4_m_id = linalg::tensor::kronecker(m, id);
        auto tensor4_id_m = linalg::tensor::kronecker(id, m);
        // Do the tracing
        auto tensor2_m  = linalg::tensor::trace(tensor4_m_id, tenx::idx({1}, {3}));
        auto tensor2_id = linalg::tensor::trace(tensor4_m_id, tenx::idx({0}, {2}));

        REQUIRE(equal(matrix_m, tensor2_m));
        REQUIRE(equal(matrix_id, tensor2_id));

        // Now do the trace once more to get a scalar
        auto trace_m_a  = linalg::tensor::trace(tensor2_m, tenx::idx({0}, {1}));
        auto trace_id_a = linalg::tensor::trace(tensor2_id, tenx::idx({0}, {1}));
        REQUIRE(trace_m_a(0) == tr_id * tr_m);
        REQUIRE(trace_id_a(0) == tr_id * tr_m);

        // Trace directly down to scalar
        auto trace_m_b  = linalg::tensor::trace(tensor4_m_id, tenx::idx({0, 1}, {2, 3}));
        auto trace_id_b = linalg::tensor::trace(tensor4_id_m, tenx::idx({0, 1}, {2, 3}));
        REQUIRE(trace_m_b(0) == tr_id * tr_m);
        REQUIRE(trace_id_b(0) == tr_id * tr_m);
    }
    SECTION("Rank 6") {
        Eigen::MatrixXcd id = qm::spin::half::id;
        Eigen::MatrixXcd m(2, 2);
        for(long i = 0; i < m.size(); i++) m(i) = static_cast<double>(i + 1);
        m(1, 0) = -2;
        m(0, 1) = std::complex<double>(-2.3, -10);

        // Compute the traces
        auto tr_id = id.trace();
        auto tr_m  = m.trace();
        // Define the expected results
        Eigen::MatrixXcd matrix_m    = m * tr_id * tr_id;
        Eigen::MatrixXcd matrix_id   = id * tr_m * tr_id;
        Eigen::MatrixXcd matrix_m_id = linalg::matrix::kronecker(id, m) * tr_id;
        Eigen::MatrixXcd matrix_id_m = linalg::matrix::kronecker(m, id) * tr_id;

        // Define the tensors to trace
        auto tensor6_m_id_id = linalg::tensor::kronecker(linalg::tensor::kronecker(m, id), id);
        auto tensor6_id_m_id = linalg::tensor::kronecker(linalg::tensor::kronecker(id, m), id);
        auto tensor6_id_id_m = linalg::tensor::kronecker(linalg::tensor::kronecker(id, id), m);
        // Do the tracing one step
        auto tensor4_m_id_a = linalg::tensor::trace(tensor6_m_id_id, tenx::idx({2}, {5}));
        auto tensor4_m_id_b = linalg::tensor::trace(tensor6_id_m_id, tenx::idx({0}, {3}));
        auto tensor4_id_m_a = linalg::tensor::trace(tensor6_id_m_id, tenx::idx({2}, {5}));
        auto tensor4_id_m_b = linalg::tensor::trace(tensor6_id_id_m, tenx::idx({0}, {3}));

        REQUIRE(equal(tensor4_m_id_a, tensor4_m_id_b));
        REQUIRE(equal(tensor4_id_m_a, tensor4_id_m_b));
        REQUIRE(equal(tensor4_m_id_a, matrix_m_id));
        REQUIRE(equal(tensor4_id_m_a, matrix_id_m));

        // Do the tracing once more
        auto tensor2_m_a  = linalg::tensor::trace(tensor4_m_id_a, tenx::idx({1}, {3}));
        auto tensor2_m_b  = linalg::tensor::trace(tensor4_id_m_a, tenx::idx({0}, {2}));
        auto tensor2_id_a = linalg::tensor::trace(tensor4_m_id_a, tenx::idx({0}, {2}));
        auto tensor2_id_b = linalg::tensor::trace(tensor4_id_m_a, tenx::idx({1}, {3}));

        REQUIRE(equal(tensor2_m_a, tensor2_m_b));
        REQUIRE(equal(tensor2_id_a, tensor2_id_b));
        REQUIRE(equal(tensor2_m_a, matrix_m));
        REQUIRE(equal(tensor2_id_a, matrix_id));

        // Do the tracing in two steps directly
        auto tensor2_m_c = linalg::tensor::trace(tensor6_m_id_id, tenx::idx({1, 2}, {4, 5}));
        auto tensor2_m_d = linalg::tensor::trace(tensor6_id_m_id, tenx::idx({0, 2}, {3, 5}));
        auto tensor2_m_e = linalg::tensor::trace(tensor6_id_id_m, tenx::idx({0, 1}, {3, 4}));

        REQUIRE(equal(tensor2_m_a, tensor2_m_c));
        REQUIRE(equal(tensor2_m_a, tensor2_m_d));
        REQUIRE(equal(tensor2_m_a, tensor2_m_e));
    }
}

int main(int argc, char **argv) {
    Catch::Session session; // There must be exactly one instance
    int            returnCode = session.applyCommandLine(argc, argv);
    if(returnCode != 0) // Indicates a command line error
        return returnCode;
    // session.configData().showSuccessfulTests = false;
    // session.configData().reporterName = "compact";
    return session.run();
}
