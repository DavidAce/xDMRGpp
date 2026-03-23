#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "debug/exceptions.h"
#include "qm/gate.h"
#include "qm/spin.h"
#include <cmath>

namespace {
    constexpr double tol = 1e-12;

    void require_tensor_close(const Eigen::Tensor<cx64, 2> &actual, const Eigen::Tensor<cx64, 2> &expected, double prec = tol) {
        REQUIRE(actual.dimension(0) == expected.dimension(0));
        REQUIRE(actual.dimension(1) == expected.dimension(1));
        for(long row = 0; row < actual.dimension(0); ++row) {
            for(long col = 0; col < actual.dimension(1); ++col) REQUIRE(std::abs(actual(row, col) - expected(row, col)) < prec);
        }
    }

    void require_vector_close(const Eigen::VectorXcd &actual, const Eigen::VectorXcd &expected, double prec = tol) {
        REQUIRE(actual.size() == expected.size());
        for(long idx = 0; idx < actual.size(); ++idx) REQUIRE(std::abs(actual(idx) - expected(idx)) < prec);
    }

    void require_tensor_vector_close(const Eigen::Tensor<cx64, 1> &actual, const Eigen::VectorXcd &expected, double prec = tol) {
        REQUIRE(actual.dimension(0) == expected.size());
        for(long idx = 0; idx < expected.size(); ++idx) REQUIRE(std::abs(actual(idx) - expected(idx)) < prec);
    }
}

TEST_CASE("Trace gate positions by physical labels", "[qm][gate][trace_pos]") {
    Eigen::Matrix4cd op;
    for(long row = 0; row < op.rows(); ++row)
        for(long col = 0; col < op.cols(); ++col) op(row, col) = cx64{static_cast<double>(row * 4 + col + 1), static_cast<double>(row - col)};

    qm::Gate gate(op, {3ul, 7ul}, {2l, 2l});
    auto     rank4          = gate.op.reshape(tenx::array4{2, 2, 2, 2});
    auto     expected_right = rank4.trace(Eigen::array<Eigen::Index, 2>{1, 3}).reshape(tenx::array2{2, 2});
    auto     expected_left  = rank4.trace(Eigen::array<Eigen::Index, 2>{0, 2}).reshape(tenx::array2{2, 2});

    auto traced_right = gate.trace_pos(7);
    REQUIRE(traced_right.pos == std::vector<size_t>{3ul});
    REQUIRE(traced_right.dim == std::vector<long>{2l});
    require_tensor_close(traced_right.op, expected_right);

    auto traced_left = gate.trace_pos(3);
    REQUIRE(traced_left.pos == std::vector<size_t>{7ul});
    REQUIRE(traced_left.dim == std::vector<long>{2l});
    require_tensor_close(traced_left.op, expected_left);

    REQUIRE_THROWS_AS(gate.trace_pos(11), except::logic_error);
}

TEST_CASE("Tracing all positions is order-independent", "[qm][gate][trace]") {
    Eigen::Matrix4cd op;
    for(long row = 0; row < op.rows(); ++row)
        for(long col = 0; col < op.cols(); ++col) op(row, col) = cx64{static_cast<double>(2 * (row * 4 + col + 1)), static_cast<double>(col - row)};

    qm::Gate gate(op, {5ul, 9ul}, {2l, 2l});

    auto traced_forward  = gate.trace_pos(std::vector<size_t>{5ul, 9ul});
    auto traced_backward = gate.trace_pos(std::vector<size_t>{9ul, 5ul});

    REQUIRE(traced_forward.pos.empty());
    REQUIRE(traced_backward.pos.empty());
    REQUIRE(traced_forward.dim.empty());
    REQUIRE(traced_backward.dim.empty());
    REQUIRE(std::abs(traced_forward.op(0, 0) - traced_backward.op(0, 0)) < tol);
    REQUIRE(std::abs(gate.trace() - traced_forward.op(0, 0)) < tol);
}

TEST_CASE("Half-spin z spinors stay normalized and consistent", "[qm][spin][half]") {
    Eigen::Vector2cd up_expected;
    up_expected << cx64{1.0, 0.0}, cx64{0.0, 0.0};
    Eigen::Vector2cd down_expected;
    down_expected << cx64{0.0, 0.0}, cx64{1.0, 0.0};

    auto up_global   = qm::spin::half::get_spinor("z");
    auto down_global = qm::spin::half::get_spinor("-z");
    auto up_matrix   = qm::spin::half::matrix::sz_spinors[0];
    auto down_matrix = qm::spin::half::matrix::sz_spinors[1];
    auto up_tensor   = qm::spin::half::tensor::get_spinor("z");
    auto down_tensor = qm::spin::half::tensor::get_spinor("-z");

    require_vector_close(up_global, up_expected);
    require_vector_close(down_global, down_expected);
    require_vector_close(up_matrix, up_expected);
    require_vector_close(down_matrix, down_expected);
    require_tensor_vector_close(up_tensor, up_expected);
    require_tensor_vector_close(down_tensor, down_expected);

    REQUIRE(std::abs(up_global.squaredNorm() - 1.0) < tol);
    REQUIRE(std::abs(down_global.squaredNorm() - 1.0) < tol);
}

TEST_CASE("Half-spin axis parsing and pauli accessors are stable", "[qm][spin][axis]") {
    REQUIRE(qm::spin::half::is_valid_axis("z"));
    REQUIRE(qm::spin::half::is_valid_axis("-x"));
    REQUIRE_FALSE(qm::spin::half::is_valid_axis("bad"));

    REQUIRE(qm::spin::half::get_sign("+z") == 1);
    REQUIRE(qm::spin::half::get_sign("-z") == -1);
    REQUIRE(qm::spin::half::get_sign("z") == 0);
    REQUIRE(qm::spin::half::get_axis_unsigned("-z") == "z");
    REQUIRE(qm::spin::half::get_axis_unsigned("id") == "i");

    auto pauli_i = qm::spin::half::get_pauli("i");
    REQUIRE(pauli_i.rows() == 2);
    REQUIRE(pauli_i.cols() == 2);
    REQUIRE(std::abs(pauli_i(0, 0) - cx64{1.0, 0.0}) < tol);
    REQUIRE(std::abs(pauli_i(1, 1) - cx64{1.0, 0.0}) < tol);
    REQUIRE(std::abs(pauli_i(0, 1)) < tol);
    REQUIRE(std::abs(pauli_i(1, 0)) < tol);

    REQUIRE_THROWS_AS(qm::spin::half::get_axis_unsigned("bad"), except::runtime_error);
    REQUIRE_THROWS_AS(qm::spin::half::get_spinor("bad"), except::runtime_error);
}

TEST_CASE("Embedded spin operators match the dedicated two-body helpers", "[qm][spin][embed]") {
    auto embedded_left  = qm::spin::half::gen_embedded_spin_half_operator(qm::spin::half::sz, 0, 2);
    auto embedded_right = qm::spin::half::gen_embedded_spin_half_operator(qm::spin::half::sz, 1, 2);
    auto twobody        = qm::spin::half::gen_twobody_spins(qm::spin::half::sz);

    REQUIRE((embedded_left - twobody.at(0)).norm() < tol);
    REQUIRE((embedded_right - twobody.at(1)).norm() < tol);
}

int main(int argc, char **argv) {
    Catch::Session session;
    int            returnCode = session.applyCommandLine(argc, argv);
    if(returnCode != 0) return returnCode;
    return session.run();
}
