#define ANKERL_NANOBENCH_IMPLEMENT
#include "env/environment.h"
#include "math/float.h"
#include "math/linalg/tensor.h"
#include "math/tenx.h"
#include "nanobench.h"
#include <fmt/core.h>
#include <string_view>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename ScalarL, typename ScalarR, auto rank>
auto asDiagonalProduct1(const Eigen::Tensor<ScalarL, 1> &L, Eigen::Tensor<ScalarR, rank> M, long Mdim) {
    assert(M.dimension(dim) == L.size());
    assert(dim < rank);
    for(long i = 0; i < L.size(); ++i) { M.chip(i, Mdim) = M.chip(i, Mdim) * static_cast<ScalarR>(L(i)); }
    return M;
}

template<typename ScalarL, typename ScalarR, auto rank>
auto asDiagonalProduct2(const Eigen::Tensor<ScalarL, 1> &L, Eigen::Tensor<ScalarR, rank> M, long Mdim) {
    assert(M.dimension(dim) == L.size());
    assert(dim < rank);
    for(long i = 0; i < L.size(); ++i) {
        M.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarR { return v * static_cast<ScalarR>(L[i]); });
    }
    return M;
}

template<typename ScalarL, typename ScalarR, auto rank>
auto asDiagonalProduct3(const Eigen::Tensor<ScalarL, 1> &L, Eigen::Tensor<ScalarR, rank> M, long Mdim) {
    assert(M.dimension(dim) == L.size());
    assert(dim < rank);
    for(long i = 0; i < L.size(); ++i) {
        M.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarR { return v * static_cast<ScalarR>(L[i]); });
    }
    return M;
}

template<typename ScalarL, typename ScalarR, auto rank>
auto asDiagonalProduct2omp(const Eigen::Tensor<ScalarL, 1> &L, Eigen::Tensor<ScalarR, rank> M, long Mdim) {
    assert(M.dimension(dim) == L.size());
    assert(dim < rank);
#pragma omp parallel for
    for(long i = 0; i < L.size(); ++i) {
        M.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarR { return v * static_cast<ScalarR>(L[i]); });
    }
    return M;
}

template<typename ScalarL, typename ScalarR, auto rank>
auto asDiagonalProduct3omp(const Eigen::Tensor<ScalarL, 1> &L, Eigen::Tensor<ScalarR, rank> M, long Mdim) {
    assert(M.dimension(dim) == L.size());
    assert(dim < rank);
#pragma omp parallel for
    for(long i = 0; i < L.size(); ++i) {
        M.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarR { return v * static_cast<ScalarR>(L[i]); });
    }
    return M;
}

template<typename ScalarL, typename ScalarR, auto rank>
void asDiagonalProduct3alloc(const Eigen::Tensor<ScalarL, 1> &L, const Eigen::Tensor<ScalarR, rank> &M, long Mdim, Eigen::Tensor<ScalarR, rank> &R) {
    assert(M.dimension(dim) == L.size());
    assert(dim < rank);
    R.resize(M.dimensions());
    for(long i = 0; i < L.size(); ++i) {
        R.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarR { return v * static_cast<ScalarR>(L[i]); });
    }
}

template<typename TensorL, typename TensorM, typename TensorR>
auto asDiagonalContractMap(const TensorL &L, const TensorM &M, long Mdim, TensorR &R) {
    using ScalarL = typename TensorL::Scalar;
    using ScalarM = typename TensorM::Scalar;
    using ScalarR = typename TensorR::Scalar;
    static_assert(std::is_convertible_v<ScalarL, ScalarM>);
    static_assert(std::is_convertible_v<ScalarM, ScalarR>);
    static_assert(TensorM::NumDimensions == TensorR::NumDimensions);

    assert(M.dimension(dim) == L.size());
    assert(dim < rank);
    assert(R.dimesions() == M.dimensions());
    for(long i = 0; i < L.size(); ++i) {
        R.chip(i, Mdim) = M.chip(i, Mdim).unaryExpr([&](const auto &v) -> ScalarM { return v * static_cast<ScalarM>(L(i)); }).template cast<ScalarR>();
    }
}

int main() {
    fmt::print("Compiler flags {}", env::build::compiler_flags);

    Eigen::Tensor<cx64, 3> M(2, 512, 512);
    Eigen::Tensor<cx64, 1> L(512);
    Eigen::Tensor<cx64, 3> R_alloc(2, 512, 512);
    M.setRandom();
    L.setRandom();
    Eigen::Tensor<cx64, 3> R1_check = tenx::asDiagonal(L).contract(M, tenx::idx({1}, {1})).shuffle(std::array<long, 3>{1, 0, 2});
    Eigen::Tensor<cx64, 3> R2_check = M.contract(tenx::asDiagonal(L), tenx::idx({2}, {0}));

    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonal", [&] {
        Eigen::Tensor<cx64, 3> R = tenx::asDiagonal(L).contract(M, tenx::idx({1}, {1})).shuffle(std::array<long, 3>{1, 0, 2});
        ankerl::nanobench::doNotOptimizeAway(R);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R_check),1e-12)) throw std::runtime_error("asDiagonal failed");
    });

    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonalProduct1", [&] {
        Eigen::Tensor<cx64, 3> R = asDiagonalProduct1(L, M, 1);
        ankerl::nanobench::doNotOptimizeAway(R);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R_check),1e-12)) throw std::runtime_error("asDiagonalProduct1 failed");
    });

    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonalProduct2", [&] {
        Eigen::Tensor<cx64, 3> R = asDiagonalProduct2(L, M, 1);
        ankerl::nanobench::doNotOptimizeAway(R);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R_check),1e-12)) {
        //     fmt::print("R \n{}\n", linalg::tensor::to_string(R,16));
        //     fmt::print("Rcheck \n{}\n", linalg::tensor::to_string(R_check,16));
        //     throw std::runtime_error("asDiagonalProduct2 failed");
        // }
    });
    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonalProduct3", [&] {
        Eigen::Tensor<cx64, 3> R = asDiagonalProduct3(L, M, 2);
        ankerl::nanobench::doNotOptimizeAway(R);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R2_check),1e-12)) {
        //     fmt::print("R \n{}\n", linalg::tensor::to_string(R,16));
        //     fmt::print("Rcheck \n{}\n", linalg::tensor::to_string(R2_check,16));
        //     throw std::runtime_error("asDiagonalProduct3 failed");
        // }
    });
    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonalProduct2omp", [&] {
        Eigen::Tensor<cx64, 3> R = asDiagonalProduct2omp(L, M, 1);
        ankerl::nanobench::doNotOptimizeAway(R);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R_check),1e-12)) {
        //     fmt::print("R \n{}\n", linalg::tensor::to_string(R,16));
        //     fmt::print("Rcheck \n{}\n", linalg::tensor::to_string(R_check,16));
        //     throw std::runtime_error("asDiagonalProduct2 failed");
        // }
    });
    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonalProduct3omp", [&] {
        Eigen::Tensor<cx64, 3> R = asDiagonalProduct3omp(L, M, 2);
        ankerl::nanobench::doNotOptimizeAway(R);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R2_check),1e-12)) {
        //     fmt::print("R \n{}\n", linalg::tensor::to_string(R,16));
        //     fmt::print("Rcheck \n{}\n", linalg::tensor::to_string(R2_check,16));
        //     throw std::runtime_error("asDiagonalProduct3 failed");
        // }
    });
    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonalProduct3alloc", [&] {
        asDiagonalProduct3alloc(L, M, 2, R_alloc);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R2_check),1e-12)) {
        //     fmt::print("R \n{}\n", linalg::tensor::to_string(R,16));
        //     fmt::print("Rcheck \n{}\n", linalg::tensor::to_string(R2_check,16));
        //     throw std::runtime_error("asDiagonalProduct3 failed");
        // }
    });
    ankerl::nanobench::Bench().minEpochIterations(10).run("cx64 asDiagonalContractMap", [&] {
        auto Lmap = Eigen::TensorMap<Eigen::Tensor<cx64,1>>(L.data(), L.size());
        auto Mmap = Eigen::TensorMap<Eigen::Tensor<cx64,3>>(M.data(), M.dimensions());
        auto Rmap = Eigen::TensorMap<Eigen::Tensor<cx64,3>>(R_alloc.data(), R_alloc.dimensions());
        asDiagonalContractMap(Lmap, Mmap, 2, Rmap);
        // if(!tenx::VectorMap(R).isApprox(tenx::VectorMap(R2_check),1e-12)) {
        //     fmt::print("R \n{}\n", linalg::tensor::to_string(R,16));
        //     fmt::print("Rcheck \n{}\n", linalg::tensor::to_string(R2_check,16));
        //     throw std::runtime_error("asDiagonalProduct3 failed");
        // }
    });
}
