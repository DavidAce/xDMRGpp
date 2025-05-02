#pragma once

#include "correlation.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "expectation_value.h"
#include "math/tenx.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"

using tools::finite::measure::RealScalar;

template<typename CalcType, typename Scalar, typename OpType>
CalcType tools::finite::measure::correlation(const StateFinite<Scalar> &state, const Eigen::Tensor<OpType, 2> &op1, const Eigen::Tensor<OpType, 2> &op2,
                                             long pos1, long pos2) {
    if(pos1 == pos2) {
        // Stack the operators
        Eigen::Tensor<OpType, 2> op12 = op1.contract(op2, tenx::idx({0}, {1}));
        LocalObservableOp        ob12 = {op12, pos1};
        return expectation_value<CalcType>(state, std::vector{ob12});
    } else {
        // No need to stack
        LocalObservableOp ob1 = {op1, pos1};
        LocalObservableOp ob2 = {op2, pos2};
        return expectation_value<CalcType>(state, std::vector{ob1, ob2});
    }
}

template<typename CalcType, typename Scalar, typename OpType>
Eigen::Tensor<CalcType, 2> tools::finite::measure::correlation_matrix(const StateFinite<Scalar> &state, const Eigen::Tensor<OpType, 2> &op1,
                                                                      const Eigen::Tensor<OpType, 2> &op2) {
    if constexpr(settings::debug) tools::log->trace("Measuring correlation matrix");

    long                       len = state.template get_length<long>();
    bool                       eq  = tenx::MatrixMap(op1) == tenx::MatrixMap(op2);
    Eigen::Tensor<CalcType, 2> C(len, len);
    C.setZero();

    for(long pos_j = 0; pos_j < len; pos_j++) {
        for(long pos_i = pos_j; pos_i < len; pos_i++) {
            C(pos_i, pos_j) = correlation<CalcType>(state, op1, op2, pos_i, pos_j);
            if(eq)
                C(pos_j, pos_i) = C(pos_i, pos_j);
            else
                C(pos_j, pos_i) = correlation<CalcType>(state, op1, op2, pos_j, pos_i);
        }
    }
    return C;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::structure_factor(const StateFinite<Scalar> &state, const Eigen::Tensor<Scalar, 2> &correlation_matrix) {
    tools::log->trace("Measuring structure factor");
    if(correlation_matrix.dimension(0) != correlation_matrix.dimension(1))
        throw except::logic_error("Correlation matrix is not square: dims {}", correlation_matrix.dimensions());
    if(correlation_matrix.dimension(0) != state.template get_length<long>())
        throw except::logic_error("Expected correlation matrix of size {}. Got {}", state.template get_length<long>(), correlation_matrix.dimension(0));
    return tenx::MatrixMap(correlation_matrix).cwiseAbs2().colwise().sum().sum() / state.template get_length<RealScalar<Scalar>>();
}
