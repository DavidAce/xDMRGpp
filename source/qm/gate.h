#pragma once
// #include "debug/exceptions.h"
#include "math/float.h"
#include "math/tenx.h"
#include <array>
#include <complex>
#include <deque>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace iter {
    enum class order;
}
namespace Eigen {
    template<typename Idx>
    struct IndexPair;
}
enum class GateOp;
namespace qm {

    /* clang-format off */
    class Gate;
    template<typename L>
    [[nodiscard]] std::vector<std::vector<size_t>> get_gate_sequence(const L & layer, bool reverse_odd = true);
    template<iter::order o>
    [[nodiscard]] std::vector<std::vector<size_t>>   get_lightcone(const std::vector<std::vector<qm::Gate>> & layers, size_t pos);
    [[nodiscard]] std::vector<std::vector<size_t>>   get_lightcone_intersection(const std::vector<std::vector<qm::Gate>> & layers, size_t pos_tau, size_t pos_sig);
    [[nodiscard]] std::vector<std::deque<qm::Gate>>  get_lightcone_gate_selection(const std::vector<std::vector<qm::Gate>> & layers,
                                                                                  const std::vector<std::vector<size_t>> & lightcone_intersection, bool reverse_odd = true);
    [[nodiscard]] std::vector<std::deque<qm::Gate>> get_lightcone_gate_selection(const std::vector<std::vector<qm::Gate>> & layers, size_t pos_tau, size_t pos_sig, bool reverse_odd = true);
    [[nodiscard]] std::vector<std::string> get_lightcone_picture(const std::vector<std::vector<qm::Gate>> & layers,
                                                                const std::vector<std::vector<size_t>> & cone, std::string_view tag,
                                                                size_t point_width = 3, std::string_view sep = ",");
    [[nodiscard]] qm::Gate insert(const qm::Gate & middle_gate , const qm::Gate & updown_gate);
    [[nodiscard]] qm::Gate connect(const qm::Gate & dn_gate , const qm::Gate & up_gate);
    [[nodiscard]] qm::Gate trace_idx(const qm::Gate & gate , const std::vector<long> & idx);
    [[nodiscard]] qm::Gate trace_pos(const qm::Gate & gate , const std::vector<size_t> & pos);
    [[nodiscard]] qm::Gate trace_pos(const qm::Gate & gate , size_t pos);
    [[nodiscard]] cx64   trace(const qm::Gate & gate);
    template<auto N>
    [[nodiscard]] qm::Gate trace(const qm::Gate & gate , const std::array<Eigen::IndexPair<Eigen::Index>, N>  & idxpair);


    class Gate {
        public:
        protected:
        template<typename scalar_t, typename alpha_t>
        Eigen::Tensor<scalar_t, 2> exp_internal(const Eigen::Tensor<scalar_t,2> & op_, alpha_t alpha) const;
        enum class Side {L, R};
//        mutable std::optional<std::vector<Eigen::Tensor<cx64,2>>> op_split;
//        mutable std::optional<Eigen::Tensor<cx64,2>> cnj = std::nullopt;
//        mutable std::optional<Eigen::Tensor<cx64,2>> adj = std::nullopt;
//        mutable std::optional<Eigen::Tensor<cx64,2>> trn = std::nullopt;

        public:
        Eigen::Tensor<cx64,2> op;
        Eigen::Tensor<cx128,2> op_t;
        std::vector<size_t> pos;
        std::vector<long> dim;
        Gate() = default;

        template<typename T>
        Gate(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &op_, std::vector<size_t> pos_, std::vector<long> dim_)
            : pos(std::move(pos_)), dim(std::move(dim_)) {
            [[maybe_unused]] auto dim_prod = std::accumulate(std::begin(dim), std::end(dim), 1, std::multiplies<>());
            assert(pos.size() == dim.size());
            auto &threads = tenx::threads::get();
            if constexpr(std::is_same_v<std::decay_t<typename T::Scalar>, cx128>){
                op.resize(tenx::array2{dim_prod, dim_prod});
                op.device(*threads->dev) = op_.eval().unaryExpr([](auto z){return std::complex<fp64>(static_cast<fp64>(z.real()), static_cast<fp64>(z.imag()));});
                op_t.resize(tenx::array2{dim_prod, dim_prod});
                op_t.device(*threads->dev) = op_.eval(); //eval().unaryExpr([](auto z){return std::complex<fp128>(z.real(), z.imag());}); // template .cast<cx128>();
            }else if (std::is_same_v<std::decay_t<typename T::Scalar>, cx64>){
                op.resize(tenx::array2{dim_prod, dim_prod});
                op.device(*threads->dev) = op_.eval();
            }
        }
        template<typename T>
        Gate(const Eigen::EigenBase<T> & op_, std::vector<size_t> pos_, std::vector<long> dim_)
            : Gate(tenx::TensorCast(op_), std::move(pos_), std::move(dim_)) {}

        explicit Gate(const Eigen::Tensor<cx64,2> & op_, std::vector<size_t> pos_, std::vector<long> dim_, cx64 alpha);
        explicit Gate(const Eigen::Tensor<cx64,2> & op_, std::vector<size_t> pos_, std::vector<long> dim_, cx128 alpha);
        explicit Gate(const Eigen::Tensor<cx128,2> & op_, std::vector<size_t> pos_, std::vector<long> dim_, cx64 alpha);
        explicit Gate(const Eigen::Tensor<cx128,2> & op_, std::vector<size_t> pos_, std::vector<long> dim_, cx128 alpha);

        [[nodiscard]] Gate exp(cx64 alpha) const;
        [[nodiscard]] Gate exp(cx128 alpha) const;
        [[nodiscard]] bool isUnitary(double prec = 1e-12) const;
        [[nodiscard]] Eigen::Tensor<cx64,2> conjugate() const;
        [[nodiscard]] Eigen::Tensor<cx64,2> transpose() const;
        [[nodiscard]] Eigen::Tensor<cx64,2> adjoint() const;
        [[nodiscard]] Eigen::Tensor<cx64,2> unaryOp(GateOp unop) const;
        [[nodiscard]] Gate insert(const Gate & other) const;
        [[nodiscard]] Gate connect_above(const Gate & other) const;
        [[nodiscard]] Gate connect_below(const Gate & other) const;
        template<auto N>
        [[nodiscard]] Gate trace(const std::array<Eigen::IndexPair<Eigen::Index>, N> & idxpair) const;
        [[nodiscard]] Gate trace_idx(const std::vector<long> & idx_) const;
        [[nodiscard]] Gate trace_pos(const std::vector<size_t> & pos_) const;
        [[nodiscard]] Gate trace_pos(size_t pos_) const;
        [[nodiscard]] cx64 trace() const;
        template<auto rank, Side s = Side::R>
        [[nodiscard]] std::array<long,rank> shape() const;
        template<auto rank>
        [[nodiscard]] std::array<long, rank> shape(const std::array<size_t,rank/2> & pos_partition) const;
        [[nodiscard]] std::vector<size_t> idx() const;
        [[nodiscard]] std::vector<size_t> idx_up() const;
        [[nodiscard]] std::vector<size_t> idx_dn() const;
        [[nodiscard]] std::vector<size_t> idx(const std::vector<size_t> &pos_) const;
        [[nodiscard]] std::vector<size_t> idx_up(const std::vector<size_t> &pos_) const;
        [[nodiscard]] std::vector<size_t> idx_dn(const std::vector<size_t> &pos_) const;
        [[nodiscard]] bool has_pos(const std::vector<size_t> &pos_) const;
        [[nodiscard]] std::vector<size_t> pos_intersection(const std::vector<size_t> &pos_) const;
        [[nodiscard]] std::vector<size_t> pos_difference(const std::vector<size_t> &pos_) const;
        void draw_pos(std::string & layer_str, std::optional<std::string> layer_tag = std::nullopt) const;
    };
    /* clang-format on */
    struct Rwap;
    struct Swap {
        size_t posL, posR;
        Swap() = default;
        Swap(size_t posL, size_t posR);
        bool operator==(const Rwap &rwap) const;
    };
    struct Rwap {
        size_t posL, posR;
        Rwap() = default;
        Rwap(size_t posL, size_t posR);
        bool operator==(const Swap &swap) const;
    };
    class SwapGate : public Gate {
        using Gate::Gate;

        public:
        std::deque<Swap>       swaps;
        std::deque<Rwap>       rwaps; // swaps sequence and reverse swap sequence
        [[nodiscard]] SwapGate exp(cx64 alpha) const;
        [[nodiscard]] SwapGate exp(cx128 alpha) const;
        void                   generate_swap_sequences();
        size_t                 cancel_swaps(std::deque<Rwap> &other_rwaps);
        size_t                 cancel_rwaps(std::deque<Swap> &other_swaps);
    };

    class MpoGate : public Gate {
        protected:
        std::vector<Eigen::Tensor<cx64, 4>> ops;

        public:
        [[nodiscard]] MpoGate                insert(const MpoGate &other) const;
        [[nodiscard]] MpoGate                insert(const Gate &other) const;
        [[nodiscard]] Eigen::Tensor<cx64, 4> get_op();
        [[nodiscard]] Eigen::Tensor<cx64, 4> get_op(const std::vector<size_t> &pos);
        [[nodiscard]] Gate                   split() const; /*!< Split into individual sites with svd */
    };
}
