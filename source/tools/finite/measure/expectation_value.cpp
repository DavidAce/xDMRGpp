#include "expectation_value.h"
#include "general/iter.h"
#include "math/num.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvPair.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/common/split.h"
#include "tools/finite/mpo.h"
#include <array>
#include <general/sfinae.h>
#include <h5pp/details/h5ppType.h>
namespace settings {
    constexpr bool debug_expval = false;
}

template<typename CalcType, typename Scalar, typename OpType>
Scalar tools::finite::measure::expectation_value(const StateFinite<Scalar> &state, const std::vector<LocalObservableOp<OpType>> &ops) {
    if constexpr(sfinae::is_std_complex_v<CalcType>) {
        bool mpsIsReal = state.is_real();
        bool opsIsReal = std::all_of(ops.begin(), ops.end(), [](const auto &op) -> bool { return tenx::isReal(op.op); });
        if(mpsIsReal and opsIsReal) {
            using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
            return expectation_value<RealScalar>(state, ops);
        }
    }

    if(state.mps_sites.empty()) throw std::runtime_error("expectation_value: state.mps_sites is empty");
    if(ops.empty()) throw std::runtime_error("expectation_value: ops is empty");
    auto d0    = state.mps_sites.front()->get_chiL();
    auto chain = tenx::TensorIdentity<CalcType>(d0);
    if(d0 != 1) tools::log->warn("expectation_value: chiL is not 1");

    auto &threads = tenx::threads::get();
    for(const auto &mps : state.mps_sites) {
        Eigen::Tensor<CalcType, 3> M   = mps->template get_M_as<CalcType>();
        const auto                 pos = mps->template get_position<long>();
        for(const auto &op : iter::reverse(ops)) {
            // Reverse to apply operators right to left, if they are on the same position
            // i.e. ABC|psi> should apply C first and A last (if A and C are on the same site).
            if(op.used or op.pos != pos) continue;
            auto temp                  = Eigen::Tensor<CalcType, 3>(op.op.dimension(0), M.dimension(1), M.dimension(2));
            temp.device(*threads->dev) = tenx::asScalarType<CalcType>(op.op).contract(M, tenx::idx({1}, {0}));
            M                          = std::move(temp);
            op.used                    = true;
        }
        auto temp                  = Eigen::Tensor<CalcType, 2>(M.dimension(2), M.dimension(2));
        temp.device(*threads->dev) = chain.contract(M, tenx::idx({0}, {1})).contract(M.conjugate(), tenx::idx({0, 1}, {1, 0}));
        chain                      = std::move(temp);
    }

    Eigen::Tensor<CalcType, 0> expval = chain.trace();
    return expval.coeff(0);
}
template cx64  tools::finite::measure::expectation_value<fp64>(const StateFinite<cx64> &state, const std::vector<LocalObservableOp<cx64>> &ops);
template cx64  tools::finite::measure::expectation_value<cx64>(const StateFinite<cx64> &state, const std::vector<LocalObservableOp<cx64>> &ops);
template cx128 tools::finite::measure::expectation_value<fp128>(const StateFinite<cx128> &state, const std::vector<LocalObservableOp<cx128>> &ops);
template cx128 tools::finite::measure::expectation_value<cx128>(const StateFinite<cx128> &state, const std::vector<LocalObservableOp<cx128>> &ops);

template<typename CalcType, typename Scalar, typename MpoType>
Scalar tools::finite::measure::expectation_value(const StateFinite<Scalar> &state, const std::vector<LocalObservableMpo<MpoType>> &mpos) {
    if constexpr(sfinae::is_std_complex_v<CalcType>) {
        bool mpsIsReal = state.is_real();
        bool mpoIsReal = std::all_of(mpos.begin(), mpos.end(), [](const auto &mpo) -> bool { return tenx::isReal(mpo.mpo); });
        if(mpsIsReal and mpoIsReal) {
            using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
            return expectation_value<RealScalar>(state, mpos);
        }
    }

    if(state.mps_sites.empty()) throw std::runtime_error("expectation_value: state.mps_sites is empty");
    if(mpos.empty()) throw std::runtime_error("expectation_value: obs is empty");

    // Generate a string of mpos for each site. If a site has no local observable given, insert an identity MPO there.
    auto mpodims = mpos.front().mpo.dimensions();

    for(auto &ob : mpos) {
        if(ob.mpo.dimension(2) != ob.mpo.dimension(3)) throw except::runtime_error("expectation_value: given mpo's of unequal spin dimension up and down");
        if(ob.mpo.dimension(0) != ob.mpo.dimension(1)) throw except::runtime_error("expectation_value: given mpo's of unequal bond dimension left and right");
        if(ob.mpo.dimensions() != mpodims) throw except::runtime_error("expectation_value: given mpo's of unequal dimensions");
    }

    // Create compatible edges
    Eigen::Tensor<CalcType, 1> Ledge(mpodims[0]); // The left  edge
    Eigen::Tensor<CalcType, 1> Redge(mpodims[1]); // The right edge
    Ledge(mpodims[0] - 1) = 1;
    Redge(0)              = 1;
    Eigen::Tensor<CalcType, 3> Ledge3, Redge3;
    {
        auto mpsdims = state.mps_sites.front()->dimensions();
        long mpsDim  = mpsdims[1];
        long mpoDim  = mpodims[0];
        Ledge3.resize(tenx::array3{mpsDim, mpsDim, mpoDim});
        Ledge3.setZero();
        for(long i = 0; i < mpsDim; i++) {
            std::array<long, 1> extent1                     = {mpoDim};
            std::array<long, 3> offset3                     = {i, i, 0};
            std::array<long, 3> extent3                     = {1, 1, mpoDim};
            Ledge3.slice(offset3, extent3).reshape(extent1) = Ledge;
        }
    }
    {
        auto mpsdims = state.mps_sites.back()->dimensions();
        long mpsDim  = mpsdims[2];
        long mpoDim  = mpodims[1];
        Redge3.resize(tenx::array3{mpsDim, mpsDim, mpoDim});
        Redge3.setZero();
        for(long i = 0; i < mpsDim; i++) {
            std::array<long, 1> extent1                     = {mpoDim};
            std::array<long, 3> offset3                     = {i, i, 0};
            std::array<long, 3> extent3                     = {1, 1, mpoDim};
            Redge3.slice(offset3, extent3).reshape(extent1) = Redge;
        }
    }

    // Generate an identity mpo with the same dimensions as the ones in obs
    Eigen::Tensor<Scalar, 4> mpoI = tenx::TensorIdentity<Scalar>(mpodims[0] * mpodims[2]).reshape(mpodims);

    // Start applying the mpo or identity on each site starting from Ledge3
    Eigen::Tensor<CalcType, 3> temp;
    for(const auto &mps : state.mps_sites) {
        const auto     pos   = mps->template get_position<long>();
        decltype(auto) M     = mps->template get_M_as<CalcType>();
        const auto     ob_it = std::find_if(mpos.begin(), mpos.end(), [&pos](const auto &ob) { return ob.pos == pos and not ob.used; });
        const auto    &mpo   = ob_it != mpos.end() ? ob_it->mpo : mpoI; // Choose the operator or an identity
        if(ob_it != mpos.end()) ob_it->used = true;
        temp.resize(M.dimension(2), M.dimension(2), mpo.dimension(1));
        temp = M.contract(Ledge3, tenx::idx({0}, {1}))
                   .contract(tenx::asScalarType<CalcType>(mpo), tenx::idx({0}, {1}))
                   .contract(M.conjugate(), tenx::idx({0, 1, 3}, {0, 2, 3}));
        Ledge3 = std::move(temp);
    }

    if(Ledge3.dimensions() != Redge3.dimensions())
        throw except::runtime_error("expectation_value: Ledge3 and Redge3 dimension mismatch: {} != {}", Ledge3.dimensions(), Redge3.dimensions());

    // Finish by contracting Redge3
    Eigen::Tensor<CalcType, 0> expval = Ledge3.contract(Redge3, tenx::idx({0, 1, 2}, {0, 1, 2}));

    if(std::imag(expval.coeff(0)) > 1e-14) tools::log->warn("expectation_value: result has imaginary part: {:8.2e}", fp(expval(0)));
    //    if(std::imag(expval(0)) > 1e-8)
    //        throw except::runtime_error("expectation_value: result has imaginary part: {:+8.2e}{:+8.2e} i", std::real(expval(0)), std::imag(expval(0)));
    return expval.coeff(0);
}
template cx64  tools::finite::measure::expectation_value<fp64>(const StateFinite<cx64> &state, const std::vector<LocalObservableMpo<cx64>> &mpos);
template cx64  tools::finite::measure::expectation_value<cx64>(const StateFinite<cx64> &state, const std::vector<LocalObservableMpo<cx64>> &mpos);
template cx128 tools::finite::measure::expectation_value<fp128>(const StateFinite<cx128> &state, const std::vector<LocalObservableMpo<cx128>> &mpos);
template cx128 tools::finite::measure::expectation_value<cx128>(const StateFinite<cx128> &state, const std::vector<LocalObservableMpo<cx128>> &mpos);

template<typename CalcType, typename Scalar, typename MpoType>
Scalar tools::finite::measure::expectation_value(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2,
                                                 const std::vector<Eigen::Tensor<MpoType, 4>> &mpos) {
    /*!
     * Calculates <state1 | mpos | state2>
     * The states and mpo can be of different type: then the mpo precision is converted to that of the state,
     * and the calculation is performed with the precision of the state.
     * Furthermore, if the mpo is real, we can check if the states are also real, in which case the whole calculation
     * can be cast to real mode (if they were complex to begin with).
     * The return type will always be that of the state, however
     *
     */
    if constexpr(sfinae::is_std_complex_v<Scalar> or sfinae::is_std_complex_v<MpoType>) {
        bool mpsIsReal = state1.is_real() and state2.is_real();
        bool mpoIsReal = std::all_of(mpos.begin(), mpos.end(), [](const auto &mpo) -> bool { return tenx::isReal(mpo); });
        if(mpsIsReal and mpoIsReal) {
            using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
            return expectation_value<RealScalar>(state1, state2, mpos);
        }
    }
    auto t_expval = tid::tic_scope("expval", tid::level::highest);

    if(!num::all_equal(state1.get_length(), state2.get_length(), mpos.size()))
        throw except::logic_error("Sizes are not equal: state1:{} state2:{} mpos:{}", state1.get_length(), state2.get_length(), mpos.size());

    auto L = mpos.size();
    if(state1.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state1 left bond dimension != 1: got {}", state1.get_mps_site(0).get_chiL());
    if(state2.get_mps_site(0).get_chiL() != 1) throw except::logic_error("state2 left bond dimension != 1: got {}", state2.get_mps_site(0).get_chiL());
    if(mpos.front().dimension(0) != 1) throw except::logic_error("mpos left bond dimension != 1: got {}", mpos.front().dimension(0));
    if(state1.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state1 right bond dimension != 1: got {}", state1.get_mps_site(L - 1).get_chiR());
    if(state2.get_mps_site(L - 1).get_chiR() != 1) throw except::logic_error("state2 right bond dimension != 1: got {}", state2.get_mps_site(L - 1).get_chiR());
    if(mpos.back().dimension(1) != 1) throw except::logic_error("mpos right bond dimension != 1: got {}", mpos.back().dimension(1));
    Eigen::Tensor<CalcType, 4> result, tmp;
    auto                      &threads = tenx::threads::get();

    for(size_t pos = 0; pos < L; ++pos) {
        decltype(auto) mps1 = state1.get_mps_site(pos).template get_M_as<CalcType>();
        decltype(auto) mps2 = state2.get_mps_site(pos).template get_M_as<CalcType>();
        decltype(auto) mpo  = tenx::asScalarType<CalcType>(mpos[pos]);
        if(pos == 0) {
            auto dim4 = tenx::array4{mpo.dimension(0) * mps1.dimension(1) * mps2.dimension(1), mps1.dimension(2), mpo.dimension(1), mps2.dimension(2)};
            auto shf6 = tenx::array6{0, 2, 4, 1, 3, 5};
            result.resize(dim4);
            result.device(*threads->dev) = mps1.contract(mpo, tenx::idx({0}, {2})).contract(mps2, tenx::idx({4}, {0})).shuffle(shf6).reshape(dim4);
            continue;
        }
        auto dim4 = tenx::array4{result.dimension(0), mps1.dimension(2), mpo.dimension(1), mps2.dimension(2)};
        tmp.resize(dim4);
        tmp.device(*threads->dev) =
            result.contract(mps1.conjugate(), tenx::idx({1}, {1})).contract(mpo, tenx::idx({1, 3}, {0, 2})).contract(mps2, tenx::idx({1, 4}, {1, 0}));
        result = std::move(tmp);
    }
    // In the end we should have a tensor of size 1 (if the state and mpo edges have dim 1).
    // We can extract and return this value
    if(result.size() != 1) tools::log->warn("expectation_value: result does not have size 1!");
    return result.coeff(0);
}
/* clang-format off */
template cx64   tools::finite::measure::expectation_value<fp32>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos);
template cx64   tools::finite::measure::expectation_value<fp64>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos);
template cx128  tools::finite::measure::expectation_value<fp128>(const StateFinite<cx128> &state1, const StateFinite<cx128> &state2, const std::vector<Eigen::Tensor<cx128, 4>> &mpos);
template cx64   tools::finite::measure::expectation_value<cx32>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos);
template cx64   tools::finite::measure::expectation_value<cx64>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos);
template cx128  tools::finite::measure::expectation_value<cx128>(const StateFinite<cx128> &state1, const StateFinite<cx128> &state2, const std::vector<Eigen::Tensor<cx128, 4>> &mpos);
/* clang-format on */

template<typename CalcType, typename Scalar, typename MpoType>
Scalar tools::finite::measure::expectation_value(const StateFinite<Scalar> &state1, const StateFinite<Scalar> &state2,
                                                 const std::vector<Eigen::Tensor<MpoType, 4>> &mpos, const Eigen::Tensor<MpoType, 1> &ledge,
                                                 const Eigen::Tensor<MpoType, 1> &redge) {
    /*!
     * Calculates <state1 | mpos | state2>
     */
    auto mpos_w_edge = mpo::get_mpos_with_edges(mpos, ledge, redge);
    return expectation_value<CalcType>(state1, state2, mpos_w_edge);
}
/* clang-format off */
template cx64   tools::finite::measure::expectation_value<fp32>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos,  const Eigen::Tensor<cx64, 1> &ledge, const Eigen::Tensor<cx64, 1> &redge);
template cx64   tools::finite::measure::expectation_value<fp64>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos,  const Eigen::Tensor<cx64, 1> &ledge, const Eigen::Tensor<cx64, 1> &redge);
template cx128  tools::finite::measure::expectation_value<fp128>(const StateFinite<cx128> &state1, const StateFinite<cx128> &state2, const std::vector<Eigen::Tensor<cx128, 4>> &mpos,  const Eigen::Tensor<cx128, 1> &ledge, const Eigen::Tensor<cx128, 1> &redge);
template cx64   tools::finite::measure::expectation_value<cx32>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos,  const Eigen::Tensor<cx64, 1> &ledge, const Eigen::Tensor<cx64, 1> &redge);
template cx64   tools::finite::measure::expectation_value<cx64>(const StateFinite<cx64> &state1, const StateFinite<cx64> &state2, const std::vector<Eigen::Tensor<cx64, 4>> &mpos,  const Eigen::Tensor<cx64, 1> &ledge, const Eigen::Tensor<cx64, 1> &redge);
template cx128  tools::finite::measure::expectation_value<cx128>(const StateFinite<cx128> &state1, const StateFinite<cx128> &state2, const std::vector<Eigen::Tensor<cx128, 4>> &mpos,  const Eigen::Tensor<cx128, 1> &ledge, const Eigen::Tensor<cx128, 1> &redge);
/* clang-format on */

template<typename CalcType, typename Scalar>
Eigen::Tensor<Scalar, 1> tools::finite::measure::expectation_values(const StateFinite<Scalar> &state, const Eigen::Tensor<cx64, 2> &op) {
    tools::log->trace("Measuring local expectation values");
    long                     len = state.template get_length<long>();
    Eigen::Tensor<Scalar, 1> expvals(len);
    expvals.setZero();
    for(long pos = 0; pos < len; pos++) {
        LocalObservableOp<cx64> ob1 = {op, pos};
        expvals(pos)                = expectation_value<CalcType>(state, std::vector{ob1});
    }
    return expvals;
}

template Eigen::Tensor<cx64, 1>  tools::finite::measure::expectation_values<fp64>(const StateFinite<cx64> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<cx64, 1>  tools::finite::measure::expectation_values<cx64>(const StateFinite<cx64> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<cx128, 1> tools::finite::measure::expectation_values<fp128>(const StateFinite<cx128> &state, const Eigen::Tensor<cx64, 2> &op);
template Eigen::Tensor<cx128, 1> tools::finite::measure::expectation_values<cx128>(const StateFinite<cx128> &state, const Eigen::Tensor<cx64, 2> &op);

template<typename CalcType, typename Scalar>
Eigen::Tensor<Scalar, 1> tools::finite::measure::expectation_values(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &op) {
    Eigen::Tensor<Eigen::Matrix2cd::Scalar, 2> tensor_op = tenx::TensorMap(op);
    return expectation_values<CalcType>(state, tensor_op);
}
template Eigen::Tensor<cx64, 1>  tools::finite::measure::expectation_values<fp64>(const StateFinite<cx64> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<cx64, 1>  tools::finite::measure::expectation_values<cx64>(const StateFinite<cx64> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<cx128, 1> tools::finite::measure::expectation_values<fp128>(const StateFinite<cx128> &state, const Eigen::Matrix2cd &op);
template Eigen::Tensor<cx128, 1> tools::finite::measure::expectation_values<cx128>(const StateFinite<cx128> &state, const Eigen::Matrix2cd &op);

template<typename CalcType, typename Scalar, typename EnvType>
Scalar tools::finite::measure::expectation_value(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet,
                                                 const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvType> &envs) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet> by applying the mpos in series without splitting the mps first
     */
    if constexpr(sfinae::is_std_complex_v<CalcType>) {
        using RealScalar = typename Scalar::value_type;
        bool mpsIsReal   = tenx::isReal(mpsBra) and tenx::isReal(mpsKet);
        bool envIsReal   = envs.L.is_real() and envs.R.is_real();
        bool mpoIsReal   = std::all_of(mpos.begin(), mpos.end(), [](const auto &mpo) -> bool { return mpo.get().is_real(); });
        if(mpsIsReal and envIsReal and mpoIsReal) { return expectation_value<RealScalar>(mpsBra, mpsKet, mpos, envs); }
    }

    auto t_expval = tid::tic_scope("expval", tid::level::highest);

    // Extract the correct tensors depending on EnvType
    decltype(auto)                          envL = envs.L.template get_block_as<CalcType>();
    decltype(auto)                          envR = envs.R.template get_block_as<CalcType>();
    std::vector<Eigen::Tensor<CalcType, 4>> mpos_shf;
    for(size_t pos = 0; pos < mpos.size(); ++pos) {
        if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvEne<Scalar>>)
            mpos_shf.emplace_back(mpos[pos].get().template MPO_as<CalcType>().shuffle(tenx::array4{2, 3, 0, 1}));
        else if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvVar<Scalar>>)
            mpos_shf.emplace_back(mpos[pos].get().template MPO2_as<CalcType>().shuffle(tenx::array4{2, 3, 0, 1}));
        else
            static_assert(h5pp::type::sfinae::invalid_type_v<EnvType>);
    }

    Eigen::Tensor<CalcType, 3> mpoMpsKet = tools::common::contraction::matrix_vector_product(tenx::asScalarType<CalcType>(mpsKet), mpos_shf, envL, envR);
    return tools::common::contraction::contract_mps_overlap(tenx::asScalarType<CalcType>(mpsBra), mpoMpsKet);
}
/* clang-format off */
template cx64 tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<EnvEne<cx64>> &envs);
template cx64 tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<EnvEne<cx64>> &envs);
template cx64 tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<EnvVar<cx64>> &envs);
template cx64 tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<EnvVar<cx64>> &envs);

template cx128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<EnvEne<cx128>> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<EnvEne<cx128>> &envs);
template cx128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<EnvVar<cx128>> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<EnvVar<cx128>> &envs);
/* clang-format on */

template<typename CalcType, typename Scalar, typename EnvType>
Scalar tools::finite::measure::expectation_value(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsBra,
                                                 const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpsKet,
                                                 const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvType> &envs) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet>
     */
    auto t_expval = tid::tic_scope("expval", tid::level::highest);

    assert(num::all_equal(mpsBra.size(), mpsKet.size(), mpos.size()));
    static_assert(sfinae::is_any_v<EnvType, EnvEne<Scalar>, EnvVar<Scalar>>);
    if constexpr(sfinae::is_std_complex_v<CalcType>) {
        using RealScalar = typename Scalar::value_type;
        bool braIsReal   = std::all_of(mpsBra.begin(), mpsBra.end(), [](const auto &mps) -> bool { return mps.get().is_real(); });
        bool ketIsReal   = std::all_of(mpsKet.begin(), mpsKet.end(), [](const auto &mps) -> bool { return mps.get().is_real(); });
        bool envIsReal   = envs.L.is_real() and envs.R.is_real();
        bool mpoIsReal   = std::all_of(mpos.begin(), mpos.end(), [](const auto &mpo) -> bool { return mpo.get().is_real(); });
        if(braIsReal and ketIsReal and envIsReal and mpoIsReal) { return expectation_value<RealScalar>(mpsBra, mpsKet, mpos, envs); }
    }

    if(mpos.size() == 1) {
        if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvEne<Scalar>>)
            return tools::common::contraction::expectation_value(
                mpsBra.front().get().template get_M_as<CalcType>(), mpsKet.front().get().template get_M_as<CalcType>(),
                mpos.front().get().template MPO_as<CalcType>(), envs.L.template get_block_as<CalcType>(), envs.R.template get_block_as<CalcType>());
        if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvVar<Scalar>>)
            return tools::common::contraction::expectation_value(
                mpsBra.front().get().template get_M_as<CalcType>(), mpsKet.front().get().template get_M_as<CalcType>(),
                mpos.front().get().template MPO2_as<CalcType>(), envs.L.template get_block_as<CalcType>(), envs.R.template get_block_as<CalcType>());
    }

    Eigen::Tensor<CalcType, 3> resL = envs.L.template get_block_as<CalcType>().shuffle(tenx::array3{0, 2, 1});

    // auto                     mporef  = std::optional<std::reference_wrapper<const Eigen::Tensor<Scalar, 4>>>(std::nullopt);
    auto &threads = tenx::threads::get();
    for(size_t pos = 0; pos < mpos.size(); ++pos) {
        Eigen::Tensor<CalcType, 4> mpo;
        if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvEne<Scalar>>)
            mpo = mpos[pos].get().template MPO_as<CalcType>();
        else if constexpr(std::is_same_v<std::remove_cvref_t<EnvType>, EnvVar<Scalar>>)
            mpo = mpos[pos].get().template MPO2_as<CalcType>();
        else
            static_assert(h5pp::type::sfinae::invalid_type_v<EnvType>);

        decltype(auto) bra = mpsBra[pos].get().template get_M_as<CalcType>();
        decltype(auto) ket = mpsKet[pos].get().template get_M_as<CalcType>();
        assert(resL.dimension(0) == bra.dimension(1));
        assert(resL.dimension(1) == mpo.dimension(0));
        assert(resL.dimension(2) == ket.dimension(1));
        assert(mpo.dimension(2) == ket.dimension(0));
        assert(mpo.dimension(3) == bra.dimension(0));

        auto                       dim3 = tenx::array3{ket.dimension(2), mpo.dimension(1), bra.dimension(2)};
        Eigen::Tensor<CalcType, 3> tmp;
        tmp.resize(dim3);
        tmp.device(*threads->dev) =
            resL.contract(ket, tenx::idx({0}, {1})).contract(mpo, tenx::idx({0, 2}, {0, 2})).contract(bra.conjugate(), tenx::idx({0, 3}, {1, 0}));
        resL = std::move(tmp);
    }
    Eigen::Tensor<CalcType, 0> res = resL.contract(envs.R.template get_block_as<CalcType>(), tenx::idx({0, 1, 2}, {0, 2, 1}));
    return res.coeff(0);
}
/* clang-format off */
template cx64 tools::finite::measure::expectation_value<fp64>(const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvEne<cx64> &> &envs);
template cx64 tools::finite::measure::expectation_value<cx64>(const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvEne<cx64> &> &envs);
template cx64 tools::finite::measure::expectation_value<fp64>(const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvVar<cx64> &> &envs);
template cx64 tools::finite::measure::expectation_value<cx64>(const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx64>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvVar<cx64> &> &envs);
template cx128 tools::finite::measure::expectation_value<fp128>(const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvEne<cx128> &> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvEne<cx128> &> &envs);
template cx128 tools::finite::measure::expectation_value<fp128>(const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvVar<cx128> &> &envs);
template cx128 tools::finite::measure::expectation_value<cx128>(const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsBra, const std::vector<std::reference_wrapper<const MpsSite<cx128>>> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvVar<cx128> &> &envs);
/* clang-format on */

template<typename CalcType, typename Scalar, typename EnvType>
Scalar tools::finite::measure::expectation_value(const Eigen::Tensor<Scalar, 3> &mpsBra, const Eigen::Tensor<Scalar, 3> &mpsKet,
                                                 const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvType> &envs,
                                                 std::optional<svd::config> svd_cfg) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet>, where the mps are already multisite.
     * E.g. for 3 sites we have
     *
     *  |----0             chiL---[ket]---chiR            0----|
     *  |                          d^3                         |
     *  |               2           2           2              |
     *  |----2      0---|---1   0---|---1   0---|---1     2----|
     *  |               3           3           3              |
     *  |                          d^3                         |
     *  |----1             chiL---[bra]---chiR            1----|
     *
     * To apply the mpo's one by one efficiently, we need to split the mps using SVD first
     * Here we make the assumption that bra and ket are not necessarily equal
     */
    if(not svd_cfg) return expectation_value<cx64>(mpsBra, mpsKet, mpos, envs);

    std::vector<long>   spin_dims_bra, spin_dims_ket;
    std::vector<size_t> positions;
    spin_dims_bra.reserve(mpos.size());
    spin_dims_ket.reserve(mpos.size());
    positions.reserve(mpos.size());
    for(const auto &mpo : mpos) {
        spin_dims_bra.emplace_back(mpo.get().get_spin_dimension());
        spin_dims_ket.emplace_back(mpo.get().get_spin_dimension());
        positions.emplace_back(mpo.get().get_position());
    }

    auto mpsBra_split = tools::common::split::split_mps<Scalar>(mpsBra, spin_dims_bra, positions, safe_cast<long>(positions.back()), svd_cfg);
    auto mpsKet_split = tools::common::split::split_mps<Scalar>(mpsKet, spin_dims_ket, positions, safe_cast<long>(positions.back()), svd_cfg);

    // Put them into a vector of reference wrappers for compatibility with the other expectation_value function
    auto mpsBra_refs = std::vector<std::reference_wrapper<const MpsSite<Scalar>>>(mpsBra_split.begin(), mpsBra_split.end());
    auto mpsKet_refs = std::vector<std::reference_wrapper<const MpsSite<Scalar>>>(mpsKet_split.begin(), mpsKet_split.end());

    return expectation_value<CalcType>(mpsBra_refs, mpsKet_refs, mpos, envs);
}

/* clang-format off */
template cx64 tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvEne<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvEne<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvVar<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<cx64, 3> &mpsBra, const Eigen::Tensor<cx64, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvVar<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvEne<cx128> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvEne<cx128> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvVar<cx128> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<cx128, 3> &mpsBra, const Eigen::Tensor<cx128, 3> &mpsKet, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvVar<cx128> &> &envs, std::optional<svd::config> svd_cfg);
/* clang-format on */

template<typename CalcType, typename Scalar, typename EnvType>
Scalar tools::finite::measure::expectation_value(const Eigen::Tensor<Scalar, 3>                                   &multisite_mps,
                                                 const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos, const env_pair<EnvType> &envs,
                                                 std::optional<svd::config> svd_cfg) {
    /*!
     * Calculates <mpsBra | mpos | mpsKet>, where the mps are already multisite.
     * E.g. for 3 sites we have
     *
     *  |----0             chiL---[ket]---chiR            0----|
     *  |                          d^3                         |
     *  |               2           2           2              |
     *  |----2      0---|---1   0---|---1   0---|---1     2----|
     *  |               3           3           3              |
     *  |                          d^3                         |
     *  |----1             chiL---[bra]---chiR            1----|
     *
     * To apply the mpo's one by one efficiently, we need to split the mps using SVD first
     * Here we make the assumption that bra and ket are not necessarily equal
     */
    if(not svd_cfg) return expectation_value<CalcType>(multisite_mps, multisite_mps, mpos, envs);

    std::vector<long>   spin_dims_bra, spin_dims_ket;
    std::vector<size_t> positions;
    spin_dims_bra.reserve(mpos.size());
    spin_dims_ket.reserve(mpos.size());
    positions.reserve(mpos.size());
    for(const auto &mpo : mpos) {
        spin_dims_bra.emplace_back(mpo.get().get_spin_dimension());
        spin_dims_ket.emplace_back(mpo.get().get_spin_dimension());
        positions.emplace_back(mpo.get().get_position());
    }
    if(positions.size() < 2) {
        tools::log->warn("expectation_value: skipped splitting a single-site multisite_mps");
        // No need to split. Also, splitting would require stashing artifacts from SVD for neighboring sites,
        // which we can't really do here.
        auto oneL = Eigen::Tensor<Scalar, 1>(multisite_mps.dimension(1));
        auto oneR = Eigen::Tensor<Scalar, 1>(multisite_mps.dimension(2));
        auto mps  = MpsSite<Scalar>(multisite_mps, oneL, positions.front(), 0.0, "AC");
        mps.set_LC(oneR);
        // Put them into a vector of reference wrappers for compatibility with the other expectation_value function
        auto mps_refs = std::vector<std::reference_wrapper<const MpsSite<Scalar>>>{mps};
        return expectation_value<CalcType>(mps_refs, mps_refs, mpos, envs);
    } else {
        // We can avoid splitting the mps by applying the mpos directly onto the mps in sequence
        // auto mpo_mps = tools::common::contraction::matrix_vector_product(multisite_mps, mpos, envs);
        // return tools::common::contraction::contract_mps_overlap(multisite_mps, mpo_mps);

        // Set the new center position in the interior of the set of positions, so we don't get stashes that need to be thrown away.
        auto mps_split = tools::common::split::split_mps<Scalar>(multisite_mps, spin_dims_bra, positions, safe_cast<long>(positions.front()), svd_cfg);
        // Put them into a vector of reference wrappers for compatibility with the other expectation_value function
        auto mps_refs = std::vector<std::reference_wrapper<const MpsSite<Scalar>>>(mps_split.begin(), mps_split.end());
        return expectation_value<CalcType>(mps_refs, mps_refs, mpos, envs);
    }
}
/* clang-format off */
template cx64 tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvEne<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvEne<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value<fp64>(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvVar<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx64 tools::finite::measure::expectation_value<cx64>(const Eigen::Tensor<cx64, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos, const env_pair<const EnvVar<cx64> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvEne<cx128> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvEne<cx128> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<fp128>(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvVar<cx128> &> &envs, std::optional<svd::config> svd_cfg);
template cx128 tools::finite::measure::expectation_value<cx128>(const Eigen::Tensor<cx128, 3> &multisite_mps, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos, const env_pair<const EnvVar<cx128> &> &envs, std::optional<svd::config> svd_cfg);
/* clang-format on */
