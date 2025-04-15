#include "../mpo.h"
#include "qm/lbit.h"
#include "qm/spin.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"
extern std::vector<Eigen::Tensor<cx64, 4>> tools::finite::mpo::get_deprojected_mpos(const StateFinite &state, const ModelFinite &model) {
    assert(state.mps_sites.size() == model.MPO.size());
    assert(state.get_spin_dim() == model.MPO.front()->get_spin_dimension());
    long d = state.get_spin_dim();
    assert(d == 2);
    auto I  = Eigen::TensorMap<const Eigen::Tensor<cx64, 4>>(qm::spin::half::tensor::id.data(), 1, 1, d, d);
    auto Id = I.dimensions();
    auto I0 = std::array<long, 4>{0, 0, 0, 0};
    auto R0 = std::array<long, 4>{Id[0], Id[0], 0, 0};

    auto constexpr shfR = std::array<long, 6>{1, 4, 2, 5, 0, 3};

    std::vector<Eigen::Tensor<cx64, 4>> prjs(model.MPO.size());

    for(size_t pos = 0; pos < state.mps_sites.size(); ++pos) {
        // Create a projector   P = I - |M><M|
        Eigen::Tensor<cx64, 3> M  = state.get_multisite_mps<cx64>({pos});
        auto                   Md = M.dimensions();
        assert(Md[0] == d);

        auto                   shpR = std::array<long, 4>{Md[1] * Md[1], Md[2] * Md[2], Md[0], Md[0]};
        Eigen::Tensor<cx64, 4> R    = M.contract(M.conjugate(), tenx::idx()).shuffle(shfR).reshape(shpR); // Rho
        auto                   Rd   = R.dimensions();

        auto &P         = prjs[pos];
        P               = Eigen::Tensor<cx64, 4>(Id[0] + Rd[0], Id[1] + Rd[1], Rd[2], Rd[3]);
        P.slice(I0, Id) = I;
        P.slice(R0, Rd) = -R;
    }
    prjs = get_deparallelized_mpos(prjs);

    std::vector<Eigen::Tensor<cx64, 4>> mpos(model.MPO.size());
    std::vector<std::array<long, 4>>    mpos_dims_prj;
    for(size_t pos = 0; pos < model.MPO.size(); ++pos) mpos[pos] = model.get_mpo(pos).MPO();



    mpos = qm::lbit::merge_unitary_mpo_layers(prjs, mpos, prjs);

    for(size_t pos = 0; pos < mpos.size(); ++pos) { tools::log->info("deprojected mpo[{}]: {}", pos, mpos[pos].dimensions()); }
    return mpos;
}
