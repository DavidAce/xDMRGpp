#include "ModelInfinite.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/num.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/site/mpo/MpoFactory.h"

template class ModelInfinite<fp32>;
template class ModelInfinite<fp64>;
template class ModelInfinite<fp128>;
template class ModelInfinite<cx32>;
template class ModelInfinite<cx64>;
template class ModelInfinite<cx128>;

template<typename Scalar>
ModelInfinite<Scalar>::ModelInfinite() = default;

// We need to define the destructor and other special functions
// because we enclose data in unique_ptr for this pimpl idiom.
// Otherwise, unique_ptr will forcibly inline its own default deleter.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
template<typename Scalar>
ModelInfinite<Scalar>::~ModelInfinite() = default; // default dtor
template<typename Scalar>
ModelInfinite<Scalar>::ModelInfinite(ModelInfinite &&other) = default; // default move ctor
template<typename Scalar>
ModelInfinite<Scalar> &ModelInfinite<Scalar>::operator=(ModelInfinite &&other) = default; // default move assign

template<typename Scalar>
ModelInfinite<Scalar>::ModelInfinite(const ModelInfinite &other)
    : cache(other.cache), HA(other.HA->clone()), HB(other.HB->clone()), model_type(other.model_type) {}

template<typename Scalar>
ModelInfinite<Scalar> &ModelInfinite<Scalar>::operator=(const ModelInfinite &other) {
    // check for self-assignment
    if(this != &other) {
        cache      = other.cache;
        HA         = other.HA->clone();
        HB         = other.HB->clone();
        model_type = other.model_type;
    }
    return *this;
}

template<typename Scalar>
void ModelInfinite<Scalar>::initialize(ModelType model_type_) {
    tools::log->trace("Initializing model");
    // Generate MPO
    model_type = model_type_;
    HA         = MpoFactory<Scalar>::create_mpo(0, model_type);
    HB         = MpoFactory<Scalar>::create_mpo(1, model_type);
}

template<typename Scalar>
void ModelInfinite<Scalar>::randomize() {
    tools::log->trace("Randomizing hamiltonian");
    HA->randomize_hamiltonian();
    HB->randomize_hamiltonian();
    std::vector<typename MpoSite<Scalar>::TableMap> all_params;
    all_params.push_back(HA->get_parameters());
    all_params.push_back(HB->get_parameters());
    HA->set_averages(all_params, true);
    HB->set_averages(all_params, true);
}

template<typename Scalar>
void ModelInfinite<Scalar>::reset_mpo_squared() {
    tools::log->debug("Resetting squared MPO");
    HA->build_mpo_squared();
    HB->build_mpo_squared();
}

template<typename Scalar>
void ModelInfinite<Scalar>::rebuild_mpo_squared() {
    if(settings::precision::use_compressed_mpo_squared != MpoCompress::NONE) {
        tools::log->trace("Compressing MPO²");
        throw std::runtime_error("Compressing the squared MPO² is currently unsupported on infinite systems.\n"
                                 "Set settings::precision::use_compressed_mpo_squared_all = false");
        auto mpo_compressed = get_compressed_mpo_squared();
        HA->set_mpo_squared(mpo_compressed[0]);
        HA->set_mpo_squared(mpo_compressed[1]);
    } else
        reset_mpo_squared();
}

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> ModelInfinite<Scalar>::get_compressed_mpo_squared() {
    // First, rebuild the MPO's
    std::vector<Eigen::Tensor<Scalar, 4>> mpos_sq;
    mpos_sq.emplace_back(HA->get_non_compressed_mpo_squared());
    mpos_sq.emplace_back(HB->get_non_compressed_mpo_squared());

    // Setup SVD
    // Here we need a lot of precision:
    //  - Use very low svd threshold
    //  - Force the use of JacobiSVD by setting the switchsize_bdc to something large
    //  - Force the use of Lapacke -- it is more precise than Eigen (I don't know why)
    auto svd_cfg = svd::config();
    // Eigen Jacobi becomes ?gesvd (i.e. using QR) with the BLAS backend.
    // See here: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1732
    svd_cfg.svd_lib = svd::lib::lapacke;
    svd_cfg.svd_rtn = svd::rtn::gesvj;
    auto svd        = svd::solver(svd_cfg);
    // Print the results
    std::vector<std::string> report;
    if(tools::log->level() == spdlog::level::trace)
        for(const auto &mpo : mpos_sq) report.emplace_back(fmt::format("{}", mpo.dimensions()));

    for(size_t iter = 0; iter < 1; iter++) {
        // Next compress from left to right
        Eigen::Tensor<Scalar, 2> T_l2r; // Transfer matrix
        Eigen::Tensor<Scalar, 4> T_mpo_sq;
        for(const auto &[idx, mpo_sq] : iter::enumerate(mpos_sq)) {
            if(T_l2r.size() == 0)
                T_mpo_sq = mpo_sq;
            else
                T_mpo_sq = T_l2r.contract(mpo_sq, tenx::idx({1}, {0}));

            if(idx == mpos_sq.size() - 1) {
                mpo_sq = T_mpo_sq;
            } else {
                std::tie(mpo_sq, T_l2r) = svd.split_mpo_l2r(T_mpo_sq);
                if(idx + 1 == mpos_sq.size())
                    // The remaining transfer matrix T can be multiplied back into the last MPO from the right
                    mpo_sq = Eigen::Tensor<Scalar, 4>(mpo_sq.contract(T_l2r, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2}));
            }
        }

        // Now we have done left to right. Next we do right to left
        Eigen::Tensor<Scalar, 2> T_r2l;    // Transfer matrix
        Eigen::Tensor<Scalar, 4> mpo_sq_T; // Absorbs transfer matrix
        for(const auto &[idx, mpo_sq] : iter::enumerate_reverse(mpos_sq)) {
            if(T_r2l.size() == 0)
                mpo_sq_T = mpo_sq;
            else
                mpo_sq_T = mpo_sq.contract(T_r2l, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
            if(idx == 0) {
                mpo_sq = mpo_sq_T;
            } else {
                std::tie(T_r2l, mpo_sq) = svd.split_mpo_r2l(mpo_sq_T);
                if(idx == 0)
                    // The remaining transfer matrix T can be multiplied back into the first MPO from the left
                    mpo_sq = Eigen::Tensor<Scalar, 4>(T_r2l.contract(mpo_sq, tenx::idx({1}, {0})));
            }
        }
    }

    // Print the results
    if(tools::log->level() == spdlog::level::trace)
        for(const auto &[idx, msg] : iter::enumerate(report)) tools::log->trace("MPO² {}: {} -> {}", idx, msg, mpos_sq[idx].dimensions());
    return mpos_sq;
}

template<typename Scalar>
bool ModelInfinite<Scalar>::is_real() const {
    return HA->is_real() and HB->is_real();
}
template<typename Scalar>
bool ModelInfinite<Scalar>::has_nan() const {
    return HA->has_nan() or HB->has_nan();
}
template<typename Scalar>
void ModelInfinite<Scalar>::assert_validity() const {
    HA->assert_validity();
    HB->assert_validity();
}

template<typename Scalar>
const MpoSite<Scalar> &ModelInfinite<Scalar>::get_mpo_siteA() const {
    return *HA;
}
template<typename Scalar>
const MpoSite<Scalar> &ModelInfinite<Scalar>::get_mpo_siteB() const {
    return *HB;
}
template<typename Scalar>
MpoSite<Scalar> &ModelInfinite<Scalar>::get_mpo_siteA() {
    return *HA;
}
template<typename Scalar>
MpoSite<Scalar> &ModelInfinite<Scalar>::get_mpo_siteB() {
    return *HB;
}

template<typename Scalar>
Eigen::DSizes<long, 4> ModelInfinite<Scalar>::dimensions() const {
    long dim0 = get_mpo_siteA().MPO().dimension(0);
    long dim1 = get_mpo_siteB().MPO().dimension(1);
    long dim2 = get_mpo_siteA().MPO().dimension(2) * get_mpo_siteB().MPO().dimension(2);
    long dim3 = get_mpo_siteA().MPO().dimension(3) * get_mpo_siteB().MPO().dimension(3);
    return Eigen::DSizes<long, 4>{dim0, dim1, dim2, dim3};
}

template<typename Scalar>
bool ModelInfinite<Scalar>::is_shifted() const {
    return HA->has_energy_shifted_mpo() and HB->has_energy_shifted_mpo();
}

template<typename Scalar>
Scalar ModelInfinite<Scalar>::get_energy_shift_per_site() const {
    if(not num::all_equal(HA->get_energy_shift_mpo(), get_mpo_siteB().get_energy_shift_mpo()))
        throw except::runtime_error("Energy shift mismatch: HA {:.16f} != HB {:.16f}", fp(get_mpo_siteA().get_energy_shift_mpo()),
                                    fp(get_mpo_siteB().get_energy_shift_mpo()));
    return HA->get_energy_shift_mpo();
}

template<typename Scalar>
void ModelInfinite<Scalar>::set_energy_shift_per_site(Scalar energy_shift_per_site) {
    HA->set_energy_shift_mpo(energy_shift_per_site);
    HB->set_energy_shift_mpo(energy_shift_per_site);
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 4> &ModelInfinite<Scalar>::get_2site_mpo_AB() const {
    if(cache.twosite_mpo_AB) return cache.twosite_mpo_AB.value();
    long dim0            = get_mpo_siteA().MPO().dimension(0);
    long dim1            = get_mpo_siteB().MPO().dimension(1);
    long dim2            = get_mpo_siteA().MPO().dimension(2) * get_mpo_siteB().MPO().dimension(2);
    long dim3            = get_mpo_siteA().MPO().dimension(3) * get_mpo_siteB().MPO().dimension(3);
    cache.twosite_mpo_AB = get_mpo_siteA()
                               .MPO()
                               .contract(get_mpo_siteB().MPO(), tenx::idx({1}, {0}))
                               .shuffle(tenx::array6{0, 3, 1, 4, 2, 5})
                               .reshape(tenx::array4{dim0, dim1, dim2, dim3});
    return cache.twosite_mpo_AB.value();
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 4> &ModelInfinite<Scalar>::get_2site_mpo_BA() const {
    if(cache.twosite_mpo_BA) return cache.twosite_mpo_BA.value();
    long dim0 = get_mpo_siteB().MPO().dimension(0);
    long dim1 = get_mpo_siteA().MPO().dimension(1);
    long dim2 = get_mpo_siteB().MPO().dimension(2) * get_mpo_siteA().MPO().dimension(2);
    long dim3 = get_mpo_siteB().MPO().dimension(3) * get_mpo_siteA().MPO().dimension(3);
    cache.twosite_mpo_BA =
        HB->MPO().contract(HA->MPO(), tenx::idx({1}, {0})).shuffle(tenx::array6{0, 3, 1, 4, 2, 5}).reshape(tenx::array4{dim0, dim1, dim2, dim3});
    return cache.twosite_mpo_BA.value();
}

template<typename Scalar>
const Eigen::Tensor<Scalar, 2> &ModelInfinite<Scalar>::get_2site_ham_AB() const {
    if(cache.twosite_ham_AB) return cache.twosite_ham_AB.value();
    auto twosite_mpo_AB  = get_2site_mpo_AB();
    auto edgeL           = get_mpo_siteA().template get_MPO_edge_left<Scalar>();
    auto edgeR           = get_mpo_siteB().template get_MPO_edge_right<Scalar>();
    cache.twosite_ham_AB = twosite_mpo_AB.contract(edgeL, tenx::idx({0}, {0})).contract(edgeR, tenx::idx({0}, {0}));
    return cache.twosite_ham_AB.value();
}
template<typename Scalar>
const Eigen::Tensor<Scalar, 2> &ModelInfinite<Scalar>::get_2site_ham_BA() const {
    if(cache.twosite_ham_BA) return cache.twosite_ham_BA.value();
    auto twosite_mpo_BA  = get_2site_mpo_BA();
    auto edgeL           = get_mpo_siteB().template get_MPO_edge_left<Scalar>();
    auto edgeR           = get_mpo_siteA().template get_MPO_edge_right<Scalar>();
    cache.twosite_ham_BA = twosite_mpo_BA.contract(edgeL, tenx::idx({0}, {0})).contract(edgeR, tenx::idx({0}, {0}));
    return cache.twosite_ham_BA.value();
}

template<typename Scalar>
void ModelInfinite<Scalar>::clear_cache() {
    cache = Cache();
}
