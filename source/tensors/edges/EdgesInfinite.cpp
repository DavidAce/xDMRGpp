#include "EdgesInfinite.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvPair.h"
#include "tensors/site/env/EnvVar.h"
#include "tools/common/log.h"

template class EdgesInfinite<cx64>;
template class EdgesInfinite<cx128>;

template<typename Scalar>
EdgesInfinite<Scalar>::EdgesInfinite()
    : eneL(std::make_unique<EnvEne<Scalar>>()), eneR(std::make_unique<EnvEne<Scalar>>()), varL(std::make_unique<EnvVar<Scalar>>()),
      varR(std::make_unique<EnvVar<Scalar>>()) {}

// We need to define the destructor and other special functions
// because we enclose data in unique_ptr for this pimpl idiom.
// Otherwise unique_ptr will forcibly inline its own default deleter.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
template<typename Scalar>
EdgesInfinite<Scalar>::~EdgesInfinite() = default; // default dtor
template<typename Scalar>
EdgesInfinite<Scalar>::EdgesInfinite(EdgesInfinite &&other) = default; // default move ctor
template<typename Scalar>
EdgesInfinite<Scalar> &EdgesInfinite<Scalar>::operator=(EdgesInfinite &&other) = default; // default move assign
template<typename Scalar>
EdgesInfinite<Scalar>::EdgesInfinite(const EdgesInfinite &other)
    : eneL(std::make_unique<EnvEne<Scalar>>(*other.eneL)), eneR(std::make_unique<EnvEne<Scalar>>(*other.eneR)),
      varL(std::make_unique<EnvVar<Scalar>>(*other.varL)), varR(std::make_unique<EnvVar<Scalar>>(*other.varR)) {}

template<typename Scalar>
EdgesInfinite<Scalar> &EdgesInfinite<Scalar>::operator=(const EdgesInfinite &other) {
    // check for self-assignment
    if(this != &other) {
        eneL = std::make_unique<EnvEne<Scalar>>(*other.eneL);
        varL = std::make_unique<EnvVar<Scalar>>(*other.varL);
        eneR = std::make_unique<EnvEne<Scalar>>(*other.eneR);
        varR = std::make_unique<EnvVar<Scalar>>(*other.varR);
    }
    return *this;
}

template<typename Scalar>
void EdgesInfinite<Scalar>::initialize() {
    eneL = std::make_unique<EnvEne<Scalar>>(0, "L", "ene");
    varL = std::make_unique<EnvVar<Scalar>>(0, "L", "var");
    eneR = std::make_unique<EnvEne<Scalar>>(1, "R", "ene");
    varR = std::make_unique<EnvVar<Scalar>>(1, "R", "var");
}

template<typename Scalar>
void EdgesInfinite<Scalar>::eject_edges() {
    eneL->clear();
    eneR->clear();
    varL->clear();
    varR->clear();
}

template<typename T, typename = std::void_t<>>
struct has_validity : public std::false_type {};
template<typename T>
struct has_validity<T, std::void_t<decltype(std::declval<T>().assertValidity())>> : public std::true_type {};
template<typename T>
inline constexpr bool has_validity_v = has_validity<T>::value;

template<typename Scalar>
size_t EdgesInfinite<Scalar>::get_length() const {
    if(not num::all_equal(eneL->get_sites(), eneR->get_sites(), varL->get_sites(), varR->get_sites()))
        throw except::runtime_error("Site mismatch in edges: eneL {} | eneR {} | varL {} | varR {}", eneL->get_sites(), eneR->get_sites(), varL->get_sites(),
                                    varR->get_sites());
    return eneL->get_sites() + eneR->get_sites() + 2;
}

template<typename Scalar>
size_t EdgesInfinite<Scalar>::get_position() const {
    if(not num::all_equal(eneL->get_position(), varL->get_position()))
        throw except::runtime_error("Position mismatch in edges: eneL {} | varL {}", eneL->get_position(), varL->get_position());
    if(not num::all_equal(eneR->get_position(), varR->get_position()))
        throw except::runtime_error("Position mismatch in edges: eneR {} | varR {}", eneR->get_position(), varR->get_position());
    return eneL->get_position();
}

/* clang-format off */
template<typename Scalar> bool EdgesInfinite<Scalar>::is_real() const {
    return eneL->is_real() and
           eneR->is_real() and
           varL->is_real() and
           varR->is_real();
}

template<typename Scalar>bool EdgesInfinite<Scalar>::has_nan() const {
    return eneL->has_nan() or
           eneR->has_nan() or
           varL->has_nan() or
           varR->has_nan();
}

template<typename Scalar>void EdgesInfinite<Scalar>::assert_validity() const {
    eneL->assert_validity();
    eneR->assert_validity();
    varL->assert_validity();
    varR->assert_validity();
}
/* clang-format on */

template<typename Scalar>
env_pair<const EnvEne<Scalar> &> EdgesInfinite<Scalar>::get_ene() const {
    return {*eneL, *eneR};
}
template<typename Scalar>
env_pair<const EnvVar<Scalar> &> EdgesInfinite<Scalar>::get_var() const {
    return {*varL, *varR};
}
template<typename Scalar>
env_pair<EnvEne<Scalar> &> EdgesInfinite<Scalar>::get_ene() {
    return {*eneL, *eneR};
}
template<typename Scalar>
env_pair<EnvVar<Scalar> &> EdgesInfinite<Scalar>::get_var() {
    return {*varL, *varR};
}

template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> EdgesInfinite<Scalar>::get_env_ene_blk() const {
    return {eneL->get_block(), eneR->get_block()};
}
template<typename Scalar>
env_pair<const Eigen::Tensor<Scalar, 3> &> EdgesInfinite<Scalar>::get_env_var_blk() const {
    return {varL->get_block(), varR->get_block()};
}
template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3> &> EdgesInfinite<Scalar>::get_env_ene_blk() {
    return {eneL->get_block(), eneR->get_block()};
}
template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3> &> EdgesInfinite<Scalar>::get_env_var_blk() {
    return {varL->get_block(), varR->get_block()};
}

template<typename Scalar>
template<typename T>
env_pair<Eigen::Tensor<T, 3>> EdgesInfinite<Scalar>::get_env_ene_blk_as() const {
    return {eneL->template get_block_as<T>(), eneR->template get_block_as<T>()};
}
template env_pair<Eigen::Tensor<fp32, 3>> EdgesInfinite<>::get_env_ene_blk_as() const;
template env_pair<Eigen::Tensor<fp64, 3>> EdgesInfinite<>::get_env_ene_blk_as() const;
template env_pair<Eigen::Tensor<cx32, 3>> EdgesInfinite<>::get_env_ene_blk_as() const;
template env_pair<Eigen::Tensor<cx64, 3>> EdgesInfinite<>::get_env_ene_blk_as() const;

template<typename Scalar>
template<typename T>
env_pair<Eigen::Tensor<T, 3>> EdgesInfinite<Scalar>::get_env_var_blk_as() const {
    return {varL->template get_block_as<T>(), varR->template get_block_as<T>()};
}

template env_pair<Eigen::Tensor<fp32, 3>> EdgesInfinite<>::get_env_var_blk_as() const;
template env_pair<Eigen::Tensor<fp64, 3>> EdgesInfinite<>::get_env_var_blk_as() const;
template env_pair<Eigen::Tensor<cx32, 3>> EdgesInfinite<>::get_env_var_blk_as() const;
template env_pair<Eigen::Tensor<cx64, 3>> EdgesInfinite<>::get_env_var_blk_as() const;