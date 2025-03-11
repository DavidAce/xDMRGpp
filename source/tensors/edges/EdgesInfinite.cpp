#include "EdgesInfinite.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvPair.h"
#include "tensors/site/env/EnvVar.h"
#include "tools/common/log.h"

EdgesInfinite::EdgesInfinite()
    : eneL(std::make_unique<EnvEne>()), eneR(std::make_unique<EnvEne>()), varL(std::make_unique<EnvVar>()), varR(std::make_unique<EnvVar>()) {}

// We need to define the destructor and other special functions
// because we enclose data in unique_ptr for this pimpl idiom.
// Otherwise unique_ptr will forcibly inline its own default deleter.
// Here we follow "rule of five", so we must also define
// our own copy/move ctor and copy/move assignments
// This has the side effect that we must define our own
// operator= and copy assignment constructor.
// Read more: https://stackoverflow.com/questions/33212686/how-to-use-unique-ptr-with-forward-declared-type
// And here:  https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
EdgesInfinite::~EdgesInfinite()                                = default; // default dtor
EdgesInfinite::EdgesInfinite(EdgesInfinite &&other)            = default; // default move ctor
EdgesInfinite &EdgesInfinite::operator=(EdgesInfinite &&other) = default; // default move assign
EdgesInfinite::EdgesInfinite(const EdgesInfinite &other)
    : eneL(std::make_unique<EnvEne>(*other.eneL)), eneR(std::make_unique<EnvEne>(*other.eneR)), varL(std::make_unique<EnvVar>(*other.varL)),
      varR(std::make_unique<EnvVar>(*other.varR)) {}

EdgesInfinite &EdgesInfinite::operator=(const EdgesInfinite &other) {
    // check for self-assignment
    if(this != &other) {
        eneL = std::make_unique<EnvEne>(*other.eneL);
        varL = std::make_unique<EnvVar>(*other.varL);
        eneR = std::make_unique<EnvEne>(*other.eneR);
        varR = std::make_unique<EnvVar>(*other.varR);
    }
    return *this;
}

void EdgesInfinite::initialize() {
    eneL = std::make_unique<EnvEne>(0, "L", "ene");
    varL = std::make_unique<EnvVar>(0, "L", "var");
    eneR = std::make_unique<EnvEne>(1, "R", "ene");
    varR = std::make_unique<EnvVar>(1, "R", "var");
}

void EdgesInfinite::eject_edges() {
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

size_t EdgesInfinite::get_length() const {
    if(not num::all_equal(eneL->get_sites(), eneR->get_sites(), varL->get_sites(), varR->get_sites()))
        throw except::runtime_error("Site mismatch in edges: eneL {} | eneR {} | varL {} | varR {}", eneL->get_sites(), eneR->get_sites(), varL->get_sites(),
                                    varR->get_sites());
    return eneL->get_sites() + eneR->get_sites() + 2;
}

size_t EdgesInfinite::get_position() const {
    if(not num::all_equal(eneL->get_position(), varL->get_position()))
        throw except::runtime_error("Position mismatch in edges: eneL {} | varL {}", eneL->get_position(), varL->get_position());
    if(not num::all_equal(eneR->get_position(), varR->get_position()))
        throw except::runtime_error("Position mismatch in edges: eneR {} | varR {}", eneR->get_position(), varR->get_position());
    return eneL->get_position();
}

/* clang-format off */
bool EdgesInfinite::is_real() const {
    return eneL->is_real() and
           eneR->is_real() and
           varL->is_real() and
           varR->is_real();
}

bool EdgesInfinite::has_nan() const {
    return eneL->has_nan() or
           eneR->has_nan() or
           varL->has_nan() or
           varR->has_nan();
}

void EdgesInfinite::assert_validity() const {
    eneL->assert_validity();
    eneR->assert_validity();
    varL->assert_validity();
    varR->assert_validity();
}
/* clang-format on */

env_pair<const EnvEne &> EdgesInfinite::get_ene() const { return {*eneL, *eneR}; }
env_pair<const EnvVar &> EdgesInfinite::get_var() const { return {*varL, *varR}; }
env_pair<EnvEne &>       EdgesInfinite::get_ene() { return {*eneL, *eneR}; }
env_pair<EnvVar &>       EdgesInfinite::get_var() { return {*varL, *varR}; }

env_pair<const Eigen::Tensor<cx64, 3> &> EdgesInfinite::get_env_ene_blk() const { return {eneL->get_block(), eneR->get_block()}; }
env_pair<const Eigen::Tensor<cx64, 3> &> EdgesInfinite::get_env_var_blk() const { return {varL->get_block(), varR->get_block()}; }
env_pair<Eigen::Tensor<cx64, 3> &>       EdgesInfinite::get_env_ene_blk() { return {eneL->get_block(), eneR->get_block()}; }
env_pair<Eigen::Tensor<cx64, 3> &>       EdgesInfinite::get_env_var_blk() { return {varL->get_block(), varR->get_block()}; }

template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3>> EdgesInfinite::get_env_ene_blk_as() const {
    return {eneL->get_block_as<Scalar>(), eneR->get_block_as<Scalar>()};
}
template env_pair<Eigen::Tensor<fp32, 3>> EdgesInfinite::get_env_ene_blk_as() const;
template env_pair<Eigen::Tensor<fp64, 3>> EdgesInfinite::get_env_ene_blk_as() const;
template env_pair<Eigen::Tensor<cx32, 3>> EdgesInfinite::get_env_ene_blk_as() const;
template env_pair<Eigen::Tensor<cx64, 3>> EdgesInfinite::get_env_ene_blk_as() const;

template<typename Scalar>
env_pair<Eigen::Tensor<Scalar, 3>> EdgesInfinite::get_env_var_blk_as() const {
    return {varL->get_block_as<Scalar>(), varR->get_block_as<Scalar>()};
}

template env_pair<Eigen::Tensor<fp32, 3>> EdgesInfinite::get_env_var_blk_as() const;
template env_pair<Eigen::Tensor<fp64, 3>> EdgesInfinite::get_env_var_blk_as() const;
template env_pair<Eigen::Tensor<cx32, 3>> EdgesInfinite::get_env_var_blk_as() const;
template env_pair<Eigen::Tensor<cx64, 3>> EdgesInfinite::get_env_var_blk_as() const;