#include "MpoFactory.h"
#include "debug/exceptions.h"
#include "IsingMajorana.h"
#include "IsingRandomField.h"
#include "IsingSelfDual.h"
#include "LBit.h"
#include "MpoSite.h"
#include <memory>

template class MpoFactory<cx64>;
template class MpoFactory<cx128>;

template<typename Scalar>
std::unique_ptr<MpoSite<Scalar>> MpoFactory<Scalar>::create_mpo(size_t position, ModelType model_type) {
    switch(model_type) {
        case ModelType::ising_tf_rf: return std::make_unique<IsingRandomField<Scalar>>(model_type, position);
        case ModelType::ising_sdual: return std::make_unique<IsingSelfDual<Scalar>>(model_type, position);
        case ModelType::ising_majorana: return std::make_unique<IsingMajorana<Scalar>>(model_type, position);
        case ModelType::lbit: return std::make_unique<LBit<Scalar>>(model_type, position);
        default: throw except::runtime_error("Wrong model type: [{}]", enum2sv(model_type));
    }
}

template<typename Scalar>
std::unique_ptr<MpoSite<Scalar>> MpoFactory<Scalar>::clone(std::unique_ptr<MpoSite<Scalar>> other) {
    return other->clone();
}
