#pragma once
#include "math/float.h"
#include <cstddef>
#include <memory>

template<typename Scalar> class MpoSite;
enum class ModelType;

template<typename Scalar = cx64>
class MpoFactory {
    public:
    static std::unique_ptr<MpoSite<Scalar>> create_mpo(size_t position, ModelType model_type);
    static std::unique_ptr<MpoSite<Scalar>> clone(std::unique_ptr<MpoSite<Scalar>> other);
};
