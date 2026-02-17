#pragma once
#include <memory>
#include <string_view>

template <typename Scalar> class EnvBase;

template<typename Scalar>
class EnvFactory {
    public:
    static std::unique_ptr<EnvBase<Scalar>> create_env(std::string_view tag);
};

