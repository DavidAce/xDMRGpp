#pragma once
#include "EnvBase.h"
#include "EnvEne.h"
#include "EnvFactory.h"
#include "EnvVar.h"
#include <memory>

template<typename Scalar>
std::unique_ptr<EnvBase<Scalar>> EnvFactory<Scalar>::create_env(std::string_view tag) {
    if(tag == "ene") return std::make_unique<EnvEne<Scalar>>();
    if(tag == "var") return std::make_unique<EnvVar<Scalar>>();
    throw std::runtime_error("EnvFactory: Could not match tag");
}
