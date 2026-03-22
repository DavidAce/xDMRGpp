#pragma once
#include "config/enum_utils.h"

/*! Selects the physical model Hamiltonian */
enum class ModelType {
    ising_tf_rf,    /*!< Random-field transverse-field Ising model */
    ising_sdual,    /*!< Self-dual Ising model */
    ising_majorana, /*!< Majorana representation of the Ising model */
    lbit,           /*!< l-bit Hamiltonian */
    xxz             /*!< XXZ spin chain */
};
template<> std::string_view enum2sv(ModelType item) noexcept;
template<> ModelType        sv2enum<ModelType>(std::string_view item);
