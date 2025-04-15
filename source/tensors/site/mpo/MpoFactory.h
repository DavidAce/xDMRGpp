#pragma once

#include "MpoSite.h"
#include <memory>

template<typename Scalar = cx64>
class MpoFactory {
    public:
    static std::unique_ptr<MpoSite<Scalar>> create_mpo(size_t position, ModelType model_type);

    template<typename... T>
    static std::unique_ptr<MpoSite<Scalar>> create_mpo(size_t position, ModelType model_type, T... args) {
        auto mpo = create_mpo(position, model_type);
        mpo->set_parameters(args...);
        return mpo;
    }
    //    static std::unique_ptr<class_hamiltonian_h5log_base> create_table(std::string model_type_str);
    static std::unique_ptr<MpoSite<Scalar>> clone(std::unique_ptr<MpoSite<Scalar>> other);
};
