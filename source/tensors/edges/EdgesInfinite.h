#pragma once
#include "tensors/site/env/EnvPair.h"
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;

template<typename Scalar>
class EdgesInfinite {
    private:
    std::unique_ptr<EnvEne<Scalar>> eneL;
    std::unique_ptr<EnvEne<Scalar>> eneR;
    std::unique_ptr<EnvVar<Scalar>> varL;
    std::unique_ptr<EnvVar<Scalar>> varR;

    public:
    EdgesInfinite();
    ~EdgesInfinite();                                     // Read comment on implementation
    EdgesInfinite(EdgesInfinite &&other);                 // default move ctor
    EdgesInfinite &operator=(EdgesInfinite &&other);      // default move assign
    EdgesInfinite(const EdgesInfinite &other);            // copy ctor
    EdgesInfinite &operator=(const EdgesInfinite &other); // copy assign

    void                 initialize();
    void                 eject_edges();
    void                 assert_validity() const;
    [[nodiscard]] size_t get_length() const;
    size_t               get_position() const; // pos of eneL or varL
    [[nodiscard]] bool   is_real() const;
    [[nodiscard]] bool   has_nan() const;

    [[nodiscard]] env_pair<const EnvEne<Scalar> &> get_ene() const;
    [[nodiscard]] env_pair<const EnvVar<Scalar> &> get_var() const;
    [[nodiscard]] env_pair<EnvEne<Scalar> &>       get_ene();
    [[nodiscard]] env_pair<EnvVar<Scalar> &>       get_var();

    env_pair<const Eigen::Tensor<Scalar, 3> &> get_env_ene_blk() const;
    env_pair<const Eigen::Tensor<Scalar, 3> &> get_env_var_blk() const;
    env_pair<Eigen::Tensor<Scalar, 3>>       get_env_ene_blk();
    env_pair<Eigen::Tensor<Scalar, 3>>       get_env_var_blk();
};
