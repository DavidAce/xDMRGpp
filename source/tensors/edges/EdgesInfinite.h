#pragma once
#pragma once
#include "config/enums.h"
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "tensors/site/env/EnvPair.h"
#include <complex>
#include <memory>
#include <optional>

class EnvEne;
class EnvVar;

class EdgesInfinite {
    public:
    private:
    std::unique_ptr<EnvEne> eneL;
    std::unique_ptr<EnvEne> eneR;
    std::unique_ptr<EnvVar> varL;
    std::unique_ptr<EnvVar> varR;

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

    [[nodiscard]] env_pair<const EnvEne &> get_ene() const;
    [[nodiscard]] env_pair<const EnvVar &> get_var() const;
    [[nodiscard]] env_pair<EnvEne &>       get_ene();
    [[nodiscard]] env_pair<EnvVar &>       get_var();

    env_pair<const Eigen::Tensor<cx64, 3> &> get_env_ene_blk() const;
    env_pair<const Eigen::Tensor<cx64, 3> &> get_env_var_blk() const;
    env_pair<Eigen::Tensor<cx64, 3> &>       get_env_ene_blk();
    env_pair<Eigen::Tensor<cx64, 3> &>       get_env_var_blk();
    template<typename Scalar>
    [[nodiscard]] env_pair<Eigen::Tensor<Scalar, 3>> get_env_ene_blk_as() const;
    template<typename Scalar>
    [[nodiscard]] env_pair<Eigen::Tensor<Scalar, 3>> get_env_var_blk_as() const;
};
