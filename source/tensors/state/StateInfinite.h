#pragma once

#include "config/settings.h"
#include "math/svd/config.h"
#include "measure/MeasurementsStateInfinite.h"
#include <memory>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename Scalar>
class MpsSite;

/*!
  \class StateInfinite
  \brief This class contains the current 2-site translationally invariant wave function in MPS form
*/

template<typename Scalar>
class StateInfinite {
    public:
    template<class U> friend class StateInfinite;

    private:
    struct Cache {
        std::optional<Eigen::Tensor<Scalar, 3>> twosite_mps = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 4>> theta       = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 3>> GA          = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 3>> GB          = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> LC_diag     = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> LA_diag     = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> LB_diag     = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> LC_diag_inv = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> LA_diag_inv = std::nullopt;
        std::optional<Eigen::Tensor<Scalar, 2>> LB_diag_inv = std::nullopt;
    };

    std::unique_ptr<MpsSite<Scalar>> MPS_A;
    std::unique_ptr<MpsSite<Scalar>> MPS_B;
    bool                             swapped = false; /*!< Tracks the swapped state of A and B positions. */
    mutable Cache                    cache;
    std::string                      name;
    AlgorithmType                    algo = AlgorithmType::ANY;

    public:
    mutable MeasurementsStateInfinite<Scalar> measurements;
    mutable double                            lowest_recorded_variance = 1.0;

    public:
    StateInfinite();
    ~StateInfinite();                                              // Read comment on implementation
    StateInfinite(StateInfinite &&other) noexcept;                 // default move ctor
    StateInfinite &operator=(StateInfinite &&other) noexcept;      // default move assign
    StateInfinite(const StateInfinite &other) noexcept;            // copy ctor
    StateInfinite &operator=(const StateInfinite &other) noexcept; // copy assign

    template<typename T>
    StateInfinite(const StateInfinite<T> &other) noexcept;

    template<typename T>
    StateInfinite &operator=(const StateInfinite<T> &other) noexcept;
    // {
    //     if constexpr(std::is_same_v<Scalar, T>) {
    //         if(this == &other) return *this; // check for self-assignment
    //     }
    //     MPS_A                    = std::make_unique<MpsSite<Scalar>>(other.get_mps_siteA());
    //     MPS_B                    = std::make_unique<MpsSite<Scalar>>(other.get_mps_siteB());
    //     swapped                  = other.swapped;
    //     name                     = other.name;
    //     algo                     = other.algo;
    //     lowest_recorded_variance = other.lowest_recorded_variance;
    //     if constexpr(std::is_same_v<Scalar, T>) {
    //         cache        = other.cache;
    //         measurements = other.measurements;
    //     }
    //     return *this;
    // }

    void initialize(ModelType model_type);

    void                      set_name(std::string_view statename);
    [[nodiscard]] std::string get_name() const;

    void                        set_algorithm(const AlgorithmType &algo_type);
    [[nodiscard]] AlgorithmType get_algorithm() const;

    void                                    assert_validity() const;
    [[nodiscard]] bool                      is_real() const;
    [[nodiscard]] bool                      has_nan() const;
    [[nodiscard]] bool                      is_swapped() const;
    [[nodiscard]] double                    get_truncation_error() const;
    [[nodiscard]] std::pair<size_t, size_t> get_positions();
    [[nodiscard]] size_t                    get_positionA();
    [[nodiscard]] size_t                    get_positionB();
    [[nodiscard]] long                      chiC() const;
    [[nodiscard]] long                      chiA() const;
    [[nodiscard]] long                      chiB() const;
    [[nodiscard]] long                      get_spin_dimA() const;
    [[nodiscard]] long                      get_spin_dimB() const;
    [[nodiscard]] Eigen::DSizes<long, 3>    dimensions() const;
    [[nodiscard]] const MpsSite<Scalar>    &get_mps_siteA() const;
    [[nodiscard]] const MpsSite<Scalar>    &get_mps_siteB() const;
    [[nodiscard]] MpsSite<Scalar>          &get_mps_siteA();
    [[nodiscard]] MpsSite<Scalar>          &get_mps_siteB();
    [[nodiscard]] const MpsSite<Scalar>    &get_mps_site(size_t pos) const;
    [[nodiscard]] MpsSite<Scalar>          &get_mps_site(size_t pos);
    [[nodiscard]] const MpsSite<Scalar>    &get_mps_site(std::string_view pos) const;
    [[nodiscard]] MpsSite<Scalar>          &get_mps_site(std::string_view pos);

    [[nodiscard]] const Eigen::Tensor<Scalar, 3> &A_bare() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3> &A() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3> &B() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 2> &LC_diag() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 2> &LA_diag() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 2> &LB_diag() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 2> &LC_diag_inv() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 2> &LA_diag_inv() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 2> &LB_diag_inv() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3> &GA() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3> &GB() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 1> &LC() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 1> &LA() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 1> &LB() const;

    [[nodiscard]] const Eigen::Tensor<Scalar, 3> &get_2site_mps(Scalar norm = 1.0) const;
    void                                          set_positions(size_t position);

    void swap_AB(); /*!< Swap the roles of A and B. Used in the infinite-DMRG stage.*/
    void set_mps(const Eigen::Tensor<Scalar, 3> &twosite_tensor, MergeEvent mevent, std::optional<svd::config> svd_cfg);
    void set_mps(const std::vector<MpsSite<Scalar>> &mps_list);
    void set_mps(const MpsSite<Scalar> &mpsA, const MpsSite<Scalar> &mpsB);
    void set_mps(const Eigen::Tensor<Scalar, 3> &MA, const Eigen::Tensor<Scalar, 1> &LC, const Eigen::Tensor<Scalar, 3> &MB);
    void set_mps(const Eigen::Tensor<Scalar, 1> &LA, const Eigen::Tensor<Scalar, 3> &MA, const Eigen::Tensor<Scalar, 1> &LC, const Eigen::Tensor<Scalar, 3> &MB,
                 const Eigen::Tensor<Scalar, 1> &LB);

    bool is_limited_by_bond(long bond_lim) const;
    bool is_truncated(double truncation_error_limit) const;

    void clear_measurements() const;
    void clear_cache() const;
};
