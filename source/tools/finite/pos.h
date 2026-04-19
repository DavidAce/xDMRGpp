#pragma once
#include "math/svd/config.h"
#include <array>
#include <optional>
#include <vector>

template<typename Scalar>
class TensorsFinite;

namespace tools::finite::pos {
    /* clang-format off */
    template<typename Scalar> [[nodiscard]] bool                has_center_point(const TensorsFinite<Scalar> &tensors);
    template<typename Scalar> [[nodiscard]] bool                position_is_the_middle(const TensorsFinite<Scalar> &tensors);
    template<typename Scalar> [[nodiscard]] bool                position_is_the_middle_any_direction(const TensorsFinite<Scalar> &tensors);
    template<typename Scalar> [[nodiscard]] bool                position_is_outward_edge_left(const TensorsFinite<Scalar> &tensors, size_t nsite = 1);
    template<typename Scalar> [[nodiscard]] bool                position_is_outward_edge_right(const TensorsFinite<Scalar> &tensors, size_t nsite = 1);
    template<typename Scalar> [[nodiscard]] bool                position_is_outward_edge(const TensorsFinite<Scalar> &tensors, size_t nsite = 1);
    template<typename Scalar> [[nodiscard]] bool                position_is_inward_edge_left(const TensorsFinite<Scalar> &tensors, size_t nsite = 1);
    template<typename Scalar> [[nodiscard]] bool                position_is_inward_edge_right(const TensorsFinite<Scalar> &tensors, size_t nsite = 1);
    template<typename Scalar> [[nodiscard]] bool                position_is_inward_edge(const TensorsFinite<Scalar> &tensors, size_t nsite = 1);
    template<typename Scalar> [[nodiscard]] bool                position_is_at(const TensorsFinite<Scalar> &tensors, long pos);
    template<typename Scalar> [[nodiscard]] bool                position_is_at(const TensorsFinite<Scalar> &tensors, long pos, int dir);
    template<typename Scalar> [[nodiscard]] bool                position_is_at(const TensorsFinite<Scalar> &tensors, long pos, int dir, bool isCenter);
    template<typename Scalar> void                              sync_active_sites(TensorsFinite<Scalar> &tensors);
    template<typename Scalar> void                              clear_active_sites(TensorsFinite<Scalar> &tensors);
    template<typename Scalar> void                              activate_sites(TensorsFinite<Scalar> &tensors, const std::vector<size_t> &sites);
    template<typename Scalar> void                              activate_sites(TensorsFinite<Scalar> &tensors);
    template<typename Scalar> void                              activate_sites(TensorsFinite<Scalar> &tensors, long threshold, size_t max_sites, size_t min_sites = 1);
    template<typename Scalar> [[nodiscard]] std::array<long, 3> active_problem_dims(const TensorsFinite<Scalar> &tensors);
    template<typename Scalar> [[nodiscard]] long                active_problem_size(const TensorsFinite<Scalar> &tensors);
    template<typename Scalar> size_t                            move_center_point(TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> size_t                            move_center_point_to_pos(TensorsFinite<Scalar> &tensors, long pos, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> size_t                            move_center_point_to_inward_edge(TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> size_t                            move_center_point_to_middle(TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> void                              move_site_mps(TensorsFinite<Scalar> &tensors, const size_t site, const long steps, std::vector<size_t> &sites_mps, std::optional<long> new_pos = std::nullopt);
    template<typename Scalar> void                              move_site_mpo(TensorsFinite<Scalar> &tensors, const size_t site, const long steps, std::vector<size_t> &sites_mpo);
    template<typename Scalar> void                              move_site_mps_to_pos(TensorsFinite<Scalar> &tensors, const size_t site, const long tgt_pos, std::vector<size_t> &sites_mps, std::optional<long> new_pos = std::nullopt);
    template<typename Scalar> void                              move_site_mpo_to_pos(TensorsFinite<Scalar> &tensors, const size_t site, const long tgt_pos, std::vector<size_t> &sites_mpo);
    template<typename Scalar> void                              move_site(TensorsFinite<Scalar> &tensors, const size_t site, const long steps, std::vector<size_t> &sites_mps, std::vector<size_t> &sites_mpo, std::optional<long> new_pos = std::nullopt);
    template<typename Scalar> void                              move_site_to_pos(TensorsFinite<Scalar> &tensors, const size_t site, const long tgt_pos, std::optional<std::vector<size_t>> &sites_mps, std::optional<std::vector<size_t>> &sites_mpo, std::optional<long> new_pos = std::nullopt);
    /* clang-format on */
}
