#include "impl.h"

using Scalar = cx64;

/* clang-format off */

template size_t tools::finite::mps::move_center_point_single_site(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg);

template size_t tools::finite::mps::move_center_point(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg);

template size_t tools::finite::mps::move_center_point_to_pos(StateFinite<Scalar> &state, long pos, std::optional<svd::config> svd_cfg);

template size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<Scalar> &state, long pos, int dir, std::optional<svd::config> svd_cfg);

template size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg);

template size_t tools::finite::mps::move_center_point_to_middle(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg);
