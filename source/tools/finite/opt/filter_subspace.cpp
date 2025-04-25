#include "../opt_mps.h"
#include "general/iter.h"
#include "math/cast.h"
#include "tools/common/log.h"
#include "tools/finite/opt/opt-internal.h"

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

using tools::finite::opt::MatrixReal;
using tools::finite::opt::MatrixType;
using tools::finite::opt::RealScalar;
using tools::finite::opt::VectorReal;
using tools::finite::opt::VectorType;

template<typename Scalar>
void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<Scalar>> &subspace, size_t max_accept) {
    // Sort the eigvec list in order of descending overlaps. If the overlaps are the same, compare instead the distance in energy to the
    // current energy
    std::sort(subspace.begin(), subspace.end(), std::greater<>());
    size_t initial_size                       = subspace.size();
    size_t min_accept                         = std::min(std::min(max_accept, 32ul), initial_size);
    max_accept                                = std::min(max_accept, initial_size);
    auto               subspace_errors        = subspace::get_subspace_errors(subspace); // Vector with decreasing (cumulative) subspace errors, 1-eps(eigvec_i)
    RealScalar<Scalar> subspace_error_initial = subspace_errors.back();
    while(true) {
        if(subspace.size() <= max_accept) break;
        if(subspace.back().is_basis_vector) subspace_errors.pop_back();
        subspace.pop_back();
    }
    RealScalar<Scalar> subspace_error_final = subspace_errors.back();
    size_t             idx                  = 0;
    for(auto &eigvec : subspace) {
        eigvec.set_name(fmt::format("eigenvector {}", idx++));
        //        tools::log->trace("Filtered {:<16}: overlap {:.16f} | energy {:>20.16f}", eigvec.get_label(), eigvec.get_overlap(),
        //                          eigvec.get_energy());
    }

    tools::log->trace("Filtered from {} down to {} states", initial_size, subspace.size());
    tools::log->trace("Filter changed subspace error = {:8.2e} --> {:8.2e}", fp(subspace_error_initial), fp(subspace_error_final));
    if(subspace.size() < min_accept) throw std::runtime_error("Filtered too many eigvecs");
}
template void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<fp32>> &subspace, size_t max_accept);
template void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<fp64>> &subspace, size_t max_accept);
template void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<fp128>> &subspace, size_t max_accept);
template void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<cx32>> &subspace, size_t max_accept);
template void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<cx64>> &subspace, size_t max_accept);
template void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<cx128>> &subspace, size_t max_accept);

template<typename Scalar>
std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<Scalar>> &eigvecs) {
    if(eigvecs.empty()) return std::nullopt;
    auto overlaps = get_overlaps(eigvecs);
    long idx      = 0;

    // Now we have a list of overlaps where nonzero elements correspond to eigvecs inside the energy window
    // Get the index to the highest overlapping element
    auto max_overlap_val = overlaps.maxCoeff(&idx);
    if(max_overlap_val == 0) {
        tools::log->warn("No overlapping eigenstates");
        return std::nullopt;
    }
    return idx;
}

template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<fp32>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<fp64>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<fp128>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<cx32>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<cx64>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<cx128>> &eigvecs);

template<typename Scalar>
std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<Scalar>> &eigvecs) {
    if(eigvecs.empty()) return std::nullopt;
    auto   var = std::numeric_limits<RealScalar<Scalar>>::infinity();
    size_t idx = 0;
    for(const auto &[i, eigvec] : iter::enumerate(eigvecs)) {
        if(not eigvec.is_basis_vector) continue;
        if(eigvec.get_variance() < var) {
            idx = i;
            var = eigvec.get_variance();
        }
    }
    if(std::isinf(var)) return std::nullopt;
    return idx;
}
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<fp32>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<fp64>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<fp128>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<cx32>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<cx64>> &eigvecs);
template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<cx128>> &eigvecs);

template<typename Scalar>
std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<Scalar>> &eigvecs,
                                                                                                   size_t                              max_eigvecs) {
    if(eigvecs.empty()) return std::vector<size_t>();
    auto overlaps = get_overlaps(eigvecs);
    long idx      = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        idx++;
    }
    // Now we have a list of overlaps where nonzero elements correspond to eigvecs inside the energy window
    // Get the index to the highest overlapping element
    auto max_overlap_val = overlaps.maxCoeff();
    if(max_overlap_val == 0) {
        tools::log->warn("No overlapping eigenstates");
        return std::vector<size_t>();
    }

    std::vector<size_t>             best_idx;
    std::vector<RealScalar<Scalar>> best_val;
    // We collect the best eigvecs from this list until the squared sum of their overlaps goes above a certain threshold.
    // Note that the squared sum overlap = 1 if the states in the list form a complete basis for the current state.
    while(true) {
        max_overlap_val = overlaps.maxCoeff(&idx);
        if(max_overlap_val == 0) break;
        best_idx.emplace_back(idx);
        const auto &eigvec = *std::next(eigvecs.begin(), safe_cast<long>(idx));
        if(eigvec.is_basis_vector) best_val.emplace_back(max_overlap_val);
        if(best_idx.size() > max_eigvecs) break;
        auto sq_sum_overlap = overlaps.cwiseAbs2().sum();
        if(sq_sum_overlap > RealScalar<Scalar>{0.6f}) break; // Half means cat state.
        overlaps(idx) = 0;                                   // Zero out the current best so the next-best is found in the next iteration
    }
    tools::log->debug("Found {} eigvecs with good overlap in energy window", best_idx.size());
    return best_idx;
}
/* clang-format off */
template std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<fp32>> &eigvecs, size_t max_eigvecs);
template std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<fp64>> &eigvecs, size_t max_eigvecs);
template std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<fp128>> &eigvecs, size_t max_eigvecs);
template std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<cx32>> &eigvecs, size_t max_eigvecs);
template std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<cx64>> &eigvecs, size_t max_eigvecs);
template std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<cx128>> &eigvecs, size_t max_eigvecs);
/* clang-format on */

template<typename Scalar>
MatrixType<Scalar> tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<Scalar>> &eigvecs) {
    long               rows = eigvecs.front().get_tensor().size();
    long               cols = safe_cast<long>(eigvecs.size());
    MatrixType<Scalar> eigvecs_mat(rows, cols);
    long               idx = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        eigvecs_mat.col(idx++) = Eigen::Map<const VectorType<Scalar>>(eigvec.get_tensor().data(), rows);
    }
    eigvecs_mat.conservativeResize(rows, idx);
    return eigvecs_mat;
}
template MatrixType<fp32>  tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<fp32>> &eigvecs);
template MatrixType<fp64>  tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<fp64>> &eigvecs);
template MatrixType<fp128> tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<fp128>> &eigvecs);
template MatrixType<cx32>  tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<cx32>> &eigvecs);
template MatrixType<cx64>  tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<cx64>> &eigvecs);
template MatrixType<cx128> tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<cx128>> &eigvecs);

template<typename Scalar>
VectorReal<Scalar> tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<Scalar>> &eigvecs) {
    long               size = safe_cast<long>(eigvecs.size());
    VectorReal<Scalar> eigvals(size);
    long               idx = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        eigvals(idx++) = eigvec.get_energy_shifted();
    }
    eigvals.conservativeResize(idx);
    return eigvals;
}

template VectorReal<fp32>  tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<fp32>> &eigvecs);
template VectorReal<fp64>  tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<fp64>> &eigvecs);
template VectorReal<fp128> tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<fp128>> &eigvecs);
template VectorReal<cx32>  tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<cx32>> &eigvecs);
template VectorReal<cx64>  tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<cx64>> &eigvecs);
template VectorReal<cx128> tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<cx128>> &eigvecs);

template<typename Scalar>
VectorReal<Scalar> tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<Scalar>> &eigvecs) {
    VectorReal<Scalar> energies(safe_cast<long>(eigvecs.size()));
    long               idx = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        energies(idx++) = eigvec.get_energy();
    }
    energies.conservativeResize(idx);
    return energies;
}
template VectorReal<fp32>  tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<fp32>> &eigvecs);
template VectorReal<fp64>  tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<fp64>> &eigvecs);
template VectorReal<fp128> tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<fp128>> &eigvecs);
template VectorReal<cx32>  tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<cx32>> &eigvecs);
template VectorReal<cx64>  tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<cx64>> &eigvecs);
template VectorReal<cx128> tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<cx128>> &eigvecs);

template<typename Scalar>
RealScalar<Scalar> tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<Scalar>> &eigvecs, std::optional<size_t> max_eigvecs) {
    RealScalar<Scalar> eps         = 0;
    size_t             num_eigvecs = 0;
    if(not max_eigvecs) max_eigvecs = eigvecs.size();
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        eps += eigvec.get_overlap() * eigvec.get_overlap();
        num_eigvecs += 1;
        if(num_eigvecs >= max_eigvecs.value()) break;
    }
    return RealScalar<Scalar>{1} - eps;
}
template fp32  tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<fp32>> &eigvecs, std::optional<size_t> max_eigvecs);
template fp64  tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<fp64>> &eigvecs, std::optional<size_t> max_eigvecs);
template fp128 tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<fp128>> &eigvecs, std::optional<size_t> max_eigvecs);
template fp32  tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<cx32>> &eigvecs, std::optional<size_t> max_eigvecs);
template fp64  tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<cx64>> &eigvecs, std::optional<size_t> max_eigvecs);
template fp128 tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<cx128>> &eigvecs, std::optional<size_t> max_eigvecs);

template<typename Scalar>
std::vector<RealScalar<Scalar>> tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<Scalar>> &eigvecs) {
    RealScalar<Scalar>              eps = 0;
    std::vector<RealScalar<Scalar>> subspace_errors;
    subspace_errors.reserve(eigvecs.size());
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        eps += eigvec.get_overlap() * eigvec.get_overlap();
        subspace_errors.emplace_back(1 - eps);
    }
    return subspace_errors;
}
template std::vector<fp32>  tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<fp32>> &eigvecs);
template std::vector<fp64>  tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<fp64>> &eigvecs);
template std::vector<fp128> tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<fp128>> &eigvecs);
template std::vector<fp32>  tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<cx32>> &eigvecs);
template std::vector<fp64>  tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<cx64>> &eigvecs);
template std::vector<fp128> tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<cx128>> &eigvecs);

template<typename Scalar>
RealScalar<Scalar> tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<RealScalar<Scalar>> &overlaps) {
    RealScalar<Scalar> eps = 0;
    for(const auto &overlap : overlaps) { eps += overlap * overlap; }
    return RealScalar<Scalar>{1} - eps;
}
template fp32  tools::finite::opt::internal::subspace::get_subspace_error<fp32>(const std::vector<fp32> &overlaps);
template fp64  tools::finite::opt::internal::subspace::get_subspace_error<fp64>(const std::vector<fp64> &overlaps);
template fp128 tools::finite::opt::internal::subspace::get_subspace_error<fp128>(const std::vector<fp128> &overlaps);
template fp32  tools::finite::opt::internal::subspace::get_subspace_error<cx32>(const std::vector<fp32> &overlaps);
template fp64  tools::finite::opt::internal::subspace::get_subspace_error<cx64>(const std::vector<fp64> &overlaps);
template fp128 tools::finite::opt::internal::subspace::get_subspace_error<cx128>(const std::vector<fp128> &overlaps);

template<typename Scalar>
VectorReal<Scalar> tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<Scalar>> &eigvecs) {
    VectorReal<Scalar> overlaps(safe_cast<long>(eigvecs.size()));
    long               idx = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        overlaps(idx++) = eigvec.get_overlap();
    }
    overlaps.conservativeResize(idx);
    return overlaps;
}
template VectorReal<fp32>  tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<fp32>> &eigvecs);
template VectorReal<fp64>  tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<fp64>> &eigvecs);
template VectorReal<fp128> tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<fp128>> &eigvecs);
template VectorReal<cx32>  tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<cx32>> &eigvecs);
template VectorReal<cx64>  tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<cx64>> &eigvecs);
template VectorReal<cx128> tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<cx128>> &eigvecs);

template<typename Scalar>
std::pair<RealScalar<Scalar>, size_t> find_max_overlap(const std::vector<RealScalar<Scalar>> &overlaps) {
    auto max_it  = max_element(overlaps.begin(), overlaps.end());
    auto max_val = *max_it;
    auto max_idx = std::distance(overlaps.begin(), max_it);
    return {max_val, safe_cast<size_t>(max_idx)};
}

template<typename Scalar>
VectorType<Scalar> tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<Scalar>> &eigvecs, size_t idx) {
    // In this function we project a vector to the subspace spanned by a small set of eigenvectors
    // Essentially this old computation
    //      VectorType<Scalar> subspace_vector = (eigvecs.adjoint() * fullspace_vector).normalized();

    auto               fullspace_vector = std::next(eigvecs.begin(), safe_cast<long>(idx))->get_vector();
    long               subspace_size    = safe_cast<long>(eigvecs.size());
    VectorType<Scalar> subspace_vector(subspace_size);
    long               subspace_idx = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        subspace_vector(subspace_idx++) = eigvec.get_vector().dot(fullspace_vector);
    }
    subspace_vector.conservativeResize(subspace_idx);
    return subspace_vector.normalized();
}
template VectorType<fp32>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<fp32>> &eigvecs, size_t idx);
template VectorType<fp64>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<fp64>> &eigvecs, size_t idx);
template VectorType<fp128> tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<fp128>> &eigvecs, size_t idx);
template VectorType<cx32>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<cx32>> &eigvecs, size_t idx);
template VectorType<cx64>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<cx64>> &eigvecs, size_t idx);
template VectorType<cx128> tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<cx128>> &eigvecs, size_t idx);

template<typename Scalar>
VectorType<Scalar> tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<Scalar>> &eigvecs,
                                                                                  const VectorType<Scalar>           &fullspace_vector) {
    // In this function we project a vector to the subspace spanned by a small set of eigenvectors
    // Essentially this old computation
    //      VectorType<Scalar> subspace_vector = (eigvecs.adjoint() * fullspace_vector).normalized();
    long               subspace_size = safe_cast<long>(eigvecs.size());
    VectorType<Scalar> subspace_vector(subspace_size);
    long               subspace_idx = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        subspace_vector(subspace_idx++) = eigvec.get_vector().dot(fullspace_vector);
    }
    subspace_vector.conservativeResize(subspace_idx);
    return subspace_vector.normalized();
}
/* clang-format off */
template VectorType<fp32>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<fp32>> &eigvecs, const VectorType<fp32> &fullspace_vector);
template VectorType<fp64>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<fp64>> &eigvecs, const VectorType<fp64> &fullspace_vector);
template VectorType<fp128> tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<fp128>> &eigvecs, const VectorType<fp128> &fullspace_vector);
template VectorType<cx32>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<cx32>> &eigvecs, const VectorType<cx32> &fullspace_vector);
template VectorType<cx64>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<cx64>> &eigvecs, const VectorType<cx64> &fullspace_vector);
template VectorType<cx128> tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<cx128>> &eigvecs, const VectorType<cx128> &fullspace_vector);
/* clang-format on */

template<typename Scalar>
VectorType<Scalar> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<Scalar>> &eigvecs,
                                                                                   const VectorType<Scalar>           &subspace_vector) {
    // In this function we project a subspace vector back to the full space
    // Essentially this old computation
    //     VectorType<Scalar> fullspace_vector = (eigvecs * subspace_vector.asDiagonal()).rowwise().sum().normalized();
    //
    // The subspace_vector is a list of weights corresponding to each eigenvector that we use to form a linear combination.
    // Therefore, all we need to do to obtain the fullspace vector is to actually add up the linear combination with those weights.

    long               fullspace_size = eigvecs.front().get_vector().size();
    VectorType<Scalar> fullspace_vector(fullspace_size);
    fullspace_vector.setZero();
    long subspace_idx = 0;
    for(const auto &eigvec : eigvecs) {
        if(not eigvec.is_basis_vector) continue;
        fullspace_vector += eigvec.get_vector() * subspace_vector(subspace_idx++);
    }
    return fullspace_vector.normalized();
}
/* clang-format off */
template VectorType<fp32> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<fp32>> &eigvecs, const VectorType<fp32>   &subspace_vector);
template VectorType<fp64> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<fp64>> &eigvecs, const VectorType<fp64>   &subspace_vector);
template VectorType<fp128> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<fp128>> &eigvecs, const VectorType<fp128>   &subspace_vector);
template VectorType<cx32> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<cx32>> &eigvecs, const VectorType<cx32>   &subspace_vector);
template VectorType<cx64> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<cx64>> &eigvecs, const VectorType<cx64>   &subspace_vector);
template VectorType<cx128> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<cx128>> &eigvecs, const VectorType<cx128>   &subspace_vector);
/* clang-format on */

template<typename Scalar>
Eigen::Tensor<Scalar, 3> tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<Scalar>> &eigvecs,
                                                                                         const VectorType<Scalar>           &subspace_vector,
                                                                                         const std::array<Eigen::Index, 3>  &dims) {
    return Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(subspace::get_vector_in_fullspace(eigvecs, subspace_vector).data(), dims);
}
/* clang-format off */
template Eigen::Tensor<fp32, 3>  tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<fp32>> &eigvecs, const VectorType<fp32> &subspace_vector, const std::array<Eigen::Index, 3>  &dims);
template Eigen::Tensor<fp64, 3>  tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<fp64>> &eigvecs, const VectorType<fp64> &subspace_vector, const std::array<Eigen::Index, 3>  &dims);
template Eigen::Tensor<fp128, 3> tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<fp128>> &eigvecs, const VectorType<fp128> &subspace_vector, const std::array<Eigen::Index, 3>  &dims);
template Eigen::Tensor<cx32, 3>  tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<cx32>> &eigvecs, const VectorType<cx32> &subspace_vector, const std::array<Eigen::Index, 3>  &dims);
template Eigen::Tensor<cx64, 3>  tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<cx64>> &eigvecs, const VectorType<cx64> &subspace_vector, const std::array<Eigen::Index, 3>  &dims);
template Eigen::Tensor<cx128, 3> tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<cx128>> &eigvecs, const VectorType<cx128> &subspace_vector, const std::array<Eigen::Index, 3>  &dims);
/* clang-format on */
