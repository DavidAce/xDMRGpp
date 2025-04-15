#include "qm/mpo.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tools/common/log.h"
#include <fmt/ranges.h>

namespace qm::mpo {
    template<typename Scalar>
    std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> pauli_mpo(const Eigen::MatrixXcd &paulimatrix)
    /*! Builds the MPO string for measuring  spin on many-body systems.
     *      P = Π  σ_{i}
     * where Π is the product sites=0...L-1 and σ_{i} is the given pauli matrix for site i.
     *
     * MPO = | s | (a 1 by 1 matrix with a single pauli matrix element)
     *
     *        2
     *        |
     *    0---s---1
     *        |
     *        3
     *
     */
    {
        long                     spin_dim = paulimatrix.rows();
        std::array<long, 4>      extent4  = {1, 1, spin_dim, spin_dim}; /*!< Extent of pauli matrices in a rank-4 tensor */
        std::array<long, 2>      extent2  = {spin_dim, spin_dim};       /*!< Extent of pauli matrices in a rank-2 tensor */
        Eigen::Tensor<Scalar, 4> MPO(1, 1, spin_dim, spin_dim);
        MPO.setZero();
        MPO.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = tenx::asScalarType<Scalar>(tenx::TensorMap(paulimatrix));

        // Create compatible edges
        Eigen::Tensor<Scalar, 3> Ledge(1, 1, 1); // The left  edge
        Eigen::Tensor<Scalar, 3> Redge(1, 1, 1); // The right edge
        Ledge(0, 0, 0) = 1;
        Redge(0, 0, 0) = 1;
        return {MPO, Ledge, Redge};
    }
    template<typename Scalar>
    std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> prod_pauli_mpo(std::string_view axis) {
        return qm::mpo::pauli_mpo<Scalar>(spin::half::get_pauli(axis));
    }

    std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>> parity_projector_mpos(const Eigen::MatrixXcd &paulimatrix,
                                                                                                                          size_t sites, int sign)
    /*! Builds the MPO that projects out the MPS component in a parity sector.
     * |psi+->  = Proj(σ,p) |psi>=  1/2 (I + pP(σ) |psi>
     * Here
     *      1/2 is an optional normalization constant, but note that Proj does not preserve normalization in general.
     *      I = outer product of L 2x2 identity matrices, i.e. ⊗_{i=0}^{L-1} I_2x2
     *      p is either +-1 to select the parity sector.
     *      P = outer product of L 2x2 pauli matrices, i.e. ⊗_{i=0}^{L-1} σ
     *
     * The projection operator is sum of operator strings which can be expressed as a string of MPO's:
     *
     *                   Ledge       O_1        O_2                               O_L       Redge
     *                   | 1/2 |  | I   0  | | I   0  | ... repeat L times ... | I   0  |   | 1 |
     *    Proj(σ,p)  =   | p/2 |  | 0   σ  | | 0   σ  | ... repeat L times ... | 0   σ  |   | 1 |
     *
     * Note how the sign p and factor 1/2 are appended at the edge of the operator string
     *        2
     *        |
     *    0---O---1
     *        |
     *        3
     *
     */
    {
        long                   spin_dim = paulimatrix.rows();
        auto                   I        = Eigen::MatrixXcd::Identity(spin_dim, spin_dim).eval();
        std::array<long, 4>    extent4  = {1, 1, spin_dim, spin_dim}; /*!< Extent of pauli matrices in a rank-4 tensor */
        std::array<long, 2>    extent2  = {spin_dim, spin_dim};       /*!< Extent of pauli matrices in a rank-2 tensor */
        Eigen::Tensor<cx64, 4> MPO(2, 2, spin_dim, spin_dim);
        MPO.setZero();
        MPO.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(I);
        MPO.slice(std::array<long, 4>{1, 1, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(paulimatrix);

        std::vector<Eigen::Tensor<cx64, 4>> mpos(sites, MPO);
        // Create compatible edges
        Eigen::Tensor<cx64, 3> Ledge(1, 1, 2); // The left  edge
        Eigen::Tensor<cx64, 3> Redge(1, 1, 2); // The right edge
        Ledge(0, 0, 0) = 0.5;                  // 0.5;
        Ledge(0, 0, 1) = 0.5 * sign;
        Redge(0, 0, 0) = 1;
        Redge(0, 0, 1) = 1;

        return {mpos, Ledge, Redge};
    }

    std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>> random_pauli_mpos(const Eigen::MatrixXcd &paulimatrix,
                                                                                                                      size_t                  sites)
    /*! Builds a string of random pauli matrix MPO's
     *      P = Π  O_i
     * where Π is the product over all sites, and O_i is one of {σ, I} on site i, where σ and I is a pauli matrix or an identity matrix, respectively
     *
     * MPO = | O |
     *
     *        2
     *        |
     *    0---O---1
     *        |
     *        3
     *
     */
    {
        long                   spin_dim = paulimatrix.rows();
        std::array<long, 4>    extent4  = {1, 1, spin_dim, spin_dim}; /*!< Extent of pauli matrices in a rank-4 tensor */
        std::array<long, 2>    extent2  = {spin_dim, spin_dim};       /*!< Extent of pauli matrices in a rank-2 tensor */
        Eigen::Tensor<cx64, 4> MPO_I(1, 1, spin_dim, spin_dim);
        Eigen::Tensor<cx64, 4> MPO_S(1, 1, spin_dim, spin_dim);
        MPO_I.setZero();
        MPO_S.setZero();
        MPO_I.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(paulimatrix);
        MPO_S.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = tenx::TensorCast(Eigen::MatrixXcd::Identity(spin_dim, spin_dim));

        // We have to push in an even number of pauli matrices to retain the parity sector.
        // Choosing randomly
        std::vector<int> binary(sites, -1);
        int              sum = 0;
        while(true) {
            binary[rnd::uniform_integer_box<size_t>(0, sites)] *= -1;
            sum = std::accumulate(binary.begin(), binary.end(), 0);
            if((num::mod<size_t>(sites, 2) == 0 and sum == 0) or (num::mod<size_t>(sites, 2) == 1 and sum == 1)) break;
        }

        std::vector<Eigen::Tensor<cx64, 4>> mpos;
        for(auto &val : binary) {
            if(val < 0)
                mpos.push_back(MPO_S);
            else
                mpos.push_back(MPO_I);
        }

        // Create compatible edges
        Eigen::Tensor<cx64, 3> Ledge(1, 1, 1); // The left  edge
        Eigen::Tensor<cx64, 3> Redge(1, 1, 1); // The right edge
        Ledge(0, 0, 0) = 1;
        Redge(0, 0, 0) = 1;
        return std::make_tuple(mpos, Ledge, Redge);
    }

    std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        random_pauli_mpos_x2(const Eigen::MatrixXcd &paulimatrix1, const Eigen::MatrixXcd &paulimatrix2, const size_t sites)
    /*! Builds a string of random pauli matrix MPO's
     *      P = Π  O_i
     * where Π is the product over all sites, and O_i is one of {S, I} on site i.
     * S is the sum of pauli matrices s1 and s2, and where I is an identity matrix of the same size
     *            | s1  0  |
     * S   =      | 0   s2 |
     *
     *            | id  0  |
     * I   =      | 0   id |
     *
     *
     *        2
     *        |
     *    0---O---1
     *        |
     *        3
     *
     */
    {
        if(paulimatrix1.rows() != paulimatrix2.rows()) throw except::logic_error("Pauli matrices must be of equal size");
        long                   spin_dim = paulimatrix1.rows();
        auto                   I        = Eigen::MatrixXcd::Identity(spin_dim, spin_dim).eval();
        std::array<long, 4>    extent4  = {1, 1, spin_dim, spin_dim}; /*!< Extent of pauli matrices in a rank-4 tensor */
        std::array<long, 2>    extent2  = {spin_dim, spin_dim};       /*!< Extent of pauli matrices in a rank-2 tensor */
        Eigen::Tensor<cx64, 4> MPO_S(2, 2, spin_dim, spin_dim);
        Eigen::Tensor<cx64, 4> MPO_I(2, 2, spin_dim, spin_dim);
        Eigen::Tensor<cx64, 4> MPO_P(2, 2, spin_dim, spin_dim);
        MPO_S.setZero();
        MPO_I.setZero();
        MPO_P.setZero();

        MPO_S.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(paulimatrix1);
        MPO_S.slice(std::array<long, 4>{1, 1, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(paulimatrix2);
        MPO_I.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(I);
        MPO_I.slice(std::array<long, 4>{1, 1, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(I);
        MPO_P.slice(std::array<long, 4>{0, 0, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(I);
        MPO_P.slice(std::array<long, 4>{1, 1, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(paulimatrix1);

        // Push in an even number of operators
        std::vector<int> binary(sites, -1);
        int              sum = 0;
        while(true) {
            binary[rnd::uniform_integer_box<size_t>(0, sites)] *= -1;
            sum = std::accumulate(binary.begin(), binary.end(), 0);
            if((num::mod<size_t>(sites, 2) == 0 and sum == 0) or (num::mod<size_t>(sites, 2) == 1 and sum == 1)) break;
        }
        if(binary.size() != sites) throw except::logic_error("Size mismatch");
        // Generate the list
        std::vector<Eigen::Tensor<cx64, 4>> mpos;
        for(auto &val : binary) {
            if(val < 0)
                mpos.push_back(MPO_S);
            else
                mpos.push_back(MPO_I);
        }

        // Create compatible edges
        Eigen::Tensor<cx64, 3> Ledge(1, 1, 2); // The left  edge
        Eigen::Tensor<cx64, 3> Redge(1, 1, 2); // The right edge
        Ledge(0, 0, 0) = 1.0 / std::sqrt(2);
        Ledge(0, 0, 1) = 1.0 / std::sqrt(2);
        Redge(0, 0, 0) = 1;
        Redge(0, 0, 1) = 1;
        return std::make_tuple(mpos, Ledge, Redge);
    }

    std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        random_pauli_mpos(const std::vector<Eigen::MatrixXcd> &paulimatrices, size_t sites)
    /*! Builds a string of random pauli matrix MPO's
     *      P = Π  O_i
     * where Π is the product over all sites, and O_i is one of {S, I} on site i.
     * S is the sum of pauli matrices s0,s1,s2... , and where I is an identity matrix of the same size
     *
     *            | s0  0   0  .  |
     * S   =      | 0   s1  0  .  |
     *            | 0   0  s2  .  |
     *            | .   .   . ... |
     *
     *            | id  0   0  .  |
     * I   =      | 0   id  0  .  |
     *            | 0   0  id  .  |
     *            | .   .   . ... |
     *
     *
     *        2
     *        |
     *    0---O---1
     *        |
     *        3
     *
     */
    {
        if(paulimatrices.empty()) throw except::runtime_error("List of pauli matrices is empty");
        long                   num_paulis = safe_cast<long>(paulimatrices.size());
        long                   spin_dim   = 2;
        auto                   I          = Eigen::MatrixXcd::Identity(spin_dim, spin_dim).eval();
        std::array<long, 4>    extent4    = {1, 1, spin_dim, spin_dim}; /*!< Extent of pauli matrices in a rank-4 tensor */
        std::array<long, 2>    extent2    = {spin_dim, spin_dim};       /*!< Extent of pauli matrices in a rank-2 tensor */
        Eigen::Tensor<cx64, 4> MPO_S(num_paulis, num_paulis, spin_dim, spin_dim);
        Eigen::Tensor<cx64, 4> MPO_I(num_paulis, num_paulis, spin_dim, spin_dim);
        MPO_S.setZero();
        MPO_I.setZero();
        for(long diag_pos = 0; diag_pos < num_paulis; diag_pos++) {
            MPO_S.slice(std::array<long, 4>{diag_pos, diag_pos, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(paulimatrices[safe_cast<size_t>(diag_pos)]);
            MPO_I.slice(std::array<long, 4>{diag_pos, diag_pos, 0, 0}, extent4).reshape(extent2) = tenx::TensorMap(I);
        }

        // Push in an even number of operators
        // This is so that we get a 50% chance of applying a gate.
        std::vector<int> binary(sites, -1);
        int              sum = 0;
        while(true) {
            binary[rnd::uniform_integer_box<size_t>(0, sites - 1)] *= -1;
            sum = std::accumulate(binary.begin(), binary.end(), 0);
            if((num::mod<size_t>(sites, 2) == 0 and sum == 0) or (num::mod<size_t>(sites, 2) == 1 and sum == 1)) break;
        }
        if(binary.size() != sites) throw except::logic_error("Size mismatch");
        // Generate the list
        std::vector<Eigen::Tensor<cx64, 4>> mpos;
        std::vector<std::string>            mpos_str;
        for(auto &val : binary) {
            if(val < 0) {
                mpos.push_back(MPO_S);
                mpos_str.emplace_back("S");
            } else {
                mpos.push_back(MPO_I);
                mpos_str.emplace_back("I");
            }
        }
        // tools::log->warn("Generated random pauli MPO string: {}", mpos_str);
        // Create compatible edges
        Eigen::Tensor<cx64, 3> Ledge(1, 1, num_paulis); // The left  edge
        Eigen::Tensor<cx64, 3> Redge(1, 1, num_paulis); // The right edge
        Ledge(0, 0, 0) = 1.0 / std::sqrt(num_paulis);
        Ledge(0, 0, 1) = 1.0 / std::sqrt(num_paulis);
        Redge(0, 0, 0) = 1;
        Redge(0, 0, 1) = 1;
        return std::make_tuple(mpos, Ledge, Redge);
    }

    std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        sum_of_pauli_mpo(const std::vector<Eigen::MatrixXcd> &paulimatrices, size_t sites, RandomizerMode mode)
    /*! Builds a string of MPO's
     *      P = Π  O_i
     * where Π is the product over all sites, and O_i are MPOs with 2x2 (pauli) matrices on the diagonal
     *
     *        2
     *        |
     *    0---O---1
     *        |
     *        3
     *
     * If mode == RandomizerMode::SHUFFLE:
     *
     *                 | s0  0   0  .  |
     *      O_i =      | 0   s1  0  .  |
     *                 | 0   0  s2  .  |
     *                 | .   .   . ... |
     *
     * where for each O_i the matrices s0, s1, s2 are shuffled randomly
     *
     * If mode == RandomizerMode::SELECT1:
     *
     *      O_i =  | s  |
     *
     *  where for each O_i one of the matrices s0, s1, s2... is selected randomly
     *
     * If mode == RandomizerMode::ASIS:
     *
     *                 | s0  0   0  .  |
     *      O_i =      | 0   s1  0  .  |
     *                 | 0   0  s2  .  |
     *                 | .   .   . ... |
     *
     * where for each O_i the matrices s0, s1, s2... are placed in order as given
     *
     */

    {
        if(paulimatrices.empty()) throw except::runtime_error("List of pauli matrices is empty");
        long                spin_dim = 2;
        std::array<long, 4> extent4  = {1, 1, spin_dim, spin_dim}; /*!< Extent of pauli matrices in a rank-4 tensor */
        std::array<long, 2> extent2  = {spin_dim, spin_dim};       /*!< Extent of pauli matrices in a rank-2 tensor */
        std::array<long, 4> offset4  = {0, 0, 0, 0};

        std::vector<Eigen::Tensor<cx64, 4>> mpos;
        auto                                pauli_idx = num::range<size_t>(0, paulimatrices.size(), 1);

        for(size_t site = 0; site < sites; site++) {
            Eigen::Tensor<cx64, 4> mpo;
            switch(mode) {
                case RandomizerMode::SELECT1: {
                    mpo.resize(1, 1, spin_dim, spin_dim);
                    mpo.setZero();
                    auto        rnd_idx                          = rnd::uniform_integer_box<size_t>(0, paulimatrices.size() - 1);
                    const auto &pauli                            = paulimatrices[rnd_idx];
                    mpo.slice(offset4, extent4).reshape(extent2) = tenx::TensorCast(pauli);
                    break;
                }
                case RandomizerMode::SHUFFLE: {
                    rnd::shuffle(pauli_idx);
                    [[fallthrough]];
                }
                case RandomizerMode::ASIS: {
                    auto num_paulis = safe_cast<long>(paulimatrices.size());
                    mpo.resize(num_paulis, num_paulis, spin_dim, spin_dim);
                    for(long idx = 0; idx < num_paulis; idx++) {
                        auto        uidx                             = safe_cast<size_t>(idx);
                        const auto &pauli                            = paulimatrices[pauli_idx[uidx]];
                        offset4                                      = {idx, idx, 0, 0};
                        mpo.slice(offset4, extent4).reshape(extent2) = tenx::TensorCast(pauli);
                    }
                    break;
                }
            }
            mpos.emplace_back(mpo);
        }

        // Create compatible edges
        Eigen::Tensor<cx64, 3> Ledge(1, 1, 1); // The left  edge
        Eigen::Tensor<cx64, 3> Redge(1, 1, 1); // The right edge
        switch(mode) {
            case RandomizerMode::SHUFFLE:
            case RandomizerMode::ASIS: {
                Ledge.resize(1, 1, paulimatrices.size());
                Redge.resize(1, 1, paulimatrices.size());
                for(size_t idx = 0; idx < paulimatrices.size(); idx++) {
                    Ledge(0, 0, idx) = 1.0 / std::sqrt(paulimatrices.size());
                    Redge(0, 0, idx) = 1;
                }
                break;
            }
            case RandomizerMode::SELECT1: {
                Ledge(0, 0, 0) = 1;
                Redge(0, 0, 0) = 1;
                break;
            }
        }
        return std::make_tuple(mpos, Ledge, Redge);
    }

    std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        random_pauli_mpos(const std::vector<Eigen::MatrixXcd> &paulimatrices, const std::vector<double> &uniform_dist_widths, size_t sites)
    /*! Builds a set of MPO's used for randomizing a state  pauli matrix MPO's with random weights picked from a uniform distribution
     *      P = Π  O_i
     * where Π is the product over all sites, and O_i is the MPO sum of pauli matrices with random weights.
     *
     *            | c0*s0   0       0     .   |
     * O_i =      | 0       c1*s1   0     .   |
     *            | 0       0       c2*s2 .   |
     *            | .       .       .     ... |
     *  Here s_i are 2x2 pauli matrices (including identity) and
     *  the weight coefficients c_i are random real numbers drawn from a uniform distribution U(-w,w).
     *
     *        2
     *        |
     *    0---O---1
     *        |
     *        3
     *
     */
    {
        if(paulimatrices.empty()) throw except::runtime_error("List of pauli matrices is empty");
        if(paulimatrices.size() != uniform_dist_widths.size()) throw except::runtime_error("List size mismatch: paulimatrices and uniform_dist_widths");
        long                num_paulis = safe_cast<long>(paulimatrices.size());
        long                spin_dim   = 2;
        std::array<long, 4> extent4    = {1, 1, spin_dim, spin_dim}; /*!< Extent of pauli matrices in a rank-4 tensor */
        std::array<long, 2> extent2    = {spin_dim, spin_dim};       /*!< Extent of pauli matrices in a rank-2 tensor */

        std::vector<Eigen::Tensor<cx64, 4>> mpos;
        for(size_t site = 0; site < sites; site++) {
            Eigen::Tensor<cx64, 4> MPO_S(num_paulis, num_paulis, spin_dim, spin_dim);
            MPO_S.setZero();
            for(long idx = 0; idx < num_paulis; idx++) {
                auto        uidx                               = safe_cast<size_t>(idx);
                auto        coeff                              = 1 + rnd::uniform_double_box(uniform_dist_widths[uidx]);
                auto        offset4                            = std::array<long, 4>{idx, idx, 0, 0};
                const auto &pauli                              = paulimatrices[uidx];
                MPO_S.slice(offset4, extent4).reshape(extent2) = tenx::TensorCast(coeff * pauli);
            }
            mpos.emplace_back(MPO_S);
        }

        // Create compatible edges
        Eigen::Tensor<cx64, 3> Ledge(1, 1, num_paulis); // The left  edge
        Eigen::Tensor<cx64, 3> Redge(1, 1, num_paulis); // The right edge
        Ledge(0, 0, 0) = 1.0 / std::sqrt(num_paulis);
        Ledge(0, 0, 1) = 1.0 / std::sqrt(num_paulis);
        Redge(0, 0, 0) = 1;
        Redge(0, 0, 1) = 1;
        return std::make_tuple(mpos, Ledge, Redge);
    }
}
