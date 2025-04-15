#include "../lbit.h"
#include "../spin.h"
#include "config/debug.h"
#include "config/enums.h"
// #include "config/settings.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "general/iter.h"
#include "io/fmt_custom.h"
#include "io/spdlog.h"
#include "math/cast.h"
#include "math/fit.h"
#include "math/float.h"
#include "math/linalg.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/stat.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/common/plot.h"
#include "tools/finite/mps.h"
#include "tools/finite/ops.h"
#include <algorithm>
#include <fmt/ranges.h>
#include <h5pp/h5pp.h>
#include <unordered_set>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

namespace settings {
    inline constexpr bool debug_circuit   = false;
    inline constexpr bool verbose_circuit = false;
}

qm::lbit::UnitaryGateProperties::UnitaryGateProperties(const std::vector<double> &h)
    : sites(settings::model::model_size), depth(settings::model::lbit::u_depth), fmix(settings::model::lbit::u_fmix), lambda(settings::model::lbit::u_lambda),
      wkind(settings::model::lbit::u_wkind), mkind(settings::model::lbit::u_mkind), hmean(settings::model::lbit::J1_mean),
      hwdth(settings::model::lbit::J1_wdth), hdist(settings::model::lbit::distribution), hvals(h) {
    if(wkind == LbitCircuitGateWeightKind::EXPDECAY and hvals.empty())
        throw except::logic_error("No onsite fields h given for LbitCircuitGateWeight::EXPDECAY");
}

std::string qm::lbit::UnitaryGateProperties::string() const {
    return fmt::format("depth {} | fmix {:.3f} | lambda {:.3f} | wkind {} | mkind {} | hvals {::.3e}", depth, fmix, lambda, enum2sv(wkind), enum2sv(mkind),
                       hvals);
}

void qm::lbit::write_unitary_circuit_parameters(h5pp::File &file, std::string_view table_path, const std::vector<UnitaryGateParameters> &circuit) {
    //    if(circuit.empty()) return;
    if(file.linkExists(table_path))
        throw except::logic_error("The random unitary circuit has already been written to file: {} | {}", file.getFilePath(), table_path);
    tools::log->info("Writing unitary circuit to [{}]", table_path);
    file.createTable(UnitaryGateParameters::get_h5_type(), table_path, "Random Unitary Circuit");
    file.writeTableRecords(circuit, table_path);
}

std::vector<qm::lbit::UnitaryGateParameters> qm::lbit::read_unitary_circuit_parameters(const h5pp::File &file, std::string_view table_path) {
    tools::log->info("Loading unitary circuit from table: [{}]", table_path);
    return file.readTableRecords<std::vector<UnitaryGateParameters>>(table_path, h5pp::TableSelection::ALL);
}
std::vector<std::vector<qm::Gate>> qm::lbit::read_unitary_2site_gate_layers(const h5pp::File &file, std::string_view table_path) {
    return get_unitary_2site_gate_layers(read_unitary_circuit_parameters(file, table_path));
}

std::vector<std::vector<qm::Gate>> qm::lbit::get_unitary_2site_gate_layers(const std::vector<UnitaryGateParameters> &circuit) {
    /*! Returns a set of unitary two site operators used to transform between physical and l-bit representations */
    std::vector<std::vector<qm::Gate>> unitary_2site_gate_layers;
    tools::log->trace("Loading twosite unitaries");
    for(auto &p : circuit) {
        if(p.layer >= unitary_2site_gate_layers.size()) unitary_2site_gate_layers.emplace_back(std::vector<qm::Gate>{});
        auto &layer = unitary_2site_gate_layers.at(p.layer);
        layer.emplace_back(get_unitary_2site_gate(p));
    }
    return unitary_2site_gate_layers;
}

template<typename Scalar>
auto get_permuted(const Eigen::Tensor<Scalar, 2> &in, MeanType meanType) {
    auto t_permute = tid::tic_scope("permute");
    // First, subtract the center position of each lbit, so we get L lbits centered around zero.
    // In practice, we make a cyclic permutation of the rows of lbit_support
    // In addition, we mirror the lbit along its vertical, so that we can average its left and right half together
    long rows = in.dimension(0);
    long cols = in.dimension(1);
    auto out  = Eigen::Tensor<Scalar, 2>(rows, cols);
    out.setZero();
    for(long j = 0; j < cols; j++) {
        for(long i = 0; i < rows; i++) {
            // If i < j, this corresponds to the left side of an lbit.
            // Consider i == j to be "origo" for an lbit, so mod(i+j,cols) becomes the index starting from that origo.
            // In addition, we fold the left side of an lbit back onto the right side.
            long j_twin   = j;
            long distance = std::abs(i - j);
            long j_perm   = distance;
            if(j >= i) {
                j_twin = i - distance;
                if(j_twin < 0) j_twin = j;
            } else {
                j_twin = i + distance;
                if(j_twin >= cols) j_twin = j;
            }
            switch(meanType) {
                case MeanType::ARITHMETIC: {
                    out(i, j_perm) = 0.5 * (in(i, j) + in(i, j_twin));
                    break;
                }
                case MeanType::GEOMETRIC: {
                    auto val1      = in(i, j) == 0 ? 0 : std::log(std::abs(in(i, j)));
                    auto val2      = in(i, j_twin) == 0 ? 0 : std::log(std::abs(in(i, j_twin)));
                    out(i, j_perm) = std::exp(0.5 * (val1 + val2));
                    break;
                }
            }
        }
    }
    if constexpr(settings::verbose_circuit) tools::log->debug("get_permuted {}: out:\n{}\n", enum2sv(meanType), linalg::tensor::to_string(out, 16));
    return out;
}

void qm::lbit::UnitaryGateProperties::randomize_hvals() const { hvals = rnd::random(hdist, hmean, hwdth, sites); }

qm::Gate qm::lbit::get_unitary_2site_gate(const UnitaryGateParameters &u) {
    /*! Returns a two site gate used in the random unitary circuit transform between physical and l-bit representations
     *
     *
     @verbatim
                   2      3               1
                   |      |               |
                [ exp(-ifwM) ] ---> [ exp(-ifwM) ]
                   |      |               |
                   0      1               0
     @endverbatim

     Where M is defined as
        θ³ n[i]     n[i+1]      +
        θ² n[i]     (1-n[i+1])  +
        θ¹(1-n[i])  n[i+1]      +
        θ⁰(1-n[i])  (1-n[i+1])  +
        c  σ+σ- +
        c* σ-σ+

     and n[i] = 0.5(σz[i] + 1) acts on site i.
     Alternatively, we could write this in terms of σz operators, as
        θ⁰/4 (1+σz[i])(1+σz[i+1]) +
        θ¹/4 (1+σz[i])(1-σz[i+1]) +
        θ²/4 (1-σz[i])(1+σz[i+1]) +
        θ³/4 (1-σz[i])(1-σz[i+1]) +
        c  σ+σ- +
        c* σ-σ+
    */
    constexpr bool kroneckerSwap = false;
    static auto    SZ            = qm::spin::half::gen_twobody_spins(qm::spin::half::sz, kroneckerSwap); // We use these as matrices
    static auto    SP            = qm::spin::half::gen_twobody_spins(qm::spin::half::sp, kroneckerSwap); // We use these as matrices
    static auto    SM            = qm::spin::half::gen_twobody_spins(qm::spin::half::sm, kroneckerSwap); // We use these as matrices
    static auto    ID            = qm::spin::half::gen_twobody_spins(qm::spin::half::id, kroneckerSwap); // We use these as matrices
    static auto    N             = std::vector<Eigen::Matrix4cd>{0.5 * (ID[0] + SZ[0]), 0.5 * (ID[1] + SZ[1])};
    static auto    spin_dims     = std::vector<long>{2l, 2l};

    auto M = Eigen::Matrix4cd();
    switch(u.mkind) {
            /* clang-format off */
        case LbitCircuitGateMatrixKind::MATRIX_V1: {
            M =   u.theta[0] * 0.25 * (ID[0] + SZ[0] + SZ[1] + u.l * SZ[0] * SZ[1])
                + u.theta[1] * 0.25 * (ID[0] - SZ[0] + SZ[1] - u.l * SZ[0] * SZ[1])
                + u.theta[2] * 0.25 * (ID[0] + SZ[0] - SZ[1] - u.l * SZ[0] * SZ[1])
                + u.theta[3] * 0.25 * (ID[0] - SZ[0] - SZ[1] + u.l * SZ[0] * SZ[1])
                + u.c * SP[0] * SM[1] + std::conj(u.c) * SM[0] * SP[1];
            break;
        }
        case LbitCircuitGateMatrixKind::MATRIX_V2: {
            M =   u.theta[0] * 0.5 * SZ[0]
                + u.theta[1] * 0.5 * SZ[1]
                + u.theta[2] * 0.5 * SZ[0] * SZ[1] * u.l
                + u.c * SP[0] * SM[1] + std::conj(u.c) * SM[0] * SP[1];
            break;
        }
        case LbitCircuitGateMatrixKind::MATRIX_V3:{
            M =   u.theta[0] * 0.5 * SZ[0]
                + u.theta[1] * 0.5 * SZ[1]
                + u.l   * 0.5 * SZ[0] * SZ[1]
                + u.c * SP[0] * SM[1] + std::conj(u.c) * SM[0] * SP[1];
            break;
        }
        /* clang-format on */
        default: throw except::runtime_error("Unrecognized gate matrix kind: {}", enum2sv(u.mkind));
    }

    // Here we shuffle to get the correct underlying index pattern: Sites are contracted left-to right, but
    // the kronecker product that generated two-site gates above has indexed right-to-left
    //         1                   3      2              2      3                1
    //         |                   |      |              |      |                |
    //   [     M     ]  --->    [     M      ]   --->  [    M     ]  --->  [     M     ]
    //         |                   |      |              |      |                |
    //         0                   1      0              0      1                0
    //        Eigen::Tensor<cx64, 2> M_shuffled = tenx::TensorMap(M, 2, 2, 2, 2).shuffle(tenx::array4{1, 0, 3, 2}).reshape(tenx::array2{4, 4});
    //        Eigen::MatrixXcd       expifwM     = (-1.0i * u.f * u.w * tenx::MatrixMap(M_shuffled)).exp();
    Eigen::MatrixXcd       expifwM_unshuffled = (-1.0i * u.f * u.w * M).exp();
    Eigen::Tensor<cx64, 2> expifwM            = tenx::TensorMap(expifwM_unshuffled, 2, 2, 2, 2).shuffle(tenx::array4{1, 0, 3, 2}).reshape(tenx::array2{4, 4});
    if constexpr(settings::debug) {
        // Sanity check
        if(not tenx::MatrixMap(expifwM).isUnitary()) throw except::logic_error("expifM is not unitary!");
        //        if(not expifwM.isUnitary()) throw except::logic_error("expifM is not unitary!");
    }
    //    tools::log->info("exp(-ifwM): {}", linalg::matrix::to_string(expifwM_unshuffled, 8));
    return {expifwM, std::vector<size_t>{u.sites.front(), u.sites.back()}, spin_dims};
    //    return {tenx::TensorMap(expifwM), std::vector<size_t>{u.sites.front(), u.sites.back()}, spin_dims};
}

std::vector<qm::Gate> qm::lbit::create_unitary_2site_gate_layer(const qm::lbit::UnitaryGateProperties &u) {
    /*! Returns a set of unitary two site operators used to transform between physical and l-bit representations */
    tools::log->trace("Generating twosite unitaries");
    if(u.wkind == LbitCircuitGateWeightKind::EXPDECAY) {
        if(u.hvals.empty() or u.hvals.size() != u.sites) throw except::logic_error("uprop.hvals.size() {} != sites {}", u.hvals.size(), u.sites);
    }
    std::vector<qm::Gate> gates;
    gates.reserve(u.sites - 1);
    for(size_t idx = 0; idx < u.sites - 1; idx++) {
        // This 2-site gate connects sites idx and idx+1
        UnitaryGateParameters p;
        p.layer = u.layer_count;
        p.sites = {idx, idx + 1};
        p.f     = u.fmix;
        p.l     = u.lambda;
        p.w     = u.wkind == LbitCircuitGateWeightKind::EXPDECAY ? std::exp(-2.0 * std::abs(u.hvals[idx] - u.hvals[idx + 1])) : 1.0;
        p.theta = {rnd::normal(0.0, 1.0), rnd::normal(0.0, 1.0), rnd::normal(0.0, 1.0), rnd::normal(0.0, 1.0)};
        switch(u.mkind) {
            case LbitCircuitGateMatrixKind::MATRIX_V1: break;
            case LbitCircuitGateMatrixKind::MATRIX_V2: p.theta[3] = 0.0; break;
            case LbitCircuitGateMatrixKind::MATRIX_V3:
                p.theta[2] = 0.0;
                p.theta[3] = 0.0;
                break;
        }
        p.c     = std::complex<double>(rnd::normal(0.0, 1.0),  //
                                       rnd::normal(0.0, 1.0)); // complex normal random variable
        p.mkind = u.mkind;
        p.wkind = u.wkind;
        gates.emplace_back(get_unitary_2site_gate(p));

        // Save for storage in a circuit table on file
        if(u.keep_circuit) { u.circuit.emplace_back(p); }
    }
    if constexpr(settings::debug) {
        // Sanity check
        for(const auto &g : gates)
            if(not tenx::MatrixMap(g.op).isUnitary()) throw except::logic_error("u is not unitary!");
    }
    u.layer_count += 1;
    return gates;
}

/*! \brief Make MPO's out of a layer of unitary 2-site gates.
 *
 * Step 1: Reshape the first unitary gate, introducing a dummy 0 index.
 *         The numbers 01 denotes the site indices in the gate.
 * @verbatim
 *          2     3            3     4          2     4
 *          |     |            |     |          |     |
 *         [ g(01) ]   =   0--[ g(01) ]  =  0--[ g(01) ] (shuffle: 0, 1, 3, 2, 4)
 *          |     |            |     |          |     |
 *          0     1            1     2          1     3
 * @endverbatim
 *
 * Step 2: Make a matrix merging indices (012,34) and split the gate using SVD into an MPO and a gate:
 * @verbatim
 *            2     4                                   2                2
 *            |     |   SVD                             |                |
 *        0--[ g(01) ]   =  US^0.5 S^0.5 V^T  =  0--[ mpo(0) ]--1   0--[g(1)]
 *            |     |                                   |                |
 *            1     3                                   3                1
 * @endverbatim
 *
 * Step 3: Connect g(1) to the next gate. Connect from under if g is odd, or from above if g is even.
 *         Remember that we apply a unitary operators from below onto a ket |psi>.
 *         A sequence looks as follows:
 * @verbatim
 *   ---A0----A1----A2----A3--       ...
 *      |     |     |     |
 *      [g(01)]     [g(23)]
 *      |     |     |     |
 *            [g(12)]
 *            |     |
 * @endverbatim
 *
 * @verbatim
 *
 *                2
 *                |
 *          0--[g(1)]
 *                |                    1     4               2     4
 *                1                    |     |               |     |
 *                2     3      =   0--[ g(12) ]     =    0--[ g(12) ]  (shuffle: 02134)
 *                |     |              |     |               |     |
 *               [ g(12) ]             2     3               1     3
 *                |     |
 *                0     1
 *
 * or on odd sites
 *
 *                2     3
 *                |     |
 *               [ g(23) ]             3     4             2     4
 *                |     |              |     |             |     |
 *                0     1      =   0--[ g(23) ]   =    0--[ g(23) ]    (shuffle: 01324)
 *                2                    |     |             |     |
 *                |                    1     2             1     3
 *          0--[g(2)]
 *                |
 *                1
 * @endverbatim
 *
 * Repeat from step 2, replacing g(01) with g(12), and so on, until there are no more gates.
 *
 * Step 4: Add a dummy index with dimension 1 to the remaining right-most gate, to convert it to an mpo.
 */
std::vector<Eigen::Tensor<cx64, 4>> qm::lbit::get_unitary_mpo_layer(const std::vector<qm::Gate> &ulayer, std::optional<svd::config> cfg) {
    if(ulayer.empty()) return {};
    if(ulayer.size() <= 1) throw except::logic_error("Can't make MPO's with less than two gates");
    auto t_mpolayer = tid::tic_scope("mpo-layer");
    auto mpos       = std::vector<Eigen::Tensor<cx64, 4>>(ulayer.size() + 1); // L-1 gates in the unitary circuit
    if(not cfg.has_value()) cfg = svd::config();
    if(not cfg->svd_lib) cfg->svd_lib = svd::lib::lapacke;
    if(not cfg->svd_rtn) cfg->svd_rtn = svd::rtn::geauto;
    if(not cfg->rank_max) cfg->rank_max = settings::flbit::cls::mpo_circuit_svd_bondlim;
    if(not cfg->truncation_limit) cfg->truncation_limit = settings::flbit::cls::mpo_circuit_svd_trnclim;
    if(not cfg->switchsize_gesdd) cfg->switchsize_gesdd = settings::precision::svd_switchsize_bdc;
    auto svd = svd::solver(cfg);

    // Start Step 1 with the first gate
    const auto &ugate0 = ulayer.front();                                                  // The left-most unitary gate
    auto        dims4  = ugate0.shape<4>();                                               // Original shape of the gate
    auto        dims5  = std::array<long, 5>{1, dims4[0], dims4[1], dims4[2], dims4[3]};  // Shape including the dummy index
    auto        shuf5  = std::array<long, 5>{0, 1, 3, 2, 4};                              // Shuffling to prepare for SVD
    auto        gate5  = Eigen::Tensor<cx64, 5>(ugate0.op.reshape(dims5).shuffle(shuf5)); // The gate including dummy index
    auto        gate3  = Eigen::Tensor<cx64, 3>();                                        // Temporary used in step 2

    // There are L-1 gates on a single layer of the unitary circuit
    for(size_t gidx = 0; gidx < ulayer.size() /* == L-1 */; ++gidx) {
        // Step 2
        std::tie(mpos[gidx], gate3) = svd.split_mpo_gate(gate5, cfg.value());
        if constexpr(settings::verbose_circuit) tools::log->debug("split ugate[idx:{}]", gidx);
        if(gidx + 1 >= ulayer.size()) break; // We can't add more gates. Now gate 3 has the mpo corresponding to site L
        // Step 3
        const auto &ugate = ulayer[gidx + 1];
        dims4             = ugate.shape<4>(); // Original shape of the gate

        if constexpr(settings::debug_circuit) {
            if(not ugate.isUnitary()) throw except::runtime_error("get_unitary_mpo_layer: ugate[pos:{}] is not unitary", ugate.pos);
        }

        if((gidx % 2) == 0 /* even */) {
            // Contract from above
            if constexpr(settings::verbose_circuit) tools::log->debug("contr gate3[idx:{}] onto ugate[idx:{} pos:{}] from above", gidx, gidx + 1, ugate.pos);
            gate5 = gate3.contract(ugate.op.reshape(dims4), tenx::idx({1}, {2})).shuffle(std::array<long, 5>{0, 2, 1, 3, 4});
        } else {
            if constexpr(settings::verbose_circuit) tools::log->debug("contr gate3[idx:{}] onto ugate[idx:{} pos:{}] from below", gidx, gidx + 1, ugate.pos);
            gate5 = gate3.contract(ugate.op.reshape(dims4), tenx::idx({2}, {0})).shuffle(std::array<long, 5>{0, 1, 3, 2, 4});
        }
    }
    if(gate3.size() != 0) {
        /* Step 4:
                2             3
                |             |
          0--[g(2)] =   0--[g(2)]-1
                |             |
                1             2
        */
        mpos.back() = gate3.reshape(std::array<long, 4>{gate3.dimension(0), 1, gate3.dimension(1), gate3.dimension(2)}).shuffle(tenx::array4{0, 1, 3, 2});
    }
    return mpos;
}

std::vector<Eigen::Tensor<cx64, 4>> qm::lbit::get_unitary_mpo_layer(const UnitaryGateProperties &u) {
    return get_unitary_mpo_layer(create_unitary_2site_gate_layer(u));
}

/*! \brief Merge multiple MPO layers into a single one using SVD.
 */
std::vector<Eigen::Tensor<cx64, 4>> qm::lbit::merge_unitary_mpo_layers(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers) {
    auto t_mpomerge = tid::tic_scope("mpo-merge");
    auto mpo_merged = std::vector<Eigen::Tensor<cx64, 4>>(mpo_layers.front());
    for(size_t idx = 1; idx < mpo_layers.size(); ++idx) { mpo_merged = merge_unitary_mpo_layers(mpo_merged, mpo_layers[idx]); }
    return mpo_merged;
}

// Eigen::Tensor<cx64, 2> qm::lbit::get_unitary_layer_as_tensor_old(const std::vector<qm::Gate> &unitary_layer) {
//     auto cfg                 = svd::config();
//     cfg.svd_lib              = svd::lib::lapacke;
//     cfg.svd_rtn              = svd::rtn::geauto;
//     cfg.rank_max             = 256;
//     cfg.truncation_limit     = 1e-14;
//     cfg.switchsize_gesdd     = settings::precision::svd_switchsize_bdc;
//     auto           mpo_layer = Eigen::Tensor<cx64, 4>();
//     constexpr auto con2      = tenx::idx({1}, {0});
//     constexpr auto shf6      = tenx::array6{0, 3, 1, 4, 2, 5};
//     for(const auto &mpo : get_unitary_mpo_layer(unitary_layer, cfg)) {
//         if(mpo_layer.size() == 0) {
//             mpo_layer = mpo;
//             continue;
//         }
//         auto                   dimL = mpo_layer.dimensions();
//         auto                   dimR = mpo.dimensions();
//         auto                   shp4 = tenx::array4{dimL[0], dimR[1], dimL[2] * dimR[2], dimL[3] * dimR[3]};
//         Eigen::Tensor<cx64, 4> tmp  = mpo_layer.contract(mpo, con2).shuffle(shf6).reshape(shp4);
//         mpo_layer                   = tmp;
//     }
//     auto shf4         = tenx::array4{0, 3, 1, 2}; // cast to matrix assuming that the virtual bonds have dimension 1
//     auto shp2         = tenx::array2{mpo_layer.dimension(0) * mpo_layer.dimension(3), mpo_layer.dimension(2) * mpo_layer.dimension(1)};
//     auto tensor_layer = Eigen::Tensor<cx64, 2>(mpo_layer.shuffle(shf4).reshape(shp2));
//     if constexpr(settings::debug_circuit)
//         if(not tenx::MatrixMap(tensor_layer).isUnitary(1e-12)) throw except::logic_error("get_unitary_layer_as_tensor: tensor_layer is not unitary");
//     return tensor_layer;
// }

Eigen::Tensor<cx64, 2> qm::lbit::get_unitary_layer_as_tensor(const std::vector<qm::Gate> &unitary_layer) {
    /* We apply gates from top to bottom
     *   ---A0----A1----A2----A3--       ...
     *      |     |     |     |
     *      [g(01)]     [g(23)]
     *      |     |     |     |
     *            [g(12)]
     *            |     |
     * where
     *    2     3
     *    |     |
     *    [g(12)]
     *    |     |
     *    0     1
     *
     */
    tools::log->trace("contracting unitary layer as tensor");
    qm::Gate fullgate;
    double   rss_start = 0;
    double   hwm_start = 0;
    auto     printlog  = [&](const std::vector<size_t> &pos, std::string_view tag) -> void {
        if constexpr(settings::debug_circuit) {
            double hwm = debug::mem_hwm_in_mb();
            double rss = debug::mem_rss_in_mb();
            tools::log->info("contracted gate {} {} | rss {:<.1f} (+{:<.1f}) hwm {:<.1f} mb (+{:<.1f})", pos, tag, rss, rss - rss_start, hwm, hwm - hwm_start);
        }
    };

    for(const auto &[i, g] : iter::enumerate(unitary_layer)) {
        if constexpr(settings::debug_circuit) {
            rss_start = debug::mem_rss_in_mb();
            hwm_start = debug::mem_hwm_in_mb();
        }
        if(i == 0) {
            // First gate
            fullgate = g;
            printlog(g.pos, "first");
            continue;
        }
        if(g.pos.front() % 2 == 0) {
            // Even gate 01, 23, 45 ... fullgate connects to g from below
            fullgate = fullgate.connect_below(g);
            printlog(g.pos, "from below");
        } else {
            // Odd gate 12, 34, 56 ... fullgate connects to g from above
            fullgate = fullgate.connect_above(g);
            printlog(g.pos, "from above");
        }
    }

    if constexpr(settings::debug_circuit)
        if(not tenx::MatrixMap(fullgate.op).isUnitary(1e-12)) throw except::logic_error("get_unitary_layer_as_tensor: tensor_layer is not unitary");
    return fullgate.op;
}

Eigen::Tensor<cx64, 2> qm::lbit::get_unitary_circuit_as_tensor(const std::vector<std::vector<qm::Gate>> &unitary_circuit) {
    Eigen::Tensor<cx64, 2> all_layers;
    double                 rss_start = 0;
    double                 hwm_start = 0;
    auto                   printlog  = [&](size_t lidx, double t) -> void {
        //        if constexpr(settings::debug_circuit) {
        double hwm = debug::mem_hwm_in_mb();
        double rss = debug::mem_rss_in_mb();
        tools::log->info("contracted layer {} dims {} ... {:.3e} s | rss {:<.1f} (+{:<.1f}) hwm {:<.1f} mb (+{:<.1f})", lidx, all_layers.dimensions(), t, rss,
                                            rss - rss_start, hwm, hwm - hwm_start);
        //        }
    };

    for(const auto &[lidx, layer] : iter::enumerate(unitary_circuit)) {
        //        if constexpr(settings::debug_circuit) {
        rss_start = debug::mem_rss_in_mb();
        hwm_start = debug::mem_hwm_in_mb();
        //        }
        auto t_con = tid::tic_token("contract-u");

        if(lidx == 0) {
            all_layers = get_unitary_layer_as_tensor(layer);
            printlog(lidx, t_con->get_last_interval());
            continue;
        }
        // New layers appended from below: we apply a circuit top to bottom, with the MPS at the top, physical index pointing down)
        // Alternatively, right to left, so ... U3 U2 U1 U0 |psi>
        if constexpr(settings::debug_circuit) tools::log->debug("contracting layer {} | dims all_layers {} ...", lidx, all_layers.dimensions());
        // This is about 2x faster because it calls BLAS/MKL GEMM. The tensor contraction version does not!
        all_layers = tenx::gemm(get_unitary_layer_as_tensor(layer), all_layers);
        printlog(lidx, t_con->get_last_interval());
    }
    if constexpr(settings::debug_circuit)
        if(not tenx::MatrixMap(all_layers).isUnitary(1e-12)) throw except::logic_error("get_unitary_circuit_as_tensor: all_layers is not unitary");
    return all_layers;
}

Eigen::Tensor<cx64, 2> qm::lbit::get_time_evolution_operator(cx128 delta_t, const Eigen::Tensor<cx64, 2> &H) {
    // Given a matrix H, this returns exp(delta_t * H)
    // For time evolution, just make sure delta_t = -i*d,  where d is a (small) real positive number.
    auto dt = cx64(static_cast<fp64>(delta_t.real()), static_cast<fp64>(delta_t.imag()));
    // TODO: Check if delta_t should be multiplied by -1.0i
    tools::log->warn("qm::lbit::get_time_evolution_operator: Should delta_t be multiplied by -1.0i here?");
    return tenx::TensorCast((dt * tenx::MatrixMap(H)).exp());
}

std::vector<qm::Gate> qm::lbit::get_time_evolution_gates(cx128 delta_t, const std::vector<qm::Gate> &hams_nsite) {
    std::vector<Gate> time_evolution_gates;
    time_evolution_gates.reserve(hams_nsite.size());
    for(auto &h : hams_nsite) {
        time_evolution_gates.emplace_back(h.exp(cx128(-1.0i) * delta_t)); // exp(-i * delta_t * h)
    }
    if constexpr(settings::debug) {
        for(auto &t : time_evolution_gates)
            if(not t.isUnitary(Eigen::NumTraits<double>::dummy_precision() * static_cast<double>(t.op_t.dimension(0)))) {
                throw except::runtime_error("Time evolution operator at pos {} is not unitary:\n{}", t.pos, linalg::tensor::to_string(t.op));
            }
    }
    return time_evolution_gates;
}

std::vector<qm::SwapGate> qm::lbit::get_time_evolution_swap_gates(cx128 delta_t, const std::vector<qm::SwapGate> &hams_nsite) {
    std::vector<SwapGate> time_evolution_swap_gates;
    time_evolution_swap_gates.reserve(hams_nsite.size());
    for(const auto &h : hams_nsite) {
        time_evolution_swap_gates.emplace_back(h.exp(cx128(-1.0i) * delta_t)); // exp(-i * delta_t * h)
        time_evolution_swap_gates.back().generate_swap_sequences();
    }
    if constexpr(settings::debug) {
        for(auto &t : time_evolution_swap_gates)
            if(not t.isUnitary(Eigen::NumTraits<double>::dummy_precision() * static_cast<double>(t.op.dimension(0)))) {
                throw except::runtime_error("Time evolution operator at pos {} is not unitary:\n{}", t.pos, linalg::tensor::to_string(t.op));
            }
    }
    return time_evolution_swap_gates;
}

std::vector<Eigen::Tensor<cx64, 2>> qm::lbit::get_time_evolution_operators_2site(size_t sites, cx128 delta_t,
                                                                                 const std::vector<Eigen::Tensor<cx64, 2>> &hams_2site) {
    // In l-bit systems we are aldready in a diagonal basis, so h_{j,j+1} and h_{j+1,j+2} commute. Therefore, we can immediately use the relation
    //      exp(-i*dt *[h_{j,j+1} + h_{j+1,j+2} + ... + h_{L-2, L-1}]) =  exp(-i*dt [h_{j,j+1}]) * exp(-i*dt*[h_{j+1,j+2}]) * ... * exp(-i*dt*[h_{L-2, L-1}])
    // without passing through the Suzuki-Trotter decomposition.
    // Here we expect "hams_2site" to contain terms like  h_{j,j+1} + h_{j+1,j+2} + ... + h_{L-2, L-1}.

    if(hams_2site.size() != sites - 1) throw except::logic_error("Wrong number of twosite hamiltonians: {}. Expected {}", hams_2site.size(), sites - 1);

    std::vector<Eigen::Tensor<cx64, 2>> time_evolution_operators;
    time_evolution_operators.reserve(sites - 1);
    for(const auto &h : hams_2site) time_evolution_operators.emplace_back(get_time_evolution_operator(delta_t, h));
    return time_evolution_operators;
}

std::vector<Eigen::Tensor<cx64, 2>> qm::lbit::get_time_evolution_operators_3site(size_t sites, cx128 delta_t,
                                                                                 const std::vector<Eigen::Tensor<cx64, 2>> &hams_3site) {
    // In l-bit systems we are aldready in a diagonal basis, so h_{i,j,k} and h_{l,m,n} commute. Therefore, we can immediately use the relation
    // exp(A + B) = exp(A)exp(B)
    // without passing through the Suzuki-Trotter decomposition.

    if(hams_3site.size() != sites - 2) throw except::logic_error("Wrong number of three-site hamiltonians: {}. Expected {}", hams_3site.size(), sites - 2);

    std::vector<Eigen::Tensor<cx64, 2>> time_evolution_operators;
    time_evolution_operators.reserve(sites - 1);
    for(const auto &h : hams_3site) time_evolution_operators.emplace_back(get_time_evolution_operator(delta_t, h));
    return time_evolution_operators;
}

cx64 qm::lbit::get_lbit_2point_correlator2(const std::vector<std::vector<qm::Gate>> &unitary_circuit, const Eigen::Matrix2cd &szi, size_t pos_szi,
                                           const Eigen::Matrix2cd &szj, size_t pos_szj, long len) {
    /*! \brief Calculates the operator overlap = Tr(τ^z_i σ^z_j) / 2^L
        Where
            τ^z_i = U σ^z_i  U†,
        and U is expressed as a unitary circuit transformation.
        See more about this here: https://link.aps.org/doi/10.1103/PhysRevB.91.085425
    */
    // Generate gates for the operators
    if constexpr(settings::verbose_circuit) tools::log->trace("Computing Tr (τ_{} σ_{}) / 2^L", pos_szi, pos_szj, pos_szj);

    auto result          = cx64(0, 0);
    auto szi_gate        = qm::Gate(szi, {pos_szi}, {2l}); //
    auto szj_gate        = qm::Gate(szj, {pos_szj}, {2l});
    auto intersection    = qm::get_lightcone_intersection(unitary_circuit, pos_szi, pos_szj);
    auto unitary_slayers = qm::get_lightcone_gate_selection(unitary_circuit, intersection); // Selected gates in each layer
    auto is_disconnected = std::any_of(intersection.begin(), intersection.end(), [](auto &layer) { return layer.empty(); });
    if(is_disconnected) {
        if constexpr(settings::verbose_circuit) tools::log->trace("σzi:{} and σzj:{} are disconnected -> result = {:.6f}", pos_szi, pos_szj, result);
        return result;
    }
    // Setup debug printing of unitary circuits
    std::deque<std::string> net;                                        // Great for debugging
    std::deque<std::string> log;                                        // Great for debugging
    size_t                  uw        = 7;                              // Width of a unitary 2-site gate box
    size_t                  op        = 3;                              // Overlap of a unitary 2-site gate box
    size_t                  tw        = 6;                              // Tag width
    size_t                  mp        = static_cast<size_t>(len) - 1ul; // Max pos
    auto                    empty_str = fmt::format("{0:^{1}}", " ", tw + mp * (uw - op) + op);

    // Start contracting selected unitary gates bottom to top.
    auto g = szi_gate; // The gate that accumulates everything. Starts at σzi position
    if constexpr(settings::verbose_circuit) {
        std::string layer_str = empty_str;
        szi_gate.draw_pos(layer_str, "szi  :");
        net.push_front(layer_str);
        log.push_front(fmt::format("insert σzi{}", szi_gate.pos));
    }

    for(auto &&[idx_slayer, slayer] : iter::enumerate(unitary_slayers)) {
        std::string layer_str = empty_str;
        std::string story_str;
        for(auto &sgate : slayer) {
            if(g.has_pos(sgate.pos)) {
                g = g.insert(sgate);

                if constexpr(settings::verbose_circuit) {
                    sgate.draw_pos(layer_str, fmt::format("u[{:^2}]:", idx_slayer));
                    story_str.append(fmt::format("insert u{} ", sgate.pos));
                }
                auto pos_out = sgate.pos_difference(intersection.at(idx_slayer + 1));
                if(not pos_out.empty()) {
                    // Trace positions outside the light cone intersection
                    g    = g.trace_pos(pos_out);
                    g.op = g.op / g.op.constant(std::pow(2.0, pos_out.size())); // Normalize by dividing the trace of each 2x2 identity.
                    if constexpr(settings::verbose_circuit) story_str.append(fmt::format("trace{} ", pos_out));
                }
            }
        }
        if constexpr(settings::verbose_circuit) {
            story_str.append(fmt::format("now{}", g.pos));
            if(layer_str != empty_str or not story_str.empty()) {
                log.emplace_back(story_str);
                net.emplace_back(layer_str);
            }
        }
    }

    if(g.has_pos(szj_gate.pos)) {
        // Connect σ^z_j at the top
        log.push_back(fmt::format("insert σzj{} ", szj_gate.pos));
        g                     = g.connect_below(szj_gate);
        std::string layer_str = empty_str;
        szj_gate.draw_pos(layer_str, "szj  :");
        net.push_back(layer_str);
    }

    // In the last step we trace everything down to a cx64
    result = g.trace();
    result /= std::pow(2, g.pos.size()); // Normalize by dividing the trace of each 2x2 identity.
    if constexpr(settings::verbose_circuit) {
        log.back().append(fmt::format("result = {:.6f}", result));
        for(const auto &[idx, layer] : iter::enumerate_reverse(net)) tools::log->debug("{} | {}", layer, log[idx]);
    }
    return result;
}

cx64 qm::lbit::get_lbit_2point_correlator3(const std::vector<std::vector<qm::Gate>> &unitary_circuit, const Eigen::Matrix2cd &szi, size_t pos_szi,
                                           const Eigen::Matrix2cd &szj, size_t pos_szj, long len) {
    /*! \brief Calculates the operator overlap O(i,j) = Tr(𝜌'_i σ^z_j) / Tr(𝜌'_i) = Tr(τ^z_i σ^z_j) / 2^L

        Consider first the density matrix

             ρ_i = 1/2^L  (σ^z_i + 1 ) ⊗ I_i',

        where I_i' is the identity matrix on sites j != i.
        Then ρ_i represents a state |psi><psi| where site i is at level 0 with probability 1,
        and the others sites are in a cat-state of level 0 or 1. E.g. ρ could be a state with
        magnetization 1 at site i, and 0 elsewhere, or ρ could be a state with 1 particle at
        site i, and cat state of 0 and 1 particles elsewhere.

        Applying the unitary transformation does not change the trace, since it is just a
        change of basis. We get

             ρ'_i = U ρ_i U†
                  = 1/2^L (U σ^z_i U† + 1)
                  = 1/2^L (τ^z_i + 1) ,

        where the l-bit τ^z_i = U σ^z_i U† acts non-trivially on all sites. Carrying out the
        trace gives us

            Tr(ρ'_i σ^z_j) / Tr(ρ'_i) = 1/2^L Tr(τ^z_i σ^z_j).

        where we used Tr(1) = 2^L, Tr(ρ_i) = Tr(ρ'_i) = 1 and Tr(τ^z_i) = 0.

        We expect an l-bit fully localized at site i to give O(i,i) = 1,  and a fully
        delocalized l-bit to give O(i,j) = 0.5.

        Note that when the light-cone from site i can't reach site j in the unitary circuit, then
        τ^z_i has no support where σ^z_j connects.  Therefore, we get

               Tr(τ^z_i σ^z_j) = Tr(τ^z_i ⊗ σ^z_j) = Tr(τ^z_i) Tr(σ^z_j) = 0

        Read more here:
        https://link.aps.org/doi/10.1103/PhysRevB.91.085425
        https://onlinelibrary.wiley.com/doi/10.1002/andp.201600322
    */

    // Generate gates for the operators
    if constexpr(settings::verbose_circuit) tools::log->trace("Computing Tr (τ_{} σ_{}) / Tr(σ_{})", pos_szi, pos_szj, pos_szj);
    auto t_olap = tid::ur();
    t_olap.tic();

    auto result          = cx64(0, 0);
    auto szi_gate        = qm::Gate(szi, {pos_szi}, {2l}); //
    auto szj_gate        = qm::Gate(szj, {pos_szj}, {2l});
    auto intersection    = qm::get_lightcone_intersection(unitary_circuit, pos_szi, pos_szj);
    auto unitary_slayers = qm::get_lightcone_gate_selection(unitary_circuit, intersection, false); // Selected gates in each layer
    auto is_disconnected = std::any_of(intersection.begin(), intersection.end(), [](auto &layer) { return layer.empty(); });
    if(is_disconnected) {
        if constexpr(settings::verbose_circuit) tools::log->trace("σzi:{} and σzj:{} are disconnected -> result = {:.6f}", pos_szi, pos_szj, result);
        return result;
    }
    // Setup debug printing of unitary circuits
    size_t                  uw        = 7;                              // Width of a unitary 2-site gate box
    size_t                  op        = 3;                              // Overlap of a unitary 2-site gate box
    size_t                  tw        = 6;                              // Tag width
    size_t                  mp        = static_cast<size_t>(len) - 1ul; // Max pos
    auto                    empty_str = fmt::format("{0:^{1}}", " ", tw + mp * (uw - op) + op);
    std::deque<std::string> net(2 + unitary_slayers.size(), empty_str); // Great for debugging
    std::deque<std::string> log(2 + unitary_slayers.size());            // Great for debugging

    auto all_empty = [](const auto &slayers) { return std::all_of(slayers.begin(), slayers.end(), [](const auto &slayer) -> bool { return slayer.empty(); }); };
    auto num_gates_L = [](const auto &slayers, size_t idx = 0) {
        size_t num = 0;
        for(size_t i = idx + 1; i < slayers.size(); i++) {
            if(slayers.at(i - 1).empty()) return num;
            if(slayers.at(i).empty()) return num;
            size_t posLL = slayers.at(i).front().pos.front();
            size_t posL  = slayers.at(i - 1).front().pos.front();
            if(posLL + 1 == posL)
                num++;
            else
                break;
        }
        return num;
    };
    auto num_gates_R = [](const auto &slayers, size_t idx = 0) {
        size_t num = 0;
        for(size_t i = idx + 1; i < slayers.size(); i++) {
            if(slayers.at(i - 1).empty()) return num;
            if(slayers.at(i).empty()) return num;
            size_t posRR = slayers.at(i).back().pos.back();
            size_t posR  = slayers.at(i - 1).back().pos.back();
            if(posR + 1 == posRR)
                num++;
            else
                break;
        }
        return num;
    };
    auto slayer_width = [&]() -> size_t {
        size_t width = 0;
        for(const auto &s : unitary_slayers) {
            size_t w = 0;
            for(const auto &l : s) w += l.pos.size();
            width = std::max(width, w);
        }
        return width;
    };
    auto slayer_diagl = [&]() -> size_t {
        size_t lb = -1ul; // left bottom position
        size_t rb = -1ul; // right bottom position
        size_t lt = -1ul; // left  top position
        size_t rt = -1ul; // right top position
        for(const auto &us : unitary_slayers) {
            if(not us.empty()) {
                if(lb == -1ul) lb = us.front().pos.front();
                if(rb == -1ul) rb = us.front().pos.back();
                lb = std::min(lb, us.front().pos.front());
            }
        }
        for(const auto &us : iter::reverse(unitary_slayers)) {
            if(not us.empty()) {
                if(rt == -1ul) rt = us.front().pos.back();
                if(lt == -1ul) lt = us.front().pos.front();
                rt = std::max(rt, us.back().pos.back());
            }
        }
        return std::max(rb - lb, rt - lt) + 1;
    };

    auto slayer_diagr = [&]() -> size_t {
        size_t lb = -1ul; // left bottom position
        size_t rb = -1ul; // right bottom position
        size_t lt = -1ul; // left  top position
        size_t rt = -1ul; // right top position
        for(const auto &us : unitary_slayers) {
            if(not us.empty()) {
                if(lb == -1ul) lb = us.back().pos.front();
                if(rb == -1ul) rb = us.back().pos.back();
                rb = std::max(rb, us.back().pos.back());
            }
        }
        for(const auto &us : iter::reverse(unitary_slayers)) {
            if(not us.empty()) {
                if(rt == -1ul) rt = us.back().pos.back();
                if(lt == -1ul) lt = us.back().pos.front();
                lt = std::min(lt, us.front().pos.front());
            }
        }
        return std::max(rb - lb, rt - lt) + 1;
    };

    auto width = slayer_width();
    auto diagr = slayer_diagr();
    auto diagl = slayer_diagl();

    bool go_diagonal = std::min(diagl, diagr) < width; // Avoid long thin diagonals.

    // Start contracting selected unitary gates bottom to top.
    auto g = szi_gate; // The gate that accumulates everything. Starts at σzi position
    if constexpr(settings::verbose_circuit) {
        szi_gate.draw_pos(net.front(), "szi  :");
        log.front().append(fmt::format("insert σzi{}", szi_gate.pos));
    }
    if(go_diagonal) {
        // Decide to take the left or right diagonal (whatever is cheapest)
        bool go_right = diagr <= diagl;
        while(not all_empty(unitary_slayers)) {
            for(auto &&[idx_slayer, slayer] : iter::enumerate(unitary_slayers)) {
                if(slayer.empty()) continue; // Already applied the whole slayer
                auto &layer_str = net.at(idx_slayer + 1);
                auto &story_str = log.at(idx_slayer + 1);
                auto  numL      = num_gates_L(unitary_slayers, idx_slayer); // number of gates remaining to take going left
                auto  numR      = num_gates_R(unitary_slayers, idx_slayer); // number of gates remaining to take going right

                std::vector<size_t> pos_out;
                if(go_right) {
                    //                    tools::log->trace("-> insert u[{}]:{}", idx_slayer, slayer.back().pos);
                    if constexpr(settings::verbose_circuit) {
                        slayer.back().draw_pos(layer_str, fmt::format("u[{:^2}]:", idx_slayer));
                        story_str.append(fmt::format("insert u{} ", slayer.back().pos));
                    }
                    g       = g.insert(slayer.back());
                    pos_out = slayer.back().pos_difference(intersection.at(idx_slayer + 1));
                    slayer.pop_back();
                } else {
                    //                    tools::log->trace("<- insert u[{}]:{}", idx_slayer, slayer.front().pos);
                    if constexpr(settings::verbose_circuit) {
                        slayer.front().draw_pos(layer_str, fmt::format("u[{:^2}]:", idx_slayer));
                        story_str.append(fmt::format("insert u{} ", slayer.front().pos));
                    }
                    g       = g.insert(slayer.front());
                    pos_out = slayer.front().pos_difference(intersection.at(idx_slayer + 1));
                    slayer.pop_front();
                }

                // Collect positions that could be traced
                if(not pos_out.empty()) {                                                      // Trace positions outside the light cone intersection
                    pos_out.erase(std::unique(pos_out.begin(), pos_out.end()), pos_out.end()); // Keep unique elements
                    //                    tools::log->trace("trace[{}]:{}", idx_slayer, pos_out);
                    g    = g.trace_pos(pos_out);
                    g.op = g.op / g.op.constant(std::pow(2, pos_out.size())); // Normalize by dividing the trace of each 2x2 identity.
                    if constexpr(settings::verbose_circuit) story_str.append(fmt::format("trace{} ", pos_out));
                }

                if((numR == 0 and go_right) or (numL == 0 and not go_right)) {
                    if constexpr(settings::verbose_circuit) story_str.append(fmt::format("break {} ", g.pos));
                    break; // Start from the bottom of the circuit
                }
            }
        }
    } else {
        for(auto &&[idx_slayer, slayer] : iter::enumerate(unitary_slayers)) {
            auto &layer_str = net.at(idx_slayer + 1);
            auto &story_str = log.at(idx_slayer + 1);
            for(auto &sgate : slayer) {
                if(g.has_pos(sgate.pos)) {
                    g = g.insert(sgate);
                    //                    tools::log->trace("insert u[{}]:{}", idx_slayer, sgate.pos);
                    if constexpr(settings::verbose_circuit) {
                        sgate.draw_pos(layer_str, fmt::format("u[{:^2}]:", idx_slayer));
                        story_str.append(fmt::format("insert u{} ", sgate.pos));
                    }
                    auto pos_out = sgate.pos_difference(intersection.at(idx_slayer + 1));
                    if(not pos_out.empty()) {
                        // Trace positions outside the light cone intersection
                        g    = g.trace_pos(pos_out);
                        g.op = g.op / g.op.constant(std::pow(2, pos_out.size())); // Normalize by dividing the trace of each 2x2 identity.
                        if constexpr(settings::verbose_circuit) story_str.append(fmt::format("trace{} ", pos_out));
                    }
                }
            }
            if constexpr(settings::verbose_circuit) story_str.append(fmt::format("now{} ", g.pos));
        }
    }

    if(g.has_pos(szj_gate.pos)) {
        // Connect σ^z_j at the top
        szj_gate.draw_pos(net.back(), "szj  :");
        log.back().append(fmt::format("insert σzj{} ", szj_gate.pos));
        g = szj_gate.connect_below(g);
    }
    // In the last step we trace everything down to a cx64
    result = g.trace();
    result /= std::pow(2, g.pos.size()); // Normalize by dividing the trace of each 2x2 identity.
    t_olap.toc();
    if constexpr(settings::verbose_circuit) {
        log.back().append(fmt::format("result = {:.6f} | time {:.3e}", result, t_olap.get_time()));
        for(const auto &[idx, layer] : iter::enumerate_reverse(net)) tools::log->debug("{} | {}", layer, log[idx]);
    }
    return result;
}

cx64 qm::lbit::get_lbit_2point_correlator4(const std::vector<Eigen::Tensor<cx64, 4>> &mpo_layer, const Eigen::Matrix2cd &szi, size_t pos_szi,
                                           const Eigen::Matrix2cd &szj, size_t pos_szj) {
    /*! \brief Calculates the operator overlap O(i,j) = Tr(𝜌'_i σ^z_j) / Tr(𝜌'_i) = Tr(τ^z_i σ^z_j) / 2^L

        Consider first the density matrix

             ρ_i = 1/2^L  (σ^z_i + 1 ) ⊗ I_i',

        where I_i' is the identity matrix on sites j != i.
        Then ρ_i represents a state |psi><psi| where site i is at level 0 with probability 1,
        and the others sites are in a cat-state of level 0 or 1. E.g. ρ could be a state with
        magnetization 1 at site i, and 0 elsewhere, or ρ could be a state with 1 particle at
        site i, and cat state of 0 and 1 particles elsewhere.

        Applying the unitary transformation does not change the trace, since it is just a
        change of basis. We get

             ρ'_i = U ρ_i U†
                  = 1/2^L (U σ^z_i U† + 1)
                  = 1/2^L (τ^z_i + 1) ,

        where the l-bit τ^z_i = U σ^z_i U† acts non-trivially on all sites. Carrying out the
        trace gives us

            Tr(ρ'_i σ^z_j) / Tr(ρ'_i) = 1/2^L Tr(τ^z_i σ^z_j).

        where we used Tr(1) = 2^L, Tr(ρ_i) = Tr(ρ'_i) = 1 and Tr(τ^z_i) = 0.

        We expect an l-bit fully localized at site i to give O(i,i) = 1,  and a fully
        delocalized l-bit to give O(i,j) = 0.5.

        Note that when the light-cone from site i can't reach site j in the unitary circuit, then
        τ^z_i has no support where σ^z_j connects.  Therefore, we get

               Tr(τ^z_i σ^z_j) = Tr(τ^z_i ⊗ σ^z_j) = Tr(τ^z_i) Tr(σ^z_j) = 0

        Read more here:
        https://link.aps.org/doi/10.1103/PhysRevB.91.085425
        https://onlinelibrary.wiley.com/doi/10.1002/andp.201600322


        In this implementation we have contracted the unitary circuit into a series of MPOs.
        The top index would connect to a ket: In matrix form 0-U-1 0-|psi>,

        Step 1: Contract the top operator (I or sigma_j) on top of the upper mpo

        @verbatim
                          1
                          |
                        [opj]
                          |                       3
                          0                       |
                          2          =    0--[ mpo_up_opj ]--1
                          |                       |
                   0--[ mpo_up ]--1               2
                          |
                          3

        @endverbatim

        Step 2: Similarly, contract the middle operator (I or sigma_i) on top of the bottom mpo^dagger
        @verbatim
                           1                                      1
                           |                                      |
                         [opi]                                  [opi]
                           |                                      |                            3                                   2
                           0                   shuffle            0                            |              shuffle              |
                  [        2         ]^dagger    =                3             =     0--[ mpo_dn_opi  ]--1     =         0--[ mpo_dn_opi  ]--1
                  [        |         ]                            |                            |                                   |
                  [ 0--[ mpo_dn ]--1 ]                     0--[ mpo_dn*]--1                    2                                   3
                  [        |         ]                            |
                  [        3         ]                            2

        We can save two shuffles by contracting on the opposite spin index immediately instead
                           1
                           |
                         [opi]
                           |                                      2
                           0 ------------                         |
                           2            |               [0--[ mpo_dn_opi  ]--1]^T
                           |            |       =                 |
                    0--[ mpo_dn* ]--1   |                         3
                           |            |
                           3 <-----------

        Then contract this object upside down using leg 3 when conecting to mpo_up_opj.
        @endverbatim

     Step 3: Zip mpo_dn and mpo_up with the running result

                                  3                           1
                                  |                           |
            |------0  -  0--[ mpo_up_opj ]--1                 |----- 0
            |                     |                           |
            |                     2                           |
         [result]                 |                  =        |
            |                     2                           |
            |                     |                           |
            |------1  -  0--[ mpo_dn_opi ]--1                 |----- 2
                                  |                           |
                                  3                           3

    Step 4: Trace the physical indices 1 and 3, and divide by two, to get a new running result, and go to the next site
    In practice, we can simply contract these indices immediately in step 3, instead of doing a separate trace.

    */

    // Generate gates for the operators
    if constexpr(settings::verbose_circuit) tools::log->trace("Computing Tr (τ_{} σ_{}) / Tr(σ_{})", pos_szi, pos_szj, pos_szj);
    auto t_olap = tid::ur();
    t_olap.tic();

    auto id         = tenx::TensorCast(qm::spin::half::id);
    auto si         = tenx::TensorCast(szi);
    auto sj         = tenx::TensorCast(szj);
    auto result     = Eigen::Tensor<cx64, 2>(1, 1);
    auto temp2      = Eigen::Tensor<cx64, 2>();
    auto mpo_dn_opi = Eigen::Tensor<cx64, 4>();
    auto mpo_up_opj = Eigen::Tensor<cx64, 4>();
    result.setConstant(1.0);
    auto &threads = tenx::threads::get();
    for(size_t idx = 0; idx < mpo_layer.size(); ++idx) {
        auto opi = idx == pos_szi ? si : id;
        auto opj = idx == pos_szj ? sj : id;
        auto dim = mpo_layer[idx].dimensions();

        {
            auto t_opj = tid::tic_token("opj");
            mpo_up_opj.resize(dim);
            mpo_up_opj.device(*threads->dev) = mpo_layer[idx].contract(opj, tenx::idx({2}, {0}));
        }
        {
            auto t_opj = tid::tic_token("opi");
            mpo_dn_opi.resize(dim);
            mpo_dn_opi.device(*threads->dev) = mpo_layer[idx].conjugate().contract(opi, tenx::idx({3}, {0}));
        }
        {
            auto t_updntr = tid::tic_token("updntr");
            temp2.resize(mpo_dn_opi.dimension(1), mpo_up_opj.dimension(1));
            temp2.device(*threads->dev) = result.contract(mpo_dn_opi, tenx::idx({0}, {0})).contract(mpo_up_opj, tenx::idx({0, 3, 2}, {0, 2, 3}));
            result                      = temp2 / temp2.constant(2.0); // Divide by two for each trace
        }
    }
    //    tools::log->info("result {:+.16f}{:+.16f}i | time tot {:.3e}", std::real(result.coeff(0)), std::imag(result.coeff(0)), t_olap.get_last_interval());
    return result.coeff(0);
}

Eigen::Tensor<fp64, 2> qm::lbit::get_lbit_correlation_matrix(const std::vector<std::vector<qm::Gate>> &unitary_circuit, size_t sites) {
    /*! \brief Calculates the operator overlap O(i,j) = Tr(𝜌_i σ^z_j) / Tr(𝜌_j)
                                                      = Tr(τ^z_i σ^z_j) / 2^L
                                                      = Tr(Uσ^z_iU† σ^z_j) / 2^L
        Where
            𝜌_i   = 1/2^L (1 + τ^z_i)   a density matrix describing a state localized around site i
            τ^z_i = U σ^z_i U†,        is an l-bit operator at site i
            U                           is a unitary transformation in the form of a finite-depth circuit.
        The second equality comes from the fact that Tr(τ^z_i) = 0 (l-bit operators are traceless) and Tr(1) = 2^L.
        An l-bit fully localized at site i gives O(i,i) = 1,  and when fully delocalized one gets O(i,j) = 0.5

        Read more here:
        https://link.aps.org/doi/10.1103/PhysRevB.91.085425
        https://onlinelibrary.wiley.com/doi/10.1002/andp.201600322
    */
    auto t_corrmat    = tid::tic_scope("corrmat3");
    auto ssites       = safe_cast<long>(sites);
    auto lbit_corrmat = Eigen::Tensor<cx64, 2>(ssites, ssites);
    for(long j = 0; j < ssites; j++) {
        for(long i = 0; i < ssites; i++) {
            lbit_corrmat(i, j) = qm::lbit::get_lbit_2point_correlator3(unitary_circuit, qm::spin::half::sz, static_cast<size_t>(i), qm::spin::half::sz,
                                                                       static_cast<size_t>(j), ssites);
        }
    }
    // We require that lbit_overlap(i,j) has rows that sum up to 1
    auto lbit_corrmap = tenx::MatrixMap(lbit_corrmat);
    auto sums_rowwise = lbit_corrmap.rowwise().sum();
    auto sums_colwise = lbit_corrmap.colwise().sum();
    if(not sums_colwise.cwiseAbs().isOnes(5e-1) or not sums_rowwise.cwiseAbs().isOnes(5e-1)) {
        if constexpr(settings::debug) tools::log->warn("lbit_overlap: \n{}\n", linalg::tensor::to_string(lbit_corrmat.real(), 15));
        tools::log->warn("lbit overlap cols or rows do not sum to one. Perhaps normalization is wrong.\n"
                         "col sums: {}\n"
                         "row sums: {}\n",
                         linalg::matrix::to_string(sums_colwise.real(), 6), linalg::matrix::to_string(sums_rowwise.real().transpose(), 6));
    }
    if constexpr(settings::debug) {
        if(lbit_corrmap.imag().mean() > 1e-12) {
            throw except::runtime_error("lbit_corrmat has large imaginary component:\n{}\n", linalg::tensor::to_string(lbit_corrmat, 15));
        }
    }
    return lbit_corrmat.real();
}

Eigen::Tensor<fp64, 2> qm::lbit::get_lbit_correlation_matrix(const std::vector<Eigen::Tensor<cx64, 4>> &mpo_layer) {
    auto                   ssites = safe_cast<long>(mpo_layer.size());
    Eigen::Tensor<cx64, 2> lbit_corrmat(ssites, ssites);

    /*! \brief Calculates the correlator as an operator overlap O(i,j) = Tr(𝜌_i σ^z_j) / Tr(𝜌_j)
                                                                       = Tr(τ^z_i σ^z_j) / 2^L
                                                                       = Tr(Uσ^z_iU† σ^z_j) / 2^L
        Where
            𝜌_i   = 1/2^L (1 + τ^z_i)   a density matrix describing a state localized around site i
            τ^z_i = U σ^z_i  U†,        is an l-bit operator at site i
            U                           is a unitary transformation in the form of a finite-depth circuit.
        The second equality comes from the fact that Tr(τ^z_i) = 0 (l-bit operators are traceless) and Tr(1) = 2^L.
        An l-bit fully localized at site i gives O(i,i) = 1,  and when fully delocalized one gets O(i,j) = 0.5

        Read more here:
        https://link.aps.org/doi/10.1103/PhysRevB.91.085425
        https://onlinelibrary.wiley.com/doi/10.1002/andp.201600322
    */
    auto t_corrmat = tid::tic_scope("corrmat4");
    for(long j = 0; j < ssites; j++) {
        for(long i = 0; i < ssites; i++) {
            lbit_corrmat(i, j) =
                qm::lbit::get_lbit_2point_correlator4(mpo_layer, qm::spin::half::sz, static_cast<size_t>(i), qm::spin::half::sz, static_cast<size_t>(j));
        }
    }
    // We require that lbit_overlap(i,j) has rows that sum up to 1
    auto lbit_corrmap = tenx::MatrixMap(lbit_corrmat);
    auto sums_rowwise = lbit_corrmap.rowwise().sum();
    auto sums_colwise = lbit_corrmap.colwise().sum();
    if(not sums_colwise.cwiseAbs().isOnes(5e-1) or not sums_rowwise.cwiseAbs().isOnes(5e-1)) {
        if constexpr(settings::debug) tools::log->warn("lbit_overlap: \n{}\n", linalg::tensor::to_string(lbit_corrmat.real(), 15));
        tools::log->warn("lbit overlap cols or rows do not sum to one. Perhaps normalization is wrong.\n"
                         "col sums: {}\n"
                         "row sums: {}\n",
                         linalg::matrix::to_string(sums_colwise.real(), 6), linalg::matrix::to_string(sums_rowwise.real().transpose(), 6));
    }

    if constexpr(settings::debug) {
        if(lbit_corrmap.imag().mean() > 1e-12) {
            throw except::runtime_error("lbit_corrmat has large imaginary component:\n{}\n", linalg::tensor::to_string(lbit_corrmat, 15));
        }
    }
    return lbit_corrmat.real();
}

Eigen::Tensor<fp64, 2> qm::lbit::get_lbit_correlation_matrix(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers) {
    if(mpo_layers.empty()) throw except::logic_error("mpo layers is empty");
    for(const auto &mpo_layer : mpo_layers) {
        if(mpo_layer.empty()) throw except::logic_error("mpo layer is empty");
        if(mpo_layer.size() != mpo_layers.front().size()) throw except::logic_error("mpo layer size mismatch");
    }
    auto                   ssites = safe_cast<long>(mpo_layers.front().size());
    Eigen::Tensor<cx64, 2> lbit_corrmat(ssites, ssites);

    /*! \brief Calculates the correlator as an operator overlap O(i,j) = Tr(𝜌_i σ^z_j) / Tr(𝜌_j)
                                                                       = Tr(τ^z_i σ^z_j) / 2^L
                                                                       = Tr(Uσ^z_iU† σ^z_j) / 2^L
        Where
            𝜌_i   = 1/2^L (1 + τ^z_i)   a density matrix describing a state localized around site i
            τ^z_i = U σ^z_i  U†,        is an l-bit operator at site i
            U                           is a unitary transformation in the form of a finite-depth circuit.
        The second equality comes from the fact that Tr(τ^z_i) = 0 (l-bit operators are traceless) and Tr(1) = 2^L.
        An l-bit fully localized at site i gives O(i,i) = 1,  and when fully delocalized one gets O(i,j) = 0.5

        Read more here:
        https://link.aps.org/doi/10.1103/PhysRevB.91.085425
        https://onlinelibrary.wiley.com/doi/10.1002/andp.201600322
    */
    auto t_corrmat = tid::tic_scope("corrmat5");
    for(long i = 0; i < ssites; ++i) {
        lbit_corrmat.chip(i, 0) = qm::lbit::get_lbit_2point_correlator5(mpo_layers, qm::spin::half::sz, static_cast<size_t>(i), qm::spin::half::sz);
        auto rowsum             = Eigen::Tensor<cx64, 0>(lbit_corrmat.chip(i, 0).sum());
        tools::log->info("lbit_corrmat row {}/{} sum: {:12.8f}", i, ssites, rowsum.coeff(0));
    }

    // We require that lbit_overlap(i,j) has rows that sum up to 1
    auto lbit_corrmap = tenx::MatrixMap(lbit_corrmat);
    auto sums_rowwise = lbit_corrmap.rowwise().sum();
    auto sums_colwise = lbit_corrmap.colwise().sum();
    if(not sums_colwise.cwiseAbs().isOnes(5e-1) or not sums_rowwise.cwiseAbs().isOnes(5e-1)) {
        if constexpr(settings::debug) tools::log->warn("lbit_overlap: \n{}\n", linalg::tensor::to_string(lbit_corrmat.real(), 15));
        tools::log->warn("lbit overlap cols or rows do not sum to one. Perhaps normalization is wrong.\n"
                         "col sums: {}\n"
                         "row sums: {}\n",
                         linalg::matrix::to_string(sums_colwise.real(), 6), linalg::matrix::to_string(sums_rowwise.real().transpose(), 6));
    }
    if constexpr(settings::debug) {
        if(lbit_corrmap.imag().mean() > 1e-12) {
            tools::log->warn("lbit_corrmat has large imaginary component:\n{}\n", linalg::tensor::to_string(lbit_corrmat, 15));
        }
    }
    return lbit_corrmat.real();
}
Eigen::Tensor<fp64, 2> qm::lbit::get_lbit_correlation_matrix2(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers) {
    if(mpo_layers.empty()) throw except::logic_error("mpo layers is empty");
    for(const auto &mpo_layer : mpo_layers) {
        if(mpo_layer.empty()) throw except::logic_error("mpo layer is empty");
        if(mpo_layer.size() != mpo_layers.front().size()) throw except::logic_error("mpo layer size mismatch");
    }
    auto                   ssites = safe_cast<long>(mpo_layers.front().size());
    Eigen::Tensor<cx64, 2> lbit_corrmat(ssites, ssites);

    /*! \brief Calculates the correlator as an operator overlap O(i,j) = Tr(𝜌_i σ^z_j) / Tr(𝜌_j)
                                                                       = Tr(τ^z_i σ^z_j) / 2^L
                                                                       = Tr(Uσ^z_iU† σ^z_j) / 2^L
        Where
            𝜌_i   = 1/2^L (1 + τ^z_i)   a density matrix describing a state localized around site i
            τ^z_i = U σ^z_i  U†,        is an l-bit operator at site i
            U                           is a unitary transformation in the form of a finite-depth circuit.
        The second equality comes from the fact that Tr(τ^z_i) = 0 (l-bit operators are traceless) and Tr(1) = 2^L.
        An l-bit fully localized at site i gives O(i,i) = 1,  and when fully delocalized one gets O(i,j) = 0.5

        Read more here:
        https://link.aps.org/doi/10.1103/PhysRevB.91.085425
        https://onlinelibrary.wiley.com/doi/10.1002/andp.201600322
    */
    auto t_corrmat = tid::tic_scope("corrmat6");
    for(long i = 0; i < ssites; ++i) {
        lbit_corrmat.chip(i, 0) = qm::lbit::get_lbit_2point_correlator6(mpo_layers, qm::spin::half::sz, static_cast<size_t>(i), qm::spin::half::sz);
        auto rowsum             = Eigen::Tensor<cx64, 0>(lbit_corrmat.chip(i, 0).sum());
        tools::log->info("lbit_corrmat row {}/{} sum: {:12.8f}", i, ssites, rowsum.coeff(0));
    }

    // We require that lbit_overlap(i,j) has rows that sum up to 1
    auto lbit_corrmap = tenx::MatrixMap(lbit_corrmat);
    auto sums_rowwise = lbit_corrmap.rowwise().sum();
    auto sums_colwise = lbit_corrmap.colwise().sum();
    if(not sums_colwise.cwiseAbs().isOnes(5e-1) or not sums_rowwise.cwiseAbs().isOnes(5e-1)) {
        if constexpr(settings::debug) tools::log->warn("lbit_overlap: \n{}\n", linalg::tensor::to_string(lbit_corrmat.real(), 15));
        tools::log->warn("lbit overlap cols or rows do not sum to one. Perhaps normalization is wrong.\n"
                         "col sums: {}\n"
                         "row sums: {}\n",
                         linalg::matrix::to_string(sums_colwise.real(), 6), linalg::matrix::to_string(sums_rowwise.real().transpose(), 6));
    }
    if constexpr(settings::debug) {
        if(lbit_corrmap.imag().mean() > 1e-12) {
            tools::log->warn("lbit_corrmat has large imaginary component:\n{}\n", linalg::tensor::to_string(lbit_corrmat, 15));
        }
    }
    return lbit_corrmat.real();
}

template<typename Scalar>
Eigen::Tensor<qm::lbit::RealScalar<Scalar>, 2> qm::lbit::get_lbit_correlation_matrix(const std::vector<std::vector<qm::Gate>> &unitary_circuit, size_t sites, size_t max_num_states,
                                                             double tol) {
    /*! \brief Calculates the correlation < τ^z_i σ^z_j > for a product state with spins aligned along +x = [(1,1)^T]^{\otimes L}
     *
     * We make the approximation of the correlator
     *          Tr(τ^z_i σ^z_j) / Z = 1/Z  Σ_[α=1...Z] ⟨ψ_α|τ^z_i  σ^z_j |ψ_α⟩
     *
     * where
     *          Z <= 2^L, i.e. less or equal to the Hilbert space dimension. Equality implies that the relation is exact.
     *          τ^z_i = U σ^z_i U†
     *          {ψ_α}: an ON-basis (we choose all product states combining spin up/down in x direction on each site)
     *
     *
     * We calculate each ⟨ψ|U σ^z_i U†  σ^z_j |ψ⟩, by taking the overlap ⟨state_i|state_j⟩ with
     * |state_i⟩ = U σ^z_i U† |ψ⟩
     * |state_j⟩ = σ^z_j |ψ⟩
     *
     * Note how the operators in state_i flip due to the adjoint:
     *              |state_i⟩† = (U σ^z_i U† |ψ⟩)†
     *                         = |ψ⟩† (U σ^z_i U†)†
     *                         = ⟨ψ| U σ^z_i U†
     *                         = ⟨state_i|
     */
    auto svd_cfg             = svd::config();
    svd_cfg.truncation_limit = tol;
    svd_cfg.switchsize_gesdd = settings::precision::svd_switchsize_bdc;
    svd_cfg.rank_max         = 8;
    svd_cfg.svd_lib          = svd::lib::lapacke;
    svd_cfg.svd_rtn          = svd::rtn::geauto;

    auto flipbit = [](size_t n, const size_t pos) -> size_t { return n ^= static_cast<size_t>(1) << pos; };
    auto flipall = [&flipbit](size_t n, const size_t len) -> size_t {
        for(size_t pos = 0; pos < len; ++pos) n = flipbit(n, pos);
        return n;
    };
    auto findidx = [](const auto &seq, size_t val) -> std::ptrdiff_t {
        auto it = std::find(seq.begin(), seq.end(), val);
        if(it != seq.end()) return std::distance(seq.begin(), it);
        return -1l;
    };

    auto state            = StateFinite(AlgorithmType::fLBIT, sites, 0, 2);
    auto num_states       = static_cast<Eigen::Index>(std::pow(2, sites));
    auto ssites           = static_cast<Eigen::Index>(sites);
    auto szi              = tenx::TensorCast(qm::spin::half::sz);
    auto szj              = tenx::TensorCast(qm::spin::half::sz);
    auto lbit_corrmat_vec = std::vector<Eigen::Tensor<fp64, 2>>();
    auto lbit_corrmat_avg = Eigen::Tensor<fp64, 2>(ssites, ssites);
    auto lbit_corrmat_typ = Eigen::Tensor<fp64, 2>(ssites, ssites);
    auto lbit_corrmat_err = Eigen::Tensor<fp64, 2>(ssites, ssites);
    lbit_corrmat_avg.setZero();
    auto bitseqs = std::vector<size_t>();

    bitseqs.reserve(static_cast<size_t>(num_states) / 2);
    for(auto b : num::range<size_t>(0, num_states)) {
        auto it1 = std::find(bitseqs.begin(), bitseqs.end(), b);
        auto it2 = std::find(bitseqs.begin(), bitseqs.end(), flipall(b, sites));
        if(it1 == bitseqs.end() and it2 == bitseqs.end()) bitseqs.emplace_back(b);
    }
    tools::log->info("numstates: {} | bitseqs size(): {}", num_states, bitseqs.size());

    auto   t_r        = tid::ur();
    auto   t_i        = tid::ur();
    auto   t_i_u      = tid::ur();
    auto   t_j        = tid::ur();
    long   bond_maxu  = 0;
    long   bond_maxi  = 0;
    long   bond_maxj  = 0;
    size_t num_caches = 0;
    for(const auto &[idx, b] : iter::enumerate(bitseqs)) {
        t_r.tic();
        auto bstr = std::to_string(b);
        tools::finite::mps::init::set_product_state_on_axis_using_pattern(state, StateInitType::REAL, "x", bstr);
        auto lbit_corrmat = Eigen::Tensor<cx64, 2>(ssites, ssites);
        lbit_corrmat.setZero();
        StateFinite<Scalar> state_ud = state;
        tools::finite::mps::apply_circuit(state_ud, unitary_circuit, CircuitOp::ADJ, false, GateMove::ON, svd_cfg); // Apply U† on state_ud
        bond_maxu = std::max<long>(bond_maxu, state_ud.get_largest_bond());
        t_r.toc();

        for(long i = 0; i < ssites; i++) {
            t_i.tic();
            StateFinite<Scalar> state_i = state_ud; // Make a copy of state_ud
            state_i.get_mps_site(i).apply_mpo(szi); // Apply σ^z_i on state_i
            t_i_u.tic();

            tools::finite::mps::apply_circuit(state_i, unitary_circuit, CircuitOp::NONE, false, GateMove::ON, svd_cfg); // Apply U on state_i
            bond_maxi = std::max<long>(bond_maxi, state_i.get_largest_bond());
            t_i_u.toc();
            t_i.toc();
            for(long j = 0; j < ssites; j++) {
                t_j.tic();
                {
                    // Because states ↑↓↑↑↑↑↑↓ and ↑↑↑↑↑↑↑↓ give correlation matrices that are identical on column j = 1,
                    // we can save computation by checking if the matrix element (i,1) has already been computed for the state with flipped spin.
                    auto bflp = flipbit(b, static_cast<size_t>(j));
                    long bidx = findidx(bitseqs, bflp); // r index where we might find a precomputed value for this entry in lbit_corrmat_vec
                    if(bidx >= 0 and static_cast<size_t>(bidx) < lbit_corrmat_vec.size()) {
                        // Found a precomputed value!
                        lbit_corrmat(i, j) = lbit_corrmat_vec.at(static_cast<size_t>(bidx))(i, j);
                        t_j.toc();
                        num_caches++;
                        continue;
                    }
                }

                StateFinite<Scalar> state_j = state;
                state_j.get_mps_site(j).apply_mpo(szj); // Apply σ^z_j on state j
                bond_maxj          = std::max<long>(bond_maxj, state_j.get_largest_bond());
                auto sziszj_ev     = tools::finite::ops::overlap(state_i, state_j);
                lbit_corrmat(i, j) = sziszj_ev;
                t_j.toc();
            }
        }
        lbit_corrmat_vec.emplace_back(lbit_corrmat.real());
        std::tie(lbit_corrmat_avg, lbit_corrmat_typ, lbit_corrmat_err) = qm::lbit::get_lbit_correlation_statistics(lbit_corrmat_vec);
        tools::log->info("lbit_corrmat  caches {}  : \n{}\n", num_caches, linalg::tensor::to_string(lbit_corrmat.real(), 15));
        if(idx + 1 >= max_num_states) break;

        //        lbit_correlator_avg_diff_norm = tenx::MatrixCast(lbit_corrmat_avg - lbit_corrmat_ovg, ssites, ssites).norm() / static_cast<double>(ssites);
        //        tools::log->info("lbit_correlator_avg_diff_norm: {:.3e} idx {} bmax {} {} {} ", lbit_correlator_avg_diff_norm, idx, bond_maxu, bond_maxi,
        //        bond_maxj);

        //        auto plt      = AsciiPlotter("lbit decay", 80, 30);
        //        auto lognoinf = [](const auto &v) -> double {
        //            auto res = std::log10(std::abs(v));
        //            if(std::isinf(res) or std::isinf((-res))) return -16.0;
        //            return res;
        //        };
        //        {
        //            auto [cls, sse, y, c] = qm::lbit::get_characteristic_length_scale(lbit_corrmat);
        //            auto yrel             = num::cast<fp64>(y, [](const auto &v) { return std::real(v); });
        //            auto ylog             = num::cast<fp64>(yrel, lognoinf);
        //            plt.addPlot(ylog, fmt::format("i_j cls {:.3e} sse {:.3e}: {::+.4e}", cls, sse, yrel), '.');
        //        }
        //        {
        //            auto [cls, sse, y, c] = qm::lbit::get_characteristic_length_scale(lbit_corrmat_avg);
        //            auto yrel             = num::cast<fp64>(y, [](const auto &v) { return std::real(v); });
        //            auto ylog             = num::cast<fp64>(yrel, lognoinf);
        //            plt.addPlot(ylog, fmt::format("avg cls {:.3e} sse {:.3e}: {::+.4e}", cls, sse, yrel), 'o');
        //        }
        //        plt.enable_legend();
        //        plt.show();
        //                if(std::abs(avg_sums_std) < tol) break;
    }
    tools::log->info("t_r {:.3e} {:.3e} | t_i {:.3e} {:.3e} (u {:.3e} {:.3e}) | t_j {:.3e} {:.3e} | total {:.3e}", t_r.get_time(), t_r.get_time_avg(),
                     t_i.get_time(), t_i.get_time_avg(), t_i_u.get_time(), t_i_u.get_time_avg(), t_j.get_time(), t_j.get_time_avg(),
                     t_r.get_time() + t_i.get_time() + t_j.get_time());

    return lbit_corrmat_avg;
}

std::tuple<Eigen::Tensor<fp64, 2>, Eigen::Tensor<fp64, 2>, Eigen::Tensor<fp64, 2>>
    qm::lbit::get_lbit_correlation_statistics(const std::vector<Eigen::Tensor<fp64, 2>> &lbit_corrmats) {
    auto                   t_stats = tid::tic_scope("stats");
    Eigen::Tensor<fp64, 2> avg, typ, err;
    if(lbit_corrmats.size() == 1) {
        avg = lbit_corrmats.front();
        typ = lbit_corrmats.front();
        err = avg.constant(0.0);
    } else if(not lbit_corrmats.empty()) {
        long   rows = lbit_corrmats.front().dimension(0); // dim 0
        long   cols = lbit_corrmats.front().dimension(1); // dim 1
        size_t reps = lbit_corrmats.size();               // dim 2
        avg.resize(rows, cols);
        typ.resize(rows, cols);
        err.resize(rows, cols);
        for(long c = 0; c < cols; c++) {
            for(long r = 0; r < rows; r++) {
                std::vector<fp64> slice;
                slice.reserve(reps);
                for(const auto &elem : lbit_corrmats) slice.emplace_back(elem(r, c));
                avg(r, c) = stat::mean(slice);
                typ(r, c) = stat::typical(slice);
                err(r, c) = stat::sterr(slice);
                if(typ(r, c) < 1e-64) { tools::log->warn("Very small typical value: {:.3e}", typ(r, c)); }
            }
        }
    }
    return {avg, typ, err};
}

std::tuple<double, double, double, std::vector<double>, size_t>
    qm::lbit::get_characteristic_length_scale(const Eigen::Tensor<fp64, 2> &lbit_corrmat_disorder_mean, MeanType meanType) {
    auto t_cls = tid::tic_scope("cls");
    // Permute the disorder averaged correlation matrix
    auto                   perm_dis_mean = get_permuted(lbit_corrmat_disorder_mean, meanType);
    Eigen::Tensor<fp64, 1> perm_dis_mean_site_mean;
    // Take the mean of each column to get an estimate of the lbit decay
    switch(meanType) {
        // Do not consider exact zeros when taking the mean
        case MeanType::ARITHMETIC: {
            auto mat                = tenx::MatrixMap(perm_dis_mean);
            auto sum                = mat.colwise().sum();
            auto nnz                = mat.colwise().count().cast<double>();
            perm_dis_mean_site_mean = tenx::TensorCast(sum.cwiseQuotient(nnz));
            break;
        }
        case MeanType::GEOMETRIC: {
            auto abslog             = [](const auto &val) -> double { return val == 0 ? 0 : std::log(std::abs(val)); };
            auto mat                = tenx::MatrixCast(perm_dis_mean, perm_dis_mean.dimension(0), perm_dis_mean.dimension(1));
            auto log                = mat.unaryExpr(abslog).eval();
            auto sum                = log.colwise().sum();
            auto nnz                = log.colwise().count().cast<double>();
            perm_dis_mean_site_mean = tenx::TensorCast(sum.cwiseQuotient(nnz)).exp(); // Typical over each l-bit
            break;
        }
    }

    // Data becomes noisy if the exponential has decayed, so find a cutoff to get the slope using only the first part of the curve
    auto   ymean = std::vector<fp64>(perm_dis_mean_site_mean.data(), perm_dis_mean_site_mean.data() + perm_dis_mean_site_mean.size());
    auto   ylogs = num::cast<double>(ymean, [](const auto v) { return std::log(std::abs(v)); });
    auto   xdata = num::range<double>(0, ylogs.size());
    size_t c     = 0; // Counts number of valid y-values. Also, the ending point.
    for(const auto &[i, v] : iter::enumerate(ylogs)) {
        if(std::isnan(v)) break;
        if(std::isinf(v)) break;
        if(i > 2 and v > 0.5 * (ylogs[i - 1] + ylogs[i - 2])) break; // Check if it is increasing
        c++;
    }
    if(c <= 2) return {-1.0, -1.0, -1.0, ymean, c};
    size_t e    = c - 1;
    size_t s    = std::min<size_t>(1ul, e); // starting index: skip the first point because the exp decay starts further away
    auto   lfit = stat::linearFit(xdata, ylogs, s, e);
    double cls  = 1.0 / std::abs(lfit.slope);
    tools::log->debug("Computed lbit decay | cls {:>8.6f} | rmsd {:.3e} | R² {:.6f} | using y idx {} to {}: {::.3e}", cls, lfit.rms, lfit.rsq, s, e, ymean);
    //    tools::log->info("Computed lbit decay | coeffs {::>8.6f} | status {} | tol {:.2e}", fit_log.coeffs, fit_log.status, tol);
    return {cls, lfit.rms, lfit.rsq, ymean, c};
}

std::vector<Eigen::Tensor<fp64, 2>> qm::lbit::get_lbit_correlation_matrices(const UnitaryGateProperties &uprop, size_t reps, bool randomize_fields,
                                                                            bool use_mpo) {
    auto t_reps        = tid::tic_scope(fmt::format("reps", reps));
    auto lbit_supp_vec = std::vector<Eigen::Tensor<fp64, 2>>();
    for(size_t rep = 0; rep < reps; ++rep) {
        if(use_mpo) {
            auto mpo_layers = std::vector<std::vector<Eigen::Tensor<cx64, 4>>>();
            for(const auto &layer : uprop.ulayers) mpo_layers.emplace_back(get_unitary_mpo_layer(layer));
            lbit_supp_vec.emplace_back(get_lbit_correlation_matrix(mpo_layers));
            //            auto mpo_layer = merge_unitary_mpo_layers(mpo_layers);
            //            lbit_supp_vec.emplace_back(get_lbit_correlation_matrix(mpo_layer));
            //                                    lbit_supp_vec.emplace_back(qm::lbit::get_lbit_correlations(ulayers, uprop.sites, 2048, 1e-5));
        } else {
            lbit_supp_vec.emplace_back(qm::lbit::get_lbit_correlation_matrix(uprop.ulayers, uprop.sites));
        }
        if(reps > 1) {
            uprop.ulayers.clear();
            if(randomize_fields) {
                uprop.randomize_hvals();
                tools::log->debug("Randomized fields to: {::.4e}", uprop.hvals);
            }
            tools::log->trace("Generating circuit of unitary two-site gates: u_sites {} | u_depth {}", uprop.sites, uprop.depth);
            for(size_t idx = 0; idx < uprop.depth; ++idx) { uprop.ulayers.emplace_back(qm::lbit::create_unitary_2site_gate_layer(uprop)); }
        }
    }

    return lbit_supp_vec;
}
std::vector<Eigen::Tensor<fp64, 2>> qm::lbit::get_lbit_correlation_matrices2(const UnitaryGateProperties &uprop, size_t reps, bool randomize_fields,
                                                                             bool use_mpo) {
    auto t_reps        = tid::tic_scope(fmt::format("reps", reps));
    auto lbit_supp_vec = std::vector<Eigen::Tensor<fp64, 2>>();
    for(size_t rep = 0; rep < reps; ++rep) {
        if(use_mpo) {
            auto mpo_layers = std::vector<std::vector<Eigen::Tensor<cx64, 4>>>();
            for(const auto &layer : uprop.ulayers) mpo_layers.emplace_back(get_unitary_mpo_layer(layer));
            lbit_supp_vec.emplace_back(get_lbit_correlation_matrix2(mpo_layers));
            //            auto mpo_layer = merge_unitary_mpo_layers(mpo_layers);
            //            lbit_supp_vec.emplace_back(get_lbit_correlation_matrix(mpo_layer));
            //                                    lbit_supp_vec.emplace_back(qm::lbit::get_lbit_correlations(ulayers, uprop.sites, 2048, 1e-5));
        } else {
            lbit_supp_vec.emplace_back(qm::lbit::get_lbit_correlation_matrix(uprop.ulayers, uprop.sites));
        }
        if(reps > 1) {
            uprop.ulayers.clear();
            if(randomize_fields) {
                uprop.randomize_hvals();
                tools::log->debug("Randomized fields to: {::.4e}", uprop.hvals);
            }
            tools::log->trace("Generating circuit of unitary two-site gates: u_sites {} | u_depth {}", uprop.sites, uprop.depth);
            for(size_t idx = 0; idx < uprop.depth; ++idx) { uprop.ulayers.emplace_back(qm::lbit::create_unitary_2site_gate_layer(uprop)); }
        }
    }

    return lbit_supp_vec;
}
/* clang-format off */
qm::lbit::lbitSupportAnalysis qm::lbit::get_lbit_support_analysis(const UnitaryGateProperties      & u_defaults,
                                                                  std::vector<size_t>                u_dpths,
                                                                  std::vector<double>                u_fmixs,
                                                                  std::vector<double>                u_lambdas,
                                                                  std::vector<LbitCircuitGateWeightKind> u_wkinds,
                                                                  std::vector<LbitCircuitGateMatrixKind> u_mkinds) {
    auto t_lbit_analysis = tid::tic_scope("lbit_analysis");
    if(u_dpths.empty()) u_dpths = {u_defaults.depth};
    if(u_fmixs.empty()) u_fmixs = {u_defaults.fmix};
    if(u_lambdas.empty()) u_lambdas = {u_defaults.lambda};
    if(u_wkinds.empty()) u_wkinds = {u_defaults.wkind};
    if(u_mkinds.empty()) u_mkinds = {u_defaults.mkind};

    auto reps = settings::flbit::cls::num_rnd_circuits;
    auto rndh = settings::flbit::cls::randomize_hfields;

    auto lognoinf      = [](const auto &v) -> double {
        auto res = std::log10(std::abs(std::real(v)));
        if(std::isinf(res) or std::isinf((-res))) return -16.0;
        return res;
    };

    lbitSupportAnalysis lbitSA(u_dpths.size(), u_fmixs.size(), u_lambdas.size(), u_wkinds.size(), u_mkinds.size() , reps, u_defaults.sites);
    lbitSupportAnalysis lbitSA2(u_dpths.size(), u_fmixs.size(),u_lambdas.size(), u_wkinds.size(), u_mkinds.size() , reps, u_defaults.sites);
    std::array<long, 6> offset6{}, extent6{};
    std::array<long, 8> offset8{}, extent8{};
    auto i_width = safe_cast<long>(u_defaults.sites);
    extent8 = {1, 1, 1, 1, 1,1,  i_width, i_width};

    for (const auto & [i_dpth, u_dpth] : iter::enumerate<long>(u_dpths))
    for (const auto & [i_fmix, u_fmix] : iter::enumerate<long>(u_fmixs))
    for (const auto & [i_lambda, u_lambda] : iter::enumerate<long>(u_lambdas))
    for (const auto & [i_wkind, u_wkind] : iter::enumerate<long>(u_wkinds))
    for (const auto & [i_mkind, u_mkind] : iter::enumerate<long>(u_mkinds))
    {
        auto uprop = u_defaults;
        uprop.depth = u_dpth;
        uprop.fmix = u_fmix;
        uprop.lambda = u_lambda;
        uprop.wkind = u_wkind;
        uprop.mkind = u_mkind;
        bool use_mpo = uprop.depth >= settings::flbit::cls::mpo_circuit_switchdepth;
        tools::log->debug("Computing lbit supports | reps {} | rand h {} | mpo {} | {}", reps, rndh, use_mpo,uprop.string());
        auto lbit_corrmat_vec = get_lbit_correlation_matrices(uprop, reps, rndh, use_mpo);

        auto t_post = tid::tic_scope("post");
        for (const auto & [i_reps, lbit_corrmat] : iter::enumerate<long>(lbit_corrmat_vec)){
            offset8 = {i_dpth, i_fmix, i_lambda, i_wkind, i_mkind, i_reps , 0, 0};
            lbitSA.corrmat.slice(offset8, extent8) = lbit_corrmat.reshape(extent8);
            lbitSA.corroff.slice(offset8, extent8) = get_permuted(lbit_corrmat, MeanType::GEOMETRIC).reshape(extent8);
        }

        auto [lbit_corrmat_avg,lbit_corrmat_typ, lbit_corrmat_err] = qm::lbit::get_lbit_correlation_statistics(lbit_corrmat_vec);
        auto [cls_avg, rms_avg, rsq_avg, yavg, cavg] = qm::lbit::get_characteristic_length_scale(lbit_corrmat_avg, MeanType::ARITHMETIC);
        auto [cls_typ, rms_typ, rsq_typ, ytyp, ctyp] = qm::lbit::get_characteristic_length_scale(lbit_corrmat_typ, MeanType::GEOMETRIC);
        tools::log->info("Computed lbit decay reps {} | rand h {} | mpo {} | {} | time {:8.3f} s | cls {:>8.6f} | rmsd {:.3e} | rsq {:.6f} | decay {:2} sites: {::8.2e}",
                         reps, rndh, use_mpo, uprop.string(), t_lbit_analysis->restart_lap(),cls_typ, rms_typ, rsq_typ, ctyp, ytyp);

        lbitSA.cls_avg_fit(i_dpth, i_fmix, i_lambda, i_wkind, i_mkind) = cls_avg;
        lbitSA.cls_avg_rms(i_dpth, i_fmix, i_lambda, i_wkind, i_mkind) = rms_avg;
        lbitSA.cls_avg_rsq(i_dpth, i_fmix, i_lambda, i_wkind, i_mkind) = rsq_avg;
        lbitSA.cls_typ_fit(i_dpth, i_fmix, i_lambda, i_wkind, i_mkind) = cls_typ;
        lbitSA.cls_typ_rms(i_dpth, i_fmix, i_lambda, i_wkind, i_mkind) = rms_typ;
        lbitSA.cls_typ_rsq(i_dpth, i_fmix, i_lambda, i_wkind, i_mkind) = rsq_typ;


        offset6                              = {i_dpth, i_fmix, i_lambda, i_wkind, i_mkind, 0};
        extent6                              = {1, 1, 1, 1, 1, safe_cast<long>(yavg.size())};
        lbitSA.corravg.slice(offset6, extent6) = Eigen::TensorMap<Eigen::Tensor<fp64, 6>>(yavg.data(), extent6);
        extent6                              = {1, 1, 1, 1, 1, safe_cast<long>(ytyp.size())};
        lbitSA.corrtyp.slice(offset6, extent6) = Eigen::TensorMap<Eigen::Tensor<fp64, 6>>(ytyp.data(), extent6);

        auto t_plot = tid::tic_scope("plot");
        auto plt           = AsciiPlotter("lbit decay", 64, 20);

        {   // Select the half-chain l-bit
            auto off8 = std::array<long,8>{0, 0, 0, 0, 0, 0, i_width/2,0};
            auto ext8 = std::array<long,8>{1, 1, 1, 1, 1, 1, 1, i_width};
            auto ymid_log = num::cast<fp64>(tenx::VectorCast(lbitSA.corrmat.slice (off8, ext8)), lognoinf);
            plt.addPlot(ymid_log, fmt::format("j=L/2: avg cls {:.3e} rmsd {:.3e} rsq {:.6f}: {::+.4e}", cls_avg, rms_avg, rsq_avg, yavg), 'o');
        }
        {   // Select the edge l-bit
            auto off8 = std::array<long,8>{0, 0, 0, 0, 0, 0, 0};
            auto ext8 = std::array<long,8>{1,1, 1, 1, 1, 1, 1, i_width};
            auto yedg_log = num::cast<fp64>(tenx::VectorCast(lbitSA.corrmat.slice(off8, ext8)), lognoinf);
            plt.addPlot(yedg_log, fmt::format("j=0", cls_avg, rms_avg, rsq_avg, yavg), 'x');
        }
        plt.enable_legend();
        plt.show();
        auto lbit_corrmap = tenx::MatrixMap(lbit_corrmat_avg);
        auto mid = safe_cast<long>(lbit_corrmap.rows()/2);
        auto lbit_crossup = 0.5 * (lbit_corrmap.topRightCorner(mid,mid).cwiseAbs().sum() + lbit_corrmap.bottomLeftCorner(mid,mid).cwiseAbs().sum());
        tools::log->info("lbit cross-support: {:.3e}", lbit_crossup);

    }
    /* clang-format on */
    return lbitSA;
}

template<typename Scalar>
StateFinite<Scalar> qm::lbit::transform_to_real_basis(const StateFinite<Scalar>                &state_lbit,
                                                      const std::vector<std::vector<qm::Gate>> &unitary_gates_2site_layers, svd::config svd_cfg) {
    auto                t_map      = tid::tic_scope("l2r");
    StateFinite<Scalar> state_real = state_lbit; // Make a copy
    state_real.set_name("state_real");
    state_real.clear_cache();
    state_real.clear_measurements();
    tools::log->debug("Transforming {} to {} using {} unitary layers: Ψ = U|Ψ'⟩", state_lbit.get_name(), state_real.get_name(),
                      unitary_gates_2site_layers.size());
    tools::finite::mps::apply_circuit(state_real, unitary_gates_2site_layers, CircuitOp::NONE, true, GateMove::ON, svd_cfg);

    tools::finite::mps::normalize_state(state_real, svd_cfg, NormPolicy::IFNEEDED);
    //    status.position  = tensors.get_position<long>();
    //    status.direction = tensors.state->get_direction();

    if constexpr(settings::debug) {
        auto t_dbg = tid::tic_scope("debug");
        // Check normalization
        for(const auto &mps : state_lbit.mps_sites) mps->assert_normalized();
        // Double-check the transform operation
        // Check that the transform backwards is equal to the original state
        auto state_lbit_debug = state_real;
        tools::finite::mps::apply_circuit(state_lbit_debug, unitary_gates_2site_layers, CircuitOp::ADJ, true, GateMove::ON, svd_cfg);
        auto overlap = tools::finite::ops::overlap(state_lbit, state_lbit_debug);
        tools::log->info("Debug overlap after unitary circuit: {:.16f}", overlap);
        if(svd_cfg.truncation_limit.has_value())
            if(std::abs(overlap - 1.0) > 10 * svd_cfg.truncation_limit.value())
                throw except::runtime_error("State overlap after transform back from real is not 1: Got {:.16f}", overlap);
    }
    return state_real;
}
template<typename Scalar>
StateFinite<Scalar> qm::lbit::transform_to_real_basis(const StateFinite<Scalar>                                &state_lbit,
                                                      const std::vector<std::vector<Eigen::Tensor<Scalar, 4>>> &unitary_gates_mpo_layers,
                                                      const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge, svd::config svd_cfg) {
    auto                t_map      = tid::tic_scope("l2r");
    StateFinite<Scalar> state_real = state_lbit; // Make a copy
    state_real.set_name("state_real");
    state_real.clear_cache();
    state_real.clear_measurements();
    svd_cfg.rank_max = safe_cast<long>(static_cast<double>(state_real.get_largest_bond()) * 4);
    tools::log->debug("Transforming {} to {} using {} unitary mpo layers Ψ = U|Ψ'⟩", state_lbit.get_name(), state_real.get_name(),
                      unitary_gates_mpo_layers.size());
    for(const auto &[idx_layer, mpo_layer] : iter::enumerate(unitary_gates_mpo_layers)) {
        tools::finite::ops::apply_mpos(state_real, mpo_layer, ledge, redge, true);
        if((idx_layer + 1) % 1 == 0) {
            tools::finite::mps::normalize_state(state_real, svd_cfg, NormPolicy::ALWAYS);
            svd_cfg.rank_max = safe_cast<long>(static_cast<double>(state_real.get_largest_bond()) * 4);
        }
    }

    tools::finite::mps::normalize_state(state_real, svd_cfg, NormPolicy::IFNEEDED);
    if constexpr(settings::debug) {
        auto t_dbg = tid::tic_scope("debug");
        // Check normalization
        for(const auto &mps : state_lbit.mps_sites) mps->assert_normalized();

        // Double-check the transform operation
        // Check that the transform backwards is equal to the original state
        auto state_lbit_debug = state_real;
        for(const auto &[idx_layer, mpo_layer] : iter::enumerate(unitary_gates_mpo_layers)) {
            tools::finite::ops::apply_mpos(state_lbit_debug, mpo_layer, ledge, redge, false);
            if((idx_layer + 1) % 1 == 0) {
                tools::finite::mps::normalize_state(state_lbit_debug, svd_cfg, NormPolicy::ALWAYS);
                svd_cfg.rank_max = safe_cast<long>(static_cast<double>(state_real.get_largest_bond()) * 2);
            }
        }
        tools::finite::mps::normalize_state(state_lbit_debug, std::nullopt, NormPolicy::IFNEEDED);
        auto overlap = tools::finite::ops::overlap(state_lbit, state_lbit_debug);
        tools::log->info("Debug overlap after unitary circuit: {:.16f}", overlap);
        if(svd_cfg.truncation_limit.has_value())
            if(std::abs(overlap - 1.0) > 10 * svd_cfg.truncation_limit.value())
                throw except::runtime_error("State overlap after transform back from real is not 1: Got {:.16f}", overlap);
    }
    return state_real;
}

template<typename Scalar>
StateFinite<Scalar> qm::lbit::transform_to_lbit_basis(const StateFinite<Scalar>                &state_real,
                                                      const std::vector<std::vector<qm::Gate>> &unitary_gates_2site_layers, svd::config svd_cfg) {
    auto                t_map      = tid::tic_scope("r2l");
    StateFinite<Scalar> state_lbit = state_real; // Make a copy
    state_lbit.set_name("state_lbit");
    state_lbit.clear_cache();
    state_lbit.clear_measurements();
    tools::log->info("Transforming {} to {} using {} unitary layers Ψ' = U†|Ψ⟩", state_real.get_name(), state_lbit.get_name(),
                     unitary_gates_2site_layers.size());
    tools::finite::mps::apply_circuit(state_lbit, unitary_gates_2site_layers, CircuitOp::ADJ, true, GateMove::ON, svd_cfg);

    //    auto svd_cfg = svd::config(status.bond_lim, status.trnc_lim);
    tools::finite::mps::normalize_state(state_lbit, std::nullopt, NormPolicy::IFNEEDED);
    tools::log->debug("time r2l: {:.3e} s", t_map->get_last_interval());
    if constexpr(settings::debug) {
        auto t_dbg = tid::tic_scope("debug");
        // Check normalization
        for(const auto &mps : state_lbit.mps_sites) mps->assert_normalized();

        // Double-check the that transform operation backwards is equal to the original state
        auto state_real_debug = state_lbit;
        tools::finite::mps::apply_circuit(state_real_debug, unitary_gates_2site_layers, CircuitOp::NONE, true, GateMove::ON, svd_cfg);
        auto overlap = tools::finite::ops::overlap(state_real, state_real_debug);
        tools::log->info("Debug overlap: {:.16f}", overlap);
        if(svd_cfg.truncation_limit.has_value())
            if(std::abs(overlap - 1.0) > 10 * svd_cfg.truncation_limit.value())
                throw except::runtime_error("State overlap after transform back from lbit is not 1: Got {:.16f}", overlap);
    }
    return state_lbit;
}
template<typename Scalar>
StateFinite<Scalar> qm::lbit::transform_to_lbit_basis(const StateFinite<Scalar>                                &state_real,
                                                      const std::vector<std::vector<Eigen::Tensor<Scalar, 4>>> &unitary_gates_mpo_layers,
                                                      const Eigen::Tensor<Scalar, 1> &ledge, const Eigen::Tensor<Scalar, 1> &redge, svd::config svd_cfg) {
    auto                t_map      = tid::tic_scope("r2l");
    StateFinite<Scalar> state_lbit = state_real; // Make a copy
    state_lbit.set_name("state_lbit");
    state_lbit.clear_cache();
    state_lbit.clear_measurements();
    svd_cfg.rank_max = safe_cast<long>(static_cast<double>(state_lbit.get_largest_bond()) * 4);
    tools::log->info("Transforming {} to {} using {} unitary mpo layers", state_real.get_name(), state_lbit.get_name(), unitary_gates_mpo_layers.size());
    for(const auto &[idx_layer, mpo_layer] : iter::enumerate_reverse(unitary_gates_mpo_layers)) {
        tools::finite::ops::apply_mpos(state_lbit, mpo_layer, ledge, redge, false);
        if((idx_layer) % 1 == 0) {
            tools::log->info("Normalizing with rank_max {} | max bond {}", svd_cfg.rank_max.value(), state_lbit.get_largest_bond());
            tools::finite::mps::normalize_state(state_lbit, svd_cfg, NormPolicy::ALWAYS);
            svd_cfg.rank_max = safe_cast<long>(static_cast<double>(state_lbit.get_largest_bond()) * 4);
        }
    }

    tools::finite::mps::normalize_state(state_lbit, std::nullopt, NormPolicy::IFNEEDED);
    if constexpr(settings::debug) {
        auto t_dbg = tid::tic_scope("debug");
        // Check normalization
        for(const auto &mps : state_lbit.mps_sites) mps->assert_normalized();

        // Double-check the that transform operation backwards is equal to the original state
        auto state_real_debug = state_lbit;
        for(const auto &[idx_layer, mpo_layer] : iter::enumerate(unitary_gates_mpo_layers)) {
            tools::finite::ops::apply_mpos(state_real_debug, mpo_layer, ledge, redge, true);
            if((idx_layer + 1) % 1 == 0) { tools::finite::mps::normalize_state(state_real_debug, svd_cfg, NormPolicy::ALWAYS); }
        }
        tools::finite::mps::normalize_state(state_real_debug, std::nullopt, NormPolicy::IFNEEDED);
        auto overlap = tools::finite::ops::overlap(state_real, state_real_debug);
        tools::log->info("Debug overlap: {:.16f}", overlap);
        if(svd_cfg.truncation_limit.has_value())
            if(std::abs(overlap - 1.0) > 10 * svd_cfg.truncation_limit.value())
                throw except::runtime_error("State overlap after transform back from lbit is not 1: Got {:.16f}", overlap);
    }
    return state_lbit;
}