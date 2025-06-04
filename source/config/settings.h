#pragma once

#include "debug.h"
#include "enums.h"
#include "tid/enums.h"
#include <string>
#include <thread>
#include <vector>
class Loader;
namespace h5pp {
    class File;
}

/* clang-format off */


/*!
 *  \namespace settings
 *  This namespace contains settings such as time-step length, number of iterations and precision parameters for
 *  the different algorithms.
 */
namespace settings {
    extern void load(Loader &dmrg_config);
    extern void load(std::string_view  config_filename);

    extern bool     algorithm_is_on(AlgorithmType algo_type);
    extern size_t   print_freq(AlgorithmType algo_type);
    extern long     get_bond_min(AlgorithmType algo_type);
    extern long     get_bond_max(AlgorithmType algo_type);
    extern OptRitz  get_ritz(AlgorithmType algo_type);
    extern bool     store_wave_function(AlgorithmType algo_type);
    extern size_t   get_iter_min(AlgorithmType algo_type);
    extern size_t   get_iter_max(AlgorithmType algo_type);

    /*!  \namespace settings::threading Parameters for multithreading
     *   num_threads is the total number of threads, num_threads = OMP_NUM_THREADS + std::threads
     *   If num_threads <= 0 then it is set to OMP_NUM_THREADS, or 1 if OMP_NUM_THREADS is not defined.
     *   If num_threads == OMP_NUM_THREADS then Eigen::Tensor multithreading is disabled.
     */
    namespace threading{
        /*! Total number of threads, num_threads = OMP_NUM_THREADS + std::threads.
         * If num_threads == OMP_NUM_THREADS, disables std::threads for Eigen::Tensor are disabled  num_threads <= 0 will try to use as many as possible
         * */
        inline unsigned int num_threads = 1;                                       /*!< Total number of c++11 threads, (mainly used by Eigen::Tensor)  */
        inline unsigned int max_threads = std::thread::hardware_concurrency();     /*!< Maximum number of threads supported on this runner node */
        inline unsigned int show_threads = false;                                  /*!< Show threading information and exit without running a simulation */

//        inline int omp_threads = 1;                                              /*!< Number of threads for openmp threads used in blas/lapack and Eigen. num_threads <= 0 will try to use as many as possible */
//        inline int stl_threads = 1;                                              /*!< Number of threads for c++11 threading. Used in Eigen::Tensor. stl_threads <= 0 will try to use as many as possible */
    }

    /*!  \namespace settings::input Settings for initialization */
    namespace input{
        inline long        seed                                 = 1;                            /*!< Main seed for the random number generator. */
        inline std::string config_filename                      = "input/input.cfg";            /*!< Default config filename. Can either be a .cfg file or a .h5 file with a config stored as a string in /common/config_file_contents */
        inline std::string config_file_contents;
    }

    /*!  \namespace settings::storage Settings for output-file generation
     *
     * **NOTE: Storage levels**
     *
     *  enum StorageLevel:
     *       - `NONE`:   no data is saved at all
     *       - `LIGHT`:  Tables entries replace last. Mid-chain  data. No MPS
     *       - `NORMAL`: Tables entries are appended. Full chain data. No MPS
     *       - `FULL`:   Tables entries are appended. Full chain data. MPS is saved.
     *
     * **Note: Mid-chain**
     * The 'measurements' table takes mid-chain data so the corresponding tables are redundant and skipped.

     * **Note: Resume**
     * Simulations can only be resumed from a state with `StorageLevel::FULL`, except LBIT simulations, which can always resume.
     *
     */
    namespace storage {
        inline std::string         output_filepath                 = "output/output.h5";            /*!< Name of the output HDF5 file relative to the execution point  */
        inline bool                output_append_seed              = true;                          /*!< Append the seed for the random number generator to output_filepath */
        inline size_t              storage_interval                = 1;                             /*!< Write to file this often, in units of iterations. Applies to StorageEvent::Iteration. */
        inline bool                use_temp_dir                    = true;                          /*!< If true uses a temporary directory for writes in the local drive (usually /tmp) and copies the results afterwards */
        inline size_t              copy_from_temp_freq             = 4;                             /*!< How often, in units of iterations, to copy the hdf5 file in tmp dir to target destination */
        inline std::string         temp_dir                        = "/tmp/DMRG";                   /*!< Local temp directory on the local system. If it does not exist we default to /tmp instead (or whatever is the default) */
        inline unsigned            compression_level               = 1;                             /*!< GZip compression level in HDF5. Choose between [0-9] (0 = off, 9 = max compression) */
        inline ResumePolicy        resume_policy                   = ResumePolicy::IF_UNSUCCESSFUL; /*!< Under which exit condition in the previous simulation should we resume this time */
        inline FileCollisionPolicy file_collision_policy           = FileCollisionPolicy::RESUME;   /*!< What to do when a prior output file is found. Choose between RESUME, REVIVE, BACKUP, RENAME, REPLACE */
        inline FileResumePolicy    file_resume_policy              = FileResumePolicy::FULL;        /*!< Depends on dataset "common/finished_all=bool" FULL: Ignore bool -> Scan .cfg to add missing items. FAST: exit if true. */
        inline std::string         file_resume_name                = ""  ;                          /*!< On file_collision_policy=RESUME|REVIVE: resume from state candidate matching this string. Empty implies any */
        inline size_t              file_resume_iter                = -1ul;                          /*!< On file_collision_policy=RESUME|REVIVE: which iteration to resume from. -1ul implies resume from last available iteration */

        namespace mps::state_emid{
            /*! state_emid is obtained in the xDMRG algorithm (mid energy) */
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace mps::state_emin{
            /*! state_emin is obtained in the fDMRG and xDMRG algorithms (min energy) */
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace mps::state_emax{
            /*! state_emax is obtained in the fDMRG and xDMRG algorithms (max energy) */
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace mps::state_real{
            /*! state_real is obtained in the fLBIT algorithm (in the "real" physical basis) */
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace mps::state_lbit{
            /*! state_lbit is obtained in the fLBIT algorithm (in the lbit basis) */
            inline StoragePolicy policy = StoragePolicy::NONE;
        }
        namespace mpo::model{
            inline StoragePolicy policy = StoragePolicy::NONE;
        }
        namespace table::bonds {
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace table::model{
            inline StoragePolicy policy = StoragePolicy::INIT;
        }
        namespace table::measurements{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::status{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::memory{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::timers{
            inline tid::level  level    = tid::level::normal;
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace table::entanglement_entropies{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::truncation_errors{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::bond_dimensions{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::number_entropies{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::renyi_entropies{
            inline StoragePolicy policy = StoragePolicy::ITER;
        }
        namespace table::opdm_spectrum{
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace table::information_per_scale{
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace table::information_center_of_mass{
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace table::expectation_values_spin_xyz{
            inline StoragePolicy policy = StoragePolicy::FINISH;
        }
        namespace table::random_unitary_circuit{
            inline StoragePolicy policy = StoragePolicy::INIT;
        }
        namespace dataset::lbit_analysis{
            inline StoragePolicy policy = StoragePolicy::INIT;
        }
        namespace dataset::subsystem_entanglement_entropies{
        /*! Entanglement entropy (log2) of all contiguous subsystems */
            inline StoragePolicy policy = StoragePolicy::FINISH;
            inline unsigned long chunksize = 10;
            inline auto bits_err = 1e-8;  /*!< Positive for relative error = 1-bits_found/L, negative for absolute error = L-bits_found */
            inline long eig_size = 4096l; /*!< Maximum matrix size to diagonalize */
            inline long bond_lim = 2048l; /*!< Bond dimension limit during swap operations  */
            inline auto trnc_lim = 1e-8;  /*!< Truncation error limit during swap operations  */
            inline auto precision = Precision::DOUBLE; /*!< Use SINGLE or DOUBLE precision internally */
        }
        namespace dataset::information_lattice{
            /*! Information lattice built from subsystem_entanglement_entropies */
            inline StoragePolicy policy = StoragePolicy::FINISH;
            inline unsigned long chunksize = 10;
        }
        namespace dataset::opdm{
            /*! One-particle density matrix */
            inline StoragePolicy policy = StoragePolicy::FINISH;
            inline unsigned long chunksize = 10;
        }
        namespace dataset::number_probabilities{
        /*! Probability of measuring n particles to the left of site i, for all n and i */
            inline StoragePolicy policy = StoragePolicy::FINISH;
            inline unsigned long chunksize = 10;
        }
        namespace dataset::expectation_values_spin_xyz{
            inline StoragePolicy policy = StoragePolicy::ITER;
            inline unsigned long chunksize = 10;
        }
        namespace dataset::correlation_matrix_spin_xyz{
            inline StoragePolicy policy = StoragePolicy::ITER;
            inline unsigned long chunksize = 10;
        }
        namespace tmp{
            inline std::string hdf5_temp_path;
            inline std::string hdf5_final_path;
        }
    }

    /*!  \namespace settings::timer Settings for performance profiling */
    namespace timer {
        inline tid::level   level     = tid::normal;                   /*!< How much extra to print on exit [normal | higher | highest]  */
    }

    /*! \namespace settings::console Settings for console output */
    namespace console {
        inline bool   timestamp     = false;                          /*!< Whether to put a timestamp on console outputs */
        inline size_t loglevel      = 2;                              /*!< Verbosity [0-6]. Level 0 prints everything, 6 nothing. Level 2 or 3 is recommended for normal use */
        inline size_t logh5pp       = 2;                              /*!< Verbosity of h5pp library [0-6] Level 2 or 3 is recommended for normal use */
    }

    /*! \namespace settings::strategy Settings affecting the convergence rate of the algorithms */
    namespace strategy {
        inline bool                move_sites_when_stuck       = true;                                   /*!< Try moving sites around when stuck */
        inline ProjectionPolicy    projection_policy           = ProjectionPolicy::DEFAULT;              /*!< Enum bitflag to control projections to a symmetry sector {NEVER,+INIT,WARMUP,+STUCK,ITER,+CONVERGED,FINISHED,FORCE,DEFAULT} (+ are DEFAULT) */
        inline bool                use_eigenspinors            = false;                                  /*!< Use random pauli-matrix eigenvectors when initializing each mps site along x,y or z  */
        inline size_t              iter_max_warmup             = 4;                                      /*!< Initial warmup iterations. In DMRG these iterations use an exact solver with reduced bond dimension */
        inline size_t              iter_max_stuck              = 5;                                      /*!< If status.algorithm_saturated_for > 0 then status.algorithm_has_stuck_for +=1. If stuck for this many iterations, we stop. */
        inline size_t              iter_max_saturated          = 5;                                      /*!< Saturation means that both variance and entanglement saturated. But if either saturates for this many iterations, then status.algorithm_saturated_for += 1 */
        inline size_t              iter_min_converged          = 1;                                      /*!< Require convergence at least this many iterations before success */
        inline BlockSizePolicy     dmrg_blocksize_policy       = BlockSizePolicy::MIN;                   /*!< Bitflag to control the adaptive dmrg blocksize {MIN,MAX,ICOM,ICOMPLUS1,SATURATED,STUCK,FIN_BOND,FIN_TRNC} */
        inline size_t              dmrg_min_blocksize          = 1;                                      /*!< Minimum number of sites in a dmrg optimization step. */
        inline size_t              dmrg_max_blocksize          = 4;                                      /*!< Maximum number of sites in a dmrg optimization step. */
        inline long                dmrg_max_prob_size          = 1024*2*1024;                            /*!< Restricts the dmrg blocksize to keep the problem size below this limit. Problem size = chiL * (spindim ** blocksize) * chiR */
        inline BondExpansionPolicy dmrg_bond_expansion_policy  = BondExpansionPolicy::DEFAULT;           /*!< Select options for bond expansion (aka subspace expansion)  [NONE, POSTOPT_1SITE, PREOPT_NSITE_REAR, PREOPT_NSITE_FORE, H1, H2, DEFAULT=POSTOPT_1SITE | H1 | H2] */
        namespace dmrg_bond_expansion {
           inline float               bond_factor              = 1.1f;                                    /*!< Expand the bond dimension by this factor above the current bond dimension limit (value  < 1.0 = disabled) */
            namespace postopt {
                inline float               maxalpha            = 1e-3f;                                   /*!< Upper limit for mixing factors derived from the local residual norms */
                inline float               minalpha            = 1e-15f;                                  /*!< Lower limit for mixing factors derived from the local residual norms */
            }
            namespace preopt {
                inline size_t              maxiter             = 1;                                      /*!< How many Lanczos iterations to use in the nsite bond expansion */
                inline size_t              nkrylov             = 3;                                      /*!< The krylov subspace size to use in nsite bond expansion */
            }
        }
        inline std::string         target_axis                 = "none";                                 /*!< Find an eigenstate with global spin component along this axis. Choose between Choose {none, (+-) x,y or z}  */
        inline std::string         initial_axis                = "none";                                 /*!< Initialize state with global spin component along this axis. Choose {none, (+-) x,y or z}  */
        inline StateInitType       initial_type                = StateInitType::REAL;                    /*!< Initial state can be REAL/CPLX */
        inline StateInit           initial_state               = StateInit::RANDOM_ENTANGLED_STATE;      /*!< Initial configuration for the spin chain (only for finite systems)  */
        inline std::string         initial_pattern             = {};                                     /*!< The actual random spin configuration used for the initial product state (for internal use) */
        inline double              rbds_rate                   = 0.5;                                    /*!< If rbds_rate > 0, runs reverse bond dimension scaling (rbds) after the main algorithm. Values [0,1] represent the shrink factor, while [1,infty] represents a shrink step */
        inline double              rtes_rate                   = 1e1;                                    /*!< If rtes_rate > 1, runs reverse truncation error scaling (rtes) after the main algorithm. Values [1, infty] represent the growth factor for the truncation error limit */
        inline UpdatePolicy        bond_increase_when          = UpdatePolicy::NEVER;                    /*!< If and when to increase the bond dimension limit {NEVER, WARMUP, HALFSWEEP, FULLSWEEP, TRUNCATED, SAT_EVAR, SAT_ALGO, STK_ALGO}. */
        inline double              bond_increase_rate          = 8;                                      /*!< Bond dimension growth rate. Factor if 1<x<=2, constant shift if x > 2, otherwise invalid. */
        inline UpdatePolicy        trnc_decrease_when          = UpdatePolicy::NEVER;                    /*!< If and when to decrease SVD truncation error limit {NEVER, WARMUP, HALFSWEEP, FULLSWEEP, TRUNCATED, SAT_EVAR, SAT_ALGO, STK_ALGO} */
        inline double              trnc_decrease_rate          = 1e-1;                                   /*!< Decrease SVD truncation error limit by this factor. Valid if 0 < x < 1 */
        inline UpdatePolicy        etol_decrease_when          = UpdatePolicy::NEVER;                    /*!< If and when to decrease EIGS tolerance {NEVER, WARMUP, HALFSWEEP, FULLSWEEP, TRUNCATED, SAT_EVAR, SAT_ALGO, STK_ALGO} */
        inline double              etol_decrease_rate          = 1e-1;                                   /*!< Decrease EIGS tolerance by this factor. Valid if 0 < x < 1 */
    }


    /*! \namespace settings::precision Settings for the convergence threshold and precision of MPS, SVD and eigensolvers */
    namespace precision {
        inline ScalarType         algoScalar                      = ScalarType::CX64;            /*!< Scalar type for tensor storage (state, model, edges)  */
        inline ScalarType         optScalar                       = ScalarType::CX64;            /*!< Scalar type for computations (eig, eigs, svd) */
        inline long               eig_max_size                    = 4096  ;                      /*!< Maximum problem size before switching from eig to eigs. */
        inline size_t             eigs_iter_min                   = 1000;                        /*!< Minimum number of iterations for eigenvalue solver. */
        inline size_t             eigs_iter_max                   = 100000;                      /*!< Maximum number of iterations for eigenvalue solver. */
        inline double             eigs_iter_gain                  = 2     ;                      /*!< Increase number of iterations on OptSolver::EIGS by gain^(iters without progress) */
        inline GainPolicy         eigs_iter_gain_policy           = GainPolicy::SAT_EVAR;         /*!< Bitflag for when to increase the eigensolver iterations. Choose one or more: [NEVER, ITERATION, VARSAT, SATURATED, STUCK, FIN_BOND, FIN_TRNC] */
        inline double             eigs_tol_min                    = 1e-14 ;                      /*!< Precision tolerance for halting the eigenvalue solver. */
        inline double             eigs_tol_max                    = 1e-8 ;                       /*!< Precision tolerance for halting the eigenvalue solver. */
        inline int                eigs_ncv                        = 0     ;                      /*!< Basis size (krylov space) in the eigensolver. Set ncv <= 0 for automatic selection */
        inline int                eigs_nev_min                    = 1     ;                      /*!< The minimum number of eigenpairs to request on OptSolver::EIGS */
        inline int                eigs_nev_max                    = 8     ;                      /*!< The maximum number of eigenpairs to request on OptSolver::EIGS when stuck (increases slowly) (ncv may have to increase accordingly) */
        inline long               eigs_max_size_shift_invert      = 4096  ;                      /*!< Maximum problem size allowed for shift-invert of the local (effective) hamiltonian matrix. */

        inline long               eigs_jcb_min_blocksize          = 128   ;                      /*!< Minimum block size used in the block-jacobi preconditioner */
        inline long               eigs_jcb_max_blocksize          = 4096  ;                      /*!< Maximum block size used in the block-jacobi preconditioner (increases up to max when stuck) */


        inline double             svd_truncation_min              = 1e-14 ;                      /*!< Truncation error limit, i.e. discard singular values while the truncation error is lower than this */
        inline double             svd_truncation_max              = 1e-6  ;                      /*!< If truncation error limit is updated (trnc_decrease_when != NEVER), start from this value */
        inline size_t             svd_switchsize_bdc              = 16    ;                      /*!< Linear size of a matrix, below which SVD will use slower but more precise JacobiSVD instead of BDC (default is 16 , good could be ~64) */
        inline bool               svd_save_fail                   = false ;                      /*!< Save failed SVD calculations to file */

        inline auto               use_compressed_mpo              = MpoCompress::DPL;            /*!< Select the compression scheme for the virtual bond dimensions of H  mpos. Select {NONE, SVD (high compression), DPL (high precision)} */
        inline auto               use_compressed_mpo_squared      = MpoCompress::DPL;            /*!< Select the compression scheme for the virtual bond dimensions of H² mpos. Select {NONE, SVD (high compression), DPL (high precision)} */
        inline bool               use_energy_shifted_mpo          = false ;                      /*!< Prevent catastrophic cancellation in H²-E² by subtracting the current energy from the MPOs: H²-E² -> <H-E>² - <(H-E)²> (second term ~ 0). Recommended for fDMRG. */
        inline bool               use_parity_shifted_mpo          = true  ;                      /*!< Redefining H --> (H + Q(σ)) where Q(σ) = 0.5(1 - prod(σ)) = P(-σ) is the (flipped sign) projection operator (prevents degeneracy from mixing sectors) */
        inline bool               use_parity_shifted_mpo_squared  = true  ;                      /*!< Redefining H² --> (H² + Q(σ)) where Q(σ) = 0.5(1 - prod(σ)) = P(-σ) is the (flipped sign) projection operator (prevents degeneracy from mixing sectors) */
        inline double             variance_convergence_threshold  = 1e-12 ;                      /*!< Desired precision on total energy variance. The MPS state is considered good enough when its energy variance reaches below this value */
        inline double             variance_saturation_sensitivity = 1e-1  ;                      /*!< Energy variance saturates when its log stops changing below this order of magnitude between sweeps. Good values are 1e-1 to 1e-4   */
        inline double             entropy_saturation_sensitivity  = 1e-3  ;                      /*!< Entanglement entropy saturates when it stops changing below this order of magnitude between sweeps. Good values are 1e-1 to 1e-4   */
        inline double             infocom_saturation_sensitivity  = 1e-2  ;                      /*!< Information center of mass saturates when it stops changing below this order of magnitude between sweeps. Good values are 1e-1 to 1e-4   */
        inline double             target_subspace_error           = 1e-10 ;                      /*!< The target subspace error 1-Σ|<ϕ_i|ψ>|². Eigenvectors are found until reaching this value. Measures whether the incomplete basis of eigenstates spans the current state. */
        inline size_t             max_subspace_size               = 256   ;                      /*!< Maximum number of candidate eigenstates to keep for a subspace optimization step */
        inline double             max_norm_error                  = 1e-10 ;                      /*!< Maximum norm deviation from unity during integrity checks */

        inline double   max_cache_gbts                  = 2.0   ;                  /*!< Maximum cache size (in GB) for temporary objects, e.g. density and transfer matrices used during subsystem entanglement entropy calculations. Increases the information lattice coverage. */
    }


    /*! \namespace settings::model Settings for the Hamiltonian spin-model */
    namespace model {
        inline ModelType    model_type = ModelType::ising_tf_rf;   /*!< Choice of model: {ising_tf_rf,ising_sdual, ising_majorana, lbit} */
        inline size_t       model_size = 16;                       /*!< Number of sites on the chain. Only relevant for finite algorithms: fDMRG and xDMRG */

        /*! \namespace settings::model::ising_tf_rf Settings for the Transverse-field Ising model with a random on-site field */
        namespace ising_tf_rf {
            inline double       J1         = 1;                 /*!< Ferromagnetic coupling for nearest neighbors.*/
            inline double       J2         = 1;                 /*!< Ferromagnetic coupling for next-nearest neighbors.*/
            inline double       h_tran     = 1;                 /*!< Transverse field strength */
            inline double       h_mean     = 0;                 /*!< Random field mean of distribution */
            inline double       h_wdth     = 0;                 /*!< Random field width of distribution */
            inline long         spin_dim   = 2;                 /*!< Spin dimension */
            inline std::string  distribution  = "uniform";      /*!< Random distribution for couplings and fields */
        }

        /*! \namespace settings::model::ising_sdual Settings for the Self-dual Ising model */
        namespace ising_sdual {
            inline double       lambda        = 0;              /*!< Lambda parameter related to next nearest neighbor coupling */
            inline double       delta         = 0;              /*!< Delta defined as log(J_mean) - log(h_mean). We get J_mean and h_mean by fixing max(J_mean,h_mean) = 1 */
            inline long         spin_dim      = 2;              /*!< Spin dimension */
            inline std::string  distribution  = "uniform";      /*!< Random distribution for couplings and fields */
        }

        /*! \namespace settings::model::ising_majorana Settings for the Ising-Majorana model */
        namespace ising_majorana {
            inline double       g             = 0;              /*!< Interaction parameter for nearest ZZ and next-nearest XX neighbor coupling */
            inline double       delta         = 0;              /*!< Delta defined as log(J_mean) - log(h_mean). We get J_mean and h_mean by fixing delta = 2lnW, W = J_wdth = 1/h_wdth */
            inline long         spin_dim      = 2;              /*!< Spin dimension */
            inline std::string  distribution  = "uniform";      /*!< Random distribution for couplings and fields */
        }

        /*! \namespace settings::model::lbit Settings for the l-bit Hamiltonian */
        namespace lbit {
            inline double      J1_mean       = 0;                                      /*!< Constant offset for on-site */
            inline double      J2_mean       = 0;                                      /*!< Constant offset for two-body interaction */
            inline double      J3_mean       = 0;                                      /*!< Constant offset for three-body interaction */
            inline double      J1_wdth       = 1.0;                                    /*!< Width of the distribution for on-site interactions */
            inline double      J2_wdth       = 1.0;                                    /*!< Width of the distribution for two-body interaction (st.dev. for normal distribution). */
            inline double      J3_wdth       = 1.0;                                    /*!< Width of the distribution for three-body interaction */
            inline size_t      J2_span       = -1ul;                                   /*!< Maximum allowed range for pairwise interactions, |i-j| <= J2_span. Use -1 for infinite. Note that J2_span + 1 MPOs are used */
            inline double      xi_Jcls       = 1.0;                                    /*!< The characteristic length-scale xi of the exponentially decaying interactions: J = exp(-|i-j|/xi_Jcls) * Random(i,j) */
            inline long        spin_dim      = 2;                                      /*!< Spin dimension */
            inline std::string distribution  = "normal";                               /*!< Random distribution for the interaction strengths in J2 */
            inline double      u_fmix        = 0.2;                                    /*!< Mixing factor for unitary transformation to real-space */
            inline size_t      u_depth       = 16;                                     /*!< Depth of the circuit of unitary 2-site gates which transform lbit <-> real spaces */
            inline double      u_lambda      = 1.0;                                    /*!< lambda parameter used in the Hermitian matrix g8mx for the unitary circuit gates: controls the size of the sz_i*sz_j terms */
            inline auto        u_wkind      = LbitCircuitGateWeightKind::EXPDECAY;     /*!< Weights w_i in the unitary 2-site gates. Choose [IDENTITY, EXPDECAY] for 1 or exp(-2|h[i] - h[i+1]|), h are onsite fields in the Hamiltonian */
            inline auto        u_mkind      = LbitCircuitGateMatrixKind::MATRIX_V3;    /*!< MATRIX_(V1|V2|V3) controls the kind of Hermitian matrix used in the unitary circuit gates */
        }

        /*! \namespace settings::model::xxz Settings for the XXZ model */
        namespace xxz {
            inline double       h_wdth        = 0;              /*!< Width of the distribution for on-site fields. If uniform: [-h_wdth, h_wdth] */
            inline double       delta         = 0;              /*!< Delta is the transverse ZZ coupling */
            inline long         spin_dim      = 2;              /*!< Spin dimension */
            inline std::string  distribution  = "uniform";      /*!< Random distribution for couplings and fields */
        }
    }

    /*! \namespace settings::idmrg Settings for the infinite DMRG algorithm */
    namespace idmrg {
        inline bool     on                  = false;                               /*!< Turns iDMRG simulation on/off. */
        inline size_t   iter_min            = 1;                                   /*!< Minimum number of iterations before being allowed to finish */
        inline size_t   iter_max            = 5000;                                /*!< Maximum number of iterations before forced termination */
        inline long     bond_max            = 32;                                  /*!< Bond dimension of the current position (maximum number of singular values to keep in SVD). */
        inline long     bond_min            = 16;                                  /*!< Initial bond dimension limit. Only used when bond_increase_when == true. */
        inline size_t   print_freq          = 1000;                                /*!< Print frequency for console output. In units of iterations.  (0 = off). */
    }


    /*! \namespace settings::itebd Settings for the imaginary-time infinite TEBD algorithm  */
    namespace itebd {
        inline bool      on                    = false;                            /*!< Turns iTEBD simulation on/off. */
        inline size_t    iter_min              = 1;                                /*!< Minimum number of iterations before being allowed to finish */
        inline size_t    iter_max              = 100000;                           /*!< Maximum number of iterations before forced termination */
        inline double    time_step_init_real   = 0.0;                              /*!< Real part of initial time step delta_t */
        inline double    time_step_init_imag   = 0.1;                              /*!< Imag part of initial time step delta_t */
        inline double    time_step_min         = 0.00001;                          /*!< (Absolute value) Minimum and final time step for iTEBD time evolution. */
        inline size_t    suzuki_order          = 1;                                /*!< Order of the suzuki trotter decomposition (1,2 or 4) */
        inline long      bond_max              = 8;                                /*!< Bond dimension of the current position (maximum number of singular values to keep in SVD). */
        inline long      bond_min              = 4;                                /*!< Initial bond dimension limit. Only used when bond_increase_when == true. */
        inline size_t    print_freq            = 5000;                             /*!< Print frequency for console output. In units of iterations. (0 = off).*/
    }

    /*! \namespace settings::fdmrg Settings for the finite DMRG algorithm */
    namespace fdmrg {
        inline bool      on                  = false;                              /*!< Turns fdmrg simulation on/off. */
        inline auto      ritz                = OptRitz::SR;                        /*!< Select what part of the energy eigenspectrum to target (SR:smallest real / LR: largest real) */
        inline size_t    iter_min            = 4;                                  /*!< Min number of iterations. One iterations moves L steps. */
        inline size_t    iter_max            = 10;                                 /*!< Max number of iterations. One iterations moves L steps. */
        inline long      bond_max            = 128;                                /*!< Bond dimension of the current position (maximum number of singular values to keep in SVD). */
        inline long      bond_min            = 8;                                  /*!< Initial bond dimension limit. Only used when bond_increase_when == true. */
        inline size_t    print_freq          = 100;                                /*!< Print frequency for console output. In units of iterations. (0 = off). */
        inline bool      store_wavefn        = false;                              /*!< Whether to store the wavefunction. Runs out of memory quick, recommended is false for max_length > 14 */
    }


    /*! \namespace settings::flbit Settings for the finite l-bit algorithm */
    namespace flbit {
        inline bool     on                     = false;                            /*!< Turns flbit simulation on/off. */
        inline bool     run_iter_in_parallel   = false;                            /*!< Time evolve each time step in parallel (because these are independent!) */
        inline bool     run_effective_model    = false;                            /*!< Runs the effecive model before the actual simulation */
        inline size_t   iter_min               = 4;                                /*!< Min number of iterations. One iterations moves L steps. */
        inline size_t   iter_max               = 10000;                            /*!< Max number of iterations. One iterations moves L steps. */
        inline bool     use_swap_gates         = true;                             /*!< Use gate swapping for pairwise long-range interactions rather then building a large multisite operator */
        inline bool     use_mpo_circuit        = false;                            /*!< Cast the unitary circuit to compressed mpo form (this is not generally faster or more accurate, but good for testing) */
        inline long     bond_max               = 1024;                             /*!< Maximum bond dimension (maximum number of singular values to keep in SVD). */
        inline long     bond_min               = 8;                                /*!< Minimum bond dimension */
        inline auto     time_scale             = TimeScale::LOGSPACED;             /*!< Choose linear or logarithmically spaced time points (LINEAR|LOG) */
        inline double   time_start_real        = 1e-1;                             /*!< Starting time point (real) */
        inline double   time_start_imag        = 0;                                /*!< Starting time point (imag) */
        inline double   time_final_real        = 1e6;                              /*!< Finishing time point (real) */
        inline double   time_final_imag        = 0;                                /*!< Finishing time point (imag) */
        inline size_t   time_num_steps         = 500;                              /*!< Number of steps from start to finish. Start and final times are included */
        inline size_t   print_freq             = 1;                                /*!< Print frequency for console output. In units of iterations. (0 = off). */
        inline bool     store_wavefn           = false;                            /*!< Whether to store the wavefunction. Runs out of memory quick, recommended is false for max_length > 14 */
        /*! \namespace settings::flbit::cls Settings for calculating the characteristic length-scale of lbits */
        namespace  cls {
            inline size_t   num_rnd_circuits          = 1;                         /*!< Calculate the characteristic length-scale for this many realizations of the unitary circuit */
            inline bool     exit_when_done            = false;                     /*!< If true, the program exits after calculating cls. Otherwise it starts the time evolution as usual */
            inline bool     randomize_hfields         = false;                     /*!< Randomize the on-site fields of the Hamiltonian that goes into each realization of the unitary circuits */
            inline size_t   mpo_circuit_switchdepth   = 10;                        /*!< Cast the unitary circuit to an approximate compressed MPO form when the circuit depth (u_depth) is this value or more    */
            inline long     mpo_circuit_svd_bondlim   = 128;                       /*!< The bond dimension limit used in the SVD when casting the circuit to compressed MPO form */
            inline double   mpo_circuit_svd_trnclim   = 1e-14;                     /*!< The truncation error limit used in the SVD when casting the circuit to compressed MPO form */
        }
        /*! \namespace settings::flbit::opdm Settings for calculating the averaged one-particle density matrix */
        namespace opdm {
            inline size_t num_rps                      = 0;                        /*!< Number of random product states (zero magnetization) to average over ( <=0 to disable) */
            inline bool   exit_when_done               = false;                    /*!< If true, the program exits after calculating the opdm. Otherwise it starts the time evolution as usual */
        }
}

    /*! \namespace settings::xdmrg Settings for the finite excited-state DMRG algorithm */
    namespace xdmrg {
        inline bool       on                            = false;                   /*!< Turns xDMRG simulation on/off. */
        inline OptAlgo    algo                          = OptAlgo::XDMRG;          /*!< Choose the type of DMRG algorithm [DMRG DMRGX, HYBRID_DMRGX, XDMRG, GDMRG]  */
        inline OptRitz    ritz                          = OptRitz::SM;             /*!< Select which eigenpair to target [LR (largest real), SR(largest real) SM(smallest magnitude) IS(initial state) TE(target energy density) CE(current energy)] */
        inline OptAlgo    algo_warmup                   = OptAlgo::XDMRG;          /*!< Choose the type of DMRG algorithm [DMRG DMRGX, HYBRID_DMRGX, XDMRG, GDMRG]  */
        inline OptRitz    ritz_warmup                   = OptRitz::SM;             /*!< Select which eigenpair to target [LR (largest real), SR(largest real) SM(smallest magnitude) IS(initial state) TE(target energy density) CE(current energy)] */
        inline OptAlgo    algo_stuck                    = OptAlgo::GDMRG;          /*!< Choose the type of DMRG algorithm [DMRG DMRGX, HYBRID_DMRGX, XDMRG, GDMRG]  */
        inline OptRitz    ritz_stuck                    = OptRitz::LM;             /*!< Select which eigenpair to target [LR (largest real), SR(largest real) SM(smallest magnitude) IS(initial state) TE(target energy density) CE(current energy)] */
        inline double     energy_spectrum_shift         = 1e-10 ;                  /*!< (Used with ritz == OptRitz::SM) Shift the energy eigenvalue spectrum by this amount: H -> H - shift   */
        inline double     energy_density_target         = 0.5;                     /*!< (Used with ritz == OptRitz::TE) Target energy in [0-1], Target energy = energy_density_target * (EMAX+EMIN) + EMIN. */
        inline size_t     iter_min                      = 4;                       /*!< Min number of iterations. One iterations moves L steps. */
        inline size_t     iter_max                      = 50;                      /*!< Max number of iterations. One iterations moves L steps. */
        inline long       bond_max                      = 1024;                    /*!< Maximum bond dimension (number of singular values to keep after SVD). */
        inline long       bond_min                      = 8;                       /*!< Minimum bond dimension. Used at the start, during warmup or when bond_increase_when == true, or when starting from an entangled state */
        inline size_t     print_freq                    = 1;                       /*!< Print frequency for console output. In units of iterations. (0 = off). */
        inline size_t     max_states                    = 1;                       /*!< Max number of random states to find using xDMRG on a single disorder realization */
        inline bool       store_wavefn                  = false;                   /*!< Whether to store the wavefunction. Runs out of memory quick, recommended is false for max_length > 14 */
    }
}
/* clang-format on */
