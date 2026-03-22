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
     *   num_threads controls the Eigen::Tensor/std::thread worker pool.
     *   OpenMP and BLAS/LAPACK threading are configured separately by the runtime environment.
     */
    namespace threading{
        /*! Number of Eigen::Tensor worker threads.
         *  Values 0 or 1 effectively disable extra c++ worker threads.
         * */
        inline unsigned int num_threads = 1;                                       /*!< Number of Eigen::Tensor worker threads */
        inline unsigned int max_threads = std::thread::hardware_concurrency();     /*!< Hardware concurrency reported on this machine */
        inline unsigned int show_threads = false;                                  /*!< Show threading information and exit without running a simulation */

//        inline int omp_threads = 1;                                              /*!< Number of threads for openmp threads used in blas/lapack and Eigen. num_threads <= 0 will try to use as many as possible */
//        inline int stl_threads = 1;                                              /*!< Number of threads for c++11 threading. Used in Eigen::Tensor. stl_threads <= 0 will try to use as many as possible */
    }

    /*!  \namespace settings::input Settings for initialization */
    namespace input{
        inline long        seed                                 = 1;                            /*!< Main seed for the random number generator. */
        inline std::string config_filename                      = "input/input.cfg";            /*!< Default config filename. Can either be a .cfg file or a .h5 file with a config stored as a string in /common/config_file_contents */
        inline std::string config_file_contents;                                               /*!< Copy of the loaded config file, stored for internal use and HDF5 output */
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


    /*! \namespace settings::model Settings for the Hamiltonian spin-model */
    namespace model {
        inline ModelType    model_type = ModelType::ising_tf_rf;   /*!< Choice of model: {ising_tf_rf, ising_sdual, ising_majorana, lbit, xxz} */
        inline size_t       model_size = 16;                       /*!< Number of sites on the chain. Relevant for finite algorithms such as fDMRG, xDMRG, and fLBIT */

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
            inline std::string  distribution  = "uniform";      /*!< Random distribution for couplings and fields (currently uniform only) */
        }

        /*! \namespace settings::model::lbit Settings for the l-bit Hamiltonian */
        namespace lbit {
            inline double      J1_mean       = 0;                                      /*!< Constant offset for onsite terms */
            inline double      J2_mean       = 0;                                      /*!< Constant offset for nearest-neighbor interactions */
            inline double      J3_mean       = 0;                                      /*!< Constant offset for next-nearest-neighbor interactions */
            inline double      J1_wdth       = 1.0;                                    /*!< Distribution width for onsite terms */
            inline double      J2_wdth       = 1.0;                                    /*!< Distribution width for nearest-neighbor interactions (st.dev. for normal distribution) */
            inline double      J3_wdth       = 1.0;                                    /*!< Distribution width for next-nearest-neighbor interactions */
            inline size_t      J2_span       = -1ul;                                   /*!< Maximum allowed range for pairwise interactions, |i-j| <= J2_span. Use -1 for infinite. Note that J2_span + 1 MPOs are used */
            inline double      xi_Jcls       = 1.0;                                    /*!< The characteristic length-scale xi of the exponentially decaying interactions: J = exp(-|i-j|/xi_Jcls) * Random(i,j) */
            inline long        spin_dim      = 2;                                      /*!< Spin dimension */
            inline std::string distribution  = "normal";                               /*!< Random distribution used for the l-bit couplings */
            inline double      u_fmix        = 0.2;                                    /*!< Overall gate amplitude factor f in the unitary circuit, U = exp(-i f w M) */
            inline size_t      u_depth       = 16;                                     /*!< Number of layers of 2-site gates in the unitary circuit that maps between l-bit and real space */
            inline double      u_lambda      = 1.0;                                    /*!< Lambda parameter in the Hermitian gate matrix M_i, controlling the sz_i sz_j contribution */
            inline auto        u_wkind      = LbitCircuitGateWeightKind::EXPDECAY;     /*!< Rule for the gate weights w_i in the unitary circuit [IDENTITY, EXPDECAY] */
            inline auto        u_mkind      = LbitCircuitGateMatrixKind::MATRIX_V3;    /*!< Choice of Hermitian matrix ansatz M_i used in the unitary circuit gates */
        }

        /*! \namespace settings::model::xxz Settings for the XXZ model */
        namespace xxz {
            inline double       h_wdth        = 0;              /*!< Width of the distribution for on-site fields. If uniform: [-h_wdth, h_wdth] */
            inline double       delta         = 0;              /*!< ZZ anisotropy Delta in the XXZ Hamiltonian */
            inline long         spin_dim      = 2;              /*!< Spin dimension */
            inline std::string  distribution  = "uniform";      /*!< Random distribution for on-site fields (currently uniform only) */
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
        inline auto      ritz                = OptRitz::SR;                        /*!< Which extremal eigenstate to target: SR = ground state, LR = highest-energy state */
        inline size_t    iter_min            = 4;                                  /*!< Min number of iterations. One iterations moves L steps. */
        inline size_t    iter_max            = 10;                                 /*!< Max number of iterations. One iterations moves L steps. */
        inline long      bond_max            = 128;                                /*!< Bond dimension of the current position (maximum number of singular values to keep in SVD). */
        inline long      bond_min            = 8;                                  /*!< Initial bond dimension limit. Only used when bond_increase_when == true. */
        inline size_t    print_freq          = 100;                                /*!< Print frequency for console output. In units of iterations. (0 = off). */
        inline bool      store_wavefn        = false;                              /*!< Whether to store the full wavefunction/MPS. This can become very large on bigger systems */
    }


    /*! \namespace settings::flbit Settings for the finite l-bit algorithm */
    namespace flbit {
        inline bool     on                     = false;                            /*!< Turns flbit simulation on/off. */
        inline bool     run_iter_in_parallel   = false;                            /*!< Time evolve independent target time points in parallel */
        inline bool     run_effective_model    = false;                            /*!< Also run the effective diagonal l-bit model for comparison before the full simulation */
        inline size_t   iter_min               = 4;                                /*!< Min number of iterations. One iterations moves L steps. */
        inline size_t   iter_max               = 10000;                            /*!< Max number of iterations. One iterations moves L steps. */
        inline bool     use_swap_gates         = true;                             /*!< Use gate swapping for pairwise long-range interactions rather then building a large multisite operator */
        inline bool     use_mpo_circuit        = false;                            /*!< Cast the unitary circuit to compressed mpo form (this is not generally faster or more accurate, but good for testing) */
        inline long     bond_max               = 1024;                             /*!< Maximum bond dimension (maximum number of singular values to keep in SVD). */
        inline long     bond_min               = 8;                                /*!< Minimum bond dimension */
        inline auto     time_scale             = TimeScale::LOGSPACED;             /*!< Spacing of the target time points [LINSPACED | LOGSPACED] */
        inline double   time_start_real        = 1e-1;                             /*!< Starting time point (real) */
        inline double   time_start_imag        = 0;                                /*!< Starting time point (imag) */
        inline double   time_final_real        = 1e6;                              /*!< Finishing time point (real) */
        inline double   time_final_imag        = 0;                                /*!< Finishing time point (imag) */
        inline size_t   time_num_steps         = 500;                              /*!< Number of steps from start to finish. Start and final times are included */
        inline size_t   print_freq             = 1;                                /*!< Print frequency for console output. In units of iterations. (0 = off). */
        inline bool     store_wavefn           = false;                            /*!< Whether to store the full wavefunction/MPS. This can become very large on bigger systems */
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
            inline size_t num_rps                      = 0;                        /*!< Number of random product states (zero magnetization) to average over. Set 0 to disable */
            inline bool   exit_when_done               = false;                    /*!< If true, the program exits after calculating the opdm. Otherwise it starts the time evolution as usual */
        }
}

    /*! \namespace settings::xdmrg Settings for the finite excited-state DMRG algorithm */
    namespace xdmrg {
        inline bool       on                            = false;                   /*!< Turns xDMRG simulation on/off. */
        inline OptAlgo    algo                          = OptAlgo::XDMRG;          /*!< Choose the type of DMRG algorithm [DMRG DMRGX, HYBRID_DMRGX, XDMRG, GDMRG]  */
        inline OptRitz    ritz                          = OptRitz::SM;             /*!< Which eigenpair to target [LR largest real, SR smallest real, LM largest magnitude, SM smallest magnitude, IS initial-state energy, TE target energy density] */
        inline OptAlgo    algo_warmup                   = OptAlgo::XDMRG;          /*!< Choose the type of DMRG algorithm [DMRG DMRGX, HYBRID_DMRGX, XDMRG, GDMRG]  */
        inline OptRitz    ritz_warmup                   = OptRitz::SM;             /*!< Which eigenpair to target during warmup [LR, SR, LM, SM, IS, TE] */
        inline OptAlgo    algo_stuck                    = OptAlgo::GDMRG;          /*!< Choose the type of DMRG algorithm [DMRG DMRGX, HYBRID_DMRGX, XDMRG, GDMRG]  */
        inline OptRitz    ritz_stuck                    = OptRitz::LM;             /*!< Which eigenpair to target after switching to the stuck-policy algorithm [LR, SR, LM, SM, IS, TE] */
        inline double     energy_spectrum_shift         = 0.0;                     /*!< (Used with ritz == OptRitz::SM) Shift the energy eigenvalue spectrum by this amount: H -> H - shift   */
        inline double     energy_density_target         = 0.5;                     /*!< (Used with ritz == OptRitz::TE) Target energy density in [0,1], mapped as EMIN + t * (EMAX - EMIN) */
        inline size_t     iter_min                      = 4;                       /*!< Min number of iterations. One iterations moves L steps. */
        inline size_t     iter_max                      = 50;                      /*!< Max number of iterations. One iterations moves L steps. */
        inline long       bond_max                      = 1024;                    /*!< Maximum bond dimension (number of singular values to keep after SVD). */
        inline long       bond_min                      = 8;                       /*!< Minimum bond dimension. Used at the start, during warmup or when bond_increase_when == true, or when starting from an entangled state */
        inline size_t     print_freq                    = 1;                       /*!< Print frequency for console output. In units of iterations. (0 = off). */
        inline size_t     max_states                    = 1;                       /*!< Max number of random states to find using xDMRG on a single disorder realization */
        inline bool       store_wavefn                  = false;                   /*!< Whether to store the full wavefunction/MPS. This can become very large on bigger systems */
    }




    /*! \namespace settings::strategy Settings affecting the convergence rate of the algorithms */
    namespace strategy {
        inline bool                move_sites_when_stuck       = true;                                   /*!< \deprecated Currently unused. Kept for config compatibility. */
        inline ProjectionPolicy    projection_policy           = ProjectionPolicy::DEFAULT;              /*!< Bitmask controlling when to project to the spin/parity sector requested by target_axis. DEFAULT = INIT | STUCK | CONVERGED */
        inline bool                use_eigenspinors            = false;                                  /*!< Use random pauli-matrix eigenvectors when initializing each mps site along x,y or z  */
        inline size_t              iter_max_warmup             = 4;                                      /*!< Initial warmup iterations. In DMRG these iterations use an exact solver with reduced bond dimension */
        inline size_t              iter_max_stuck              = 5;                                      /*!< Stop after this many consecutive stuck iterations once the bond and truncation limits have saturated */
        inline size_t              iter_max_saturated          = 5;                                      /*!< If any monitored quantity stays saturated this long, count the algorithm as saturated */
        inline size_t              iter_min_converged          = 1;                                      /*!< Require convergence at least this many iterations before success */
        inline BlockSizePolicy     dmrg_blocksize_policy       = BlockSizePolicy::MIN;                   /*!< Bitmask controlling the adaptive DMRG block size, combining size-selection flags, activation conditions, and ON_UPDATE/ON_BONDEXP */
        inline size_t              dmrg_min_blocksize          = 1;                                      /*!< Minimum number of sites in a dmrg optimization step. */
        inline size_t              dmrg_max_blocksize          = 4;                                      /*!< Maximum number of sites in a dmrg optimization step. */
        inline long                dmrg_max_prob_size          = 1024*2*1024;                            /*!< Restricts the dmrg blocksize to keep the problem size below this limit. Problem size = chiL * (spindim ** blocksize) * chiR */
        inline BondExpansionPolicy dmrg_bond_expansion_policy  = BondExpansionPolicy::DEFAULT;           /*!< Bitmask selecting the bond-expansion strategy, timing, and H1/H2 enrichments used during DMRG */
        namespace dmrg_bond_expansion {
            namespace dmrg3s {
                inline double               maxalpha            = 1e-2;                                   /*!< Upper limit for mixing factors derived from the local residual norms */
                inline double               minalpha            = 1e-15;                                  /*!< Lower limit for mixing factors derived from the local residual norms */
            }
            namespace preopt {
                inline float               bond_factor         = 1.05f;                                  /*!< Expand the bond dimension by this factor above the current bond dimension limit (value  <= 1.0 = disabled) */
                inline size_t              maxiter             = 1;                                      /*!< How many Lanczos iterations to use in the nsite bond expansion */
                inline size_t              nkrylov             = 3;                                      /*!< The krylov subspace size to use in nsite bond expansion */
            }
        }
        inline std::string         target_axis                 = "none";                                 /*!< Requested target spin/parity sector. Choose {none, +/-x, +/-y, +/-z} */
        inline std::string         initial_axis                = "none";                                 /*!< Axis used to build the initial state. Choose {none, +/-x, +/-y, +/-z} */
        inline StateInitType       initial_type                = StateInitType::REAL;                    /*!< Whether the initial-state amplitudes are real or complex */
        inline StateInit           initial_state               = StateInit::RANDOM_ENTANGLED_STATE;      /*!< Initialization mode for the finite-system starting state */
        inline std::string         initial_pattern             = {};                                     /*!< Product-state pattern supplied by the user or generated internally, and stored for reuse/resume */
        inline double              rbds_rate                   = 0.5;                                    /*!< If rbds_rate > 0, runs reverse bond dimension scaling (rbds) after the main algorithm. Values [0,1] represent the shrink factor, while [1,infty] represents a shrink step */
        inline double              rtes_rate                   = 1e1;                                    /*!< If rtes_rate > 1, runs reverse truncation error scaling (rtes) after the main algorithm. Values [1, infty] represent the growth factor for the truncation error limit */
        inline UpdatePolicy        bond_increase_when          = UpdatePolicy::NEVER;                    /*!< If and when to increase the bond dimension limit {NEVER, WARMUP, HALFSWEEP, FULLSWEEP, TRUNCATED, SAT_EVAR, SAT_ALGO, STK_ALGO}. */
        inline double              bond_increase_rate          = 8;                                      /*!< Bond dimension growth rate. Factor if 1<x<=2, constant shift if x > 2, otherwise invalid. */
        inline UpdatePolicy        trnc_decrease_when          = UpdatePolicy::NEVER;                    /*!< If and when to decrease SVD truncation error limit {NEVER, WARMUP, HALFSWEEP, FULLSWEEP, TRUNCATED, SAT_EVAR, SAT_ALGO, STK_ALGO} */
        inline double              trnc_decrease_rate          = 1e-1;                                   /*!< Decrease SVD truncation error limit by this factor. Valid if 0 < x < 1 */
        inline double              trnc_increase_rtol          = 1e-3;                                   /*!< Relative energy/variance drift tolerated when temporarily increasing the SVD truncation limit to compress the MPS. Nonpositive disables this */
        inline size_t              trnc_increase_iter          = 3;                                      /*!< How often (in iters) to attempt compressing the MPS (by increasing the truncation error limit) */
        inline UpdatePolicy        etol_decrease_when          = UpdatePolicy::NEVER;                    /*!< If and when to decrease EIGS tolerance {NEVER, WARMUP, HALFSWEEP, FULLSWEEP, TRUNCATED, SAT_EVAR, SAT_ALGO, STK_ALGO, DYNAMIC} */
        inline double              etol_decrease_rate          = 1e-1;                                   /*!< Decrease EIGS tolerance by this factor. Valid if 0 < x < 1 */
    }


    /*! \namespace settings::precision Settings for the convergence threshold and precision of MPS, SVD and eigensolvers */
    namespace precision {
        inline ScalarType         algoScalar                      = ScalarType::FP64;       /*!< Scalar type for tensor storage (state, model, edges)  */
        inline ScalarType         optScalar                       = ScalarType::FP64;       /*!< Scalar type for computations (eig, eigs, svd) */
        inline long               eig_max_size                    = 4096  ;                 /*!< Maximum problem size before switching from eig to eigs. */
        inline long               eigs_max_size_shift_invert      = 4096  ;                 /*!< Maximum problem size allowed for shift-invert of the local (effective) hamiltonian matrix. */
        inline size_t             eigs_iter_min                   = 1000;                   /*!< Minimum number of iterations for eigenvalue solver. */
        inline size_t             eigs_iter_max                   = 100000;                 /*!< Maximum number of iterations for eigenvalue solver. */
        inline double             eigs_iter_gain                  = 2     ;                 /*!< Increase number of iterations on OptSolver::EIGS by gain^(iters without progress) */
        inline GainPolicy         eigs_iter_gain_policy           = GainPolicy::SAT_ALGO;   /*!< Bitmask for when to increase the EIGS iteration budget [NEVER, HALFSWEEP, FULLSWEEP, SAT_EVAR, SAT_ALGO, STK_ALGO, FIN_BOND, FIN_TRNC] */
        inline double             eigs_abstol_min                 = 1e-14 ;                 /*!< Smallest absolute tolerance allowed for the iterative eigensolver */
        inline double             eigs_abstol_max                 = 1e-8  ;                 /*!< Largest absolute tolerance allowed for the iterative eigensolver */
        inline double             eigs_reltol_min                 = 1e-2  ;                 /*!< Smallest relative residual-reduction target for the iterative eigensolver. Set 0 to disable this criterion */
        inline double             eigs_reltol_max                 = 1e-1  ;                 /*!< Largest relative residual-reduction target for the iterative eigensolver. Set 0 to disable this criterion */
        inline int                eigs_ncv_min                    = 0     ;                 /*!< Minimum krylov subspace size in the eigensolver. Set ncv <= 0 for automatic selection */
        inline int                eigs_ncv_max                    = 0     ;                 /*!< Maximum krylov subspace size in the eigensolver. Set ncv <= 0 for automatic selection */
        inline int                eigs_nev_min                    = 1     ;                 /*!< The minimum number of eigenpairs to request on OptSolver::EIGS */
        inline int                eigs_nev_max                    = 8     ;                 /*!< The maximum number of eigenpairs to request on OptSolver::EIGS when stuck (increases slowly) (ncv may have to increase accordingly) */
        inline long               eigs_blk_min                    = 1;                      /*!< Minimum block size in the eigenvalue solver (ritz vectors and residuals in block size chunks) */
        inline long               eigs_blk_max                    = 2;                      /*!< Maximum block size in the eigenvalue solver (ritz vectors and residuals in block size chunks) */
        inline long               eigs_jcb_blocksize_min          = 128   ;                 /*!< Minimum block size used in the block-jacobi preconditioner */
        inline long               eigs_jcb_blocksize_max          = 256   ;                 /*!< Maximum block size used in the block-jacobi preconditioner (increases up to max when stuck) */
        inline long               eigs_jcb_overlap_size           = 32    ;                 /*!< Jacobi block overlap size */
        inline double             svd_truncation_min              = 1e-14 ;                 /*!< Truncation error limit, i.e. discard singular values while the truncation error is lower than this */
        inline double             svd_truncation_max              = 1e-6  ;                 /*!< If truncation error limit is updated (trnc_decrease_when != NEVER), start from this value */
        inline size_t             svd_switchsize_bdc              = 16    ;                 /*!< Linear size of a matrix, below which SVD will use slower but more precise JacobiSVD instead of BDC (default is 16 , good could be ~64) */
        inline bool               svd_save_fail                   = false ;                 /*!< Save failed SVD calculations to file */

        inline auto               use_compressed_mpo                  = MpoCompress::DPL;   /*!< Select the compression scheme for the virtual bond dimensions of H  mpos. Select {NONE, SVD (high compression), DPL (high precision)} */
        inline auto               use_compressed_mpo_squared          = MpoCompress::DPL;   /*!< Select the compression scheme for the virtual bond dimensions of H² mpos. Select {NONE, SVD (high compression), DPL (high precision)} */
        inline bool               use_energy_shifted_mpo              = false ;             /*!< Shift H by the current energy before building H-derived objectives, reducing catastrophic cancellation in H^2-based calculations */
        inline bool               use_parity_shifted_mpo              = true  ;             /*!< Add a parity-shift term to H so opposite spin/parity sectors do not mix near degeneracies */
        inline bool               use_parity_shifted_mpo_squared      = true  ;             /*!< Add the corresponding parity-shift term to H^2 for folded-spectrum objectives */
        inline double             variance_convergence_threshold      = 1e-13 ;             /*!< Desired precision on total energy variance. The MPS state is considered good enough when its energy variance reaches below this value */
        inline double             variance_saturation_sensitivity     = 1e-1  ;             /*!< Energy variance saturates when its log stops changing below this order of magnitude between sweeps. Good values are 1e-1 to 1e-4   */
        inline double             energy_saturation_sensitivity       = 1e-1  ;             /*!< Energy saturates when it stops changing below this order of magnitude between sweeps. Good values are 1e-2 to 1e-8   */
        inline double             entanglement_saturation_sensitivity = 1e-3  ;             /*!< Entanglement entropy saturates when it stops changing below this order of magnitude between sweeps. Good values are 1e-1 to 1e-4   */
        inline double             locinfoscale_saturation_sensitivity = 1e-2  ;             /*!< Information center of mass saturates when it stops changing below this order of magnitude between sweeps. Good values are 1e-1 to 1e-4   */
        inline double             target_subspace_error               = 1e-10 ;             /*!< The target subspace error 1-Σ|<ϕ_i|ψ>|². Eigenvectors are found until reaching this value. Measures whether the incomplete basis of eigenstates spans the current state. */
        inline size_t             max_subspace_size                   = 256   ;             /*!< Maximum number of candidate eigenstates to keep for a subspace optimization step */
        inline long               max_norm_slack                      = 1000l;              /*!< Permit norm errors within a tolerance = machine_epsilon * slack */

        inline double   max_cache_gbts                  = 2.0   ;                  /*!< Maximum cache size (in GB) for temporary objects, e.g. density and transfer matrices used during subsystem entanglement entropy calculations. Increases the information lattice coverage. */
    }


    /*!  \namespace settings::storage Settings for output-file generation
     *
     * Storage is controlled per object with StoragePolicy bitflags. A save is triggered when the current
     * StorageEvent matches any trigger flag in the policy; modifier flags then refine how that write is kept.
     *
     * Available StoragePolicy flags:
     *       - `NONE`:    never store
     *       - `INIT`:    store during initialization or preprocessing
     *       - `ITER`:    store on iteration events (see storage_interval)
     *       - `EMIN`:    store after finding the minimum-energy state
     *       - `EMAX`:    store after finding the maximum-energy state
     *       - `PROJ`:    store after projection steps
     *       - `BOND`:    store after bond-dimension updates
     *       - `TRNC`:    store after truncation-limit updates
     *       - `FAILURE`: store only on unsuccessful termination
     *       - `SUCCESS`: store only on successful termination
     *       - `FINISH`:  store when the algorithm finishes, regardless of success
     *       - `ALWAYS`:  store whenever a save opportunity is encountered
     *       - `REPLACE`: keep only the latest matching entry when possible
     *       - `RBDS`:    store reverse-bond-dimension-scaling follow-up steps
     *       - `RTES`:    store reverse-truncation-error-scaling follow-up steps
     *
     * Example bitflag combinations:
     *       - `ITER | REPLACE`: keep only the latest per-iteration snapshot
     *       - `ITER | FINISH | REPLACE`: keep the latest iterative snapshot and the final one
     *       - `FINISH | RBDS | RTES`: store the final state and the rbds/rtes follow-up steps
     *       - `ITER | BOND | TRNC | RBDS | RTES`: store iterations, bond/truncation updates, and rbds/rtes steps
     *
     * **Note: Resume**
     * Finite-state resume requires a fully stored MPS under the selected state prefix. fLBIT can also resume from its time-evolution data.
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
        inline ResumePolicy        resume_policy                   = ResumePolicy::IF_UNSUCCESSFUL; /*!< Which exit conditions from a previous run qualify a state for resume */
        inline FileCollisionPolicy file_collision_policy           = FileCollisionPolicy::RESUME;   /*!< What to do when a prior output file is found. Choose between RESUME, REVIVE, BACKUP, RENAME, REPLACE */
        inline FileResumePolicy    file_resume_policy              = FileResumePolicy::FULL;        /*!< What to do when common/finished_all is true: FULL keeps scanning the file/config, FAST exits immediately */
        inline std::string         file_resume_name                = ""  ;                          /*!< On file_collision_policy=RESUME|REVIVE: resume from state candidate matching this string. Empty implies any */
        inline size_t              file_resume_iter                = -1ul;                          /*!< On file_collision_policy=RESUME|REVIVE: which iteration to resume from. -1ul implies resume from last available iteration */

        namespace mps::state_emid{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the xDMRG mid-spectrum MPS state */
        }
        namespace mps::state_emin{
            inline StoragePolicy policy = StoragePolicy::FINISH;                                   /*!< Storage policy for the minimum-energy MPS state from fDMRG/xDMRG */
        }
        namespace mps::state_emax{
            inline StoragePolicy policy = StoragePolicy::FINISH;                                   /*!< Storage policy for the maximum-energy MPS state from fDMRG/xDMRG */
        }
        namespace mps::state_real{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the fLBIT state in the real-space basis */
        }
        namespace mps::state_lbit{
            inline StoragePolicy policy = StoragePolicy::NONE;                                     /*!< Storage policy for the fLBIT state in the l-bit basis */
        }
        namespace mpo::model{
            inline StoragePolicy policy = StoragePolicy::NONE;                                     /*!< Storage policy for the Hamiltonian MPO */
        }
        namespace table::bonds {
            inline StoragePolicy policy = StoragePolicy::FINISH;                                   /*!< Storage policy for the bond table */
        }
        namespace table::model{
            inline StoragePolicy policy = StoragePolicy::INIT;                                     /*!< Storage policy for the model-parameter table */
        }
        namespace table::measurements{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the measurements table */
        }
        namespace table::status{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the algorithm-status table */
        }
        namespace table::memory{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the memory-usage table */
        }
        namespace table::timers{
            inline tid::level  level    = tid::level::normal;                      /*!< Highest timer detail level to include in the timers table */
            inline StoragePolicy policy = StoragePolicy::FINISH;                   /*!< Storage policy for the timers table */
        }
        namespace table::entanglement_entropies{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the entanglement-entropy table */
        }
        namespace table::truncation_errors{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the truncation-error table */
        }
        namespace table::bond_dimensions{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the bond-dimension table */
        }
        namespace table::number_entropies{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the number-entropy table */
        }
        namespace table::renyi_entropies{
            inline StoragePolicy policy = StoragePolicy::ITER;                                     /*!< Storage policy for the Renyi-entropy table */
        }
        namespace table::opdm_spectrum{
            inline StoragePolicy policy = StoragePolicy::FINISH;                                   /*!< Storage policy for the OPDM-spectrum table */
        }
        namespace table::information_per_scale{
            inline StoragePolicy policy = StoragePolicy::FINISH;                                   /*!< Storage policy for the information-per-scale table */
        }
        namespace table::information_center_of_mass{
            inline StoragePolicy policy = StoragePolicy::FINISH;                                   /*!< Storage policy for the information-center-of-mass table */
        }
        namespace table::expectation_values_spin_xyz{
            inline StoragePolicy policy = StoragePolicy::FINISH;                                   /*!< Storage policy for the spin-expectation-value table */
        }
        namespace table::random_unitary_circuit{
            inline StoragePolicy policy = StoragePolicy::INIT;                                     /*!< Storage policy for the random unitary circuit table */
        }
        namespace dataset::lbit_analysis{
            inline StoragePolicy policy = StoragePolicy::INIT;                                     /*!< Storage policy for the l-bit analysis dataset */
        }
        namespace dataset::subsystem_entanglement_entropies{
            inline StoragePolicy policy = StoragePolicy::FINISH;                   /*!< Storage policy for the subsystem-entanglement dataset: entanglement entropy (log2) of all contiguous subsystems */
            inline unsigned long chunksize = 10;                                   /*!< Chunk depth for appending subsystem-entanglement records */
            inline auto bits_err = 1e-8;                                           /*!< Positive: tolerate a relative bit deficit 1 - bits_found/L. Negative: tolerate an absolute deficit L - bits_found */
            inline long eig_size = 4096l;                                          /*!< Largest reduced-density-matrix size to diagonalize exactly */
            inline long bond_lim = 2048l;                                          /*!< Bond-dimension limit used during swap-based evaluations */
            inline auto trnc_lim = 1e-8;                                           /*!< Truncation-error limit used during swap-based evaluations */
            inline auto precision = Precision::DOUBLE;                             /*!< Internal floating-point precision for this dataset calculation */
        }
        namespace dataset::information_lattice{
            inline StoragePolicy policy = StoragePolicy::FINISH;                   /*!< Storage policy for the information-lattice dataset built from subsystem_entanglement_entropies */
            inline unsigned long chunksize = 10;                                   /*!< Chunk depth for appending information-lattice records */
        }
        namespace dataset::opdm{
            inline StoragePolicy policy = StoragePolicy::FINISH;                   /*!< Storage policy for the OPDM dataset (one-particle density matrix) */
            inline unsigned long chunksize = 10;                                   /*!< Chunk depth for appending OPDM records */
        }
        namespace dataset::number_probabilities{
            inline StoragePolicy policy = StoragePolicy::FINISH;                   /*!< Storage policy for the number-probability dataset: probability of measuring n particles to the left of site i, for all n and i */
            inline unsigned long chunksize = 10;                                   /*!< Chunk depth for appending number-probability records */
        }
        namespace dataset::expectation_values_spin_xyz{
            inline StoragePolicy policy = StoragePolicy::ITER;                     /*!< Storage policy for the spin-expectation-value dataset */
            inline unsigned long chunksize = 10;                                   /*!< Chunk depth for appending spin-expectation records */
        }
        namespace dataset::correlation_matrix_spin_xyz{
            inline StoragePolicy policy = StoragePolicy::ITER;                     /*!< Storage policy for the spin-correlation-matrix dataset */
            inline unsigned long chunksize = 10;                                   /*!< Chunk depth for appending spin-correlation records */
        }
        namespace tmp{
            inline std::string hdf5_temp_path;                                     /*!< Active temporary HDF5 path when writing through temp_dir */
            inline std::string hdf5_final_path;                                    /*!< Final destination HDF5 path */
        }
    }



}
/* clang-format on */
