#include "contraction_cutensor.h"

#include "config/cuda.h"
#include "debug/exceptions.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <mutex>
#include <numeric>

#if defined(DMRG_ENABLE_CUTENSOR)
    #include <cuda_runtime_api.h>
    #include <cutensor.h>
#endif

namespace tools::common::contraction::internal {
    namespace {
#if defined(DMRG_ENABLE_CUTENSOR)
        // The cache is owned per scalar type. That keeps the public matvec path
        // simple while still letting repeated Krylov applications reuse the same
        // cuTENSOR plans and device buffers for one local tensor shape.
        template<typename Scalar>
        struct Traits;

        template<>
        struct Traits<fp32> {
            static constexpr auto data_type = CUTENSOR_R_32F;
            static auto           compute_desc() { return CUTENSOR_COMPUTE_DESC_32F; }
            static constexpr fp32 one() noexcept { return 1.0f; }
            static constexpr fp32 zero() noexcept { return 0.0f; }
        };

        template<>
        struct Traits<fp64> {
            static constexpr auto data_type = CUTENSOR_R_64F;
            static auto           compute_desc() { return CUTENSOR_COMPUTE_DESC_64F; }
            static constexpr fp64 one() noexcept { return 1.0; }
            static constexpr fp64 zero() noexcept { return 0.0; }
        };

        template<>
        struct Traits<cx32> {
            static constexpr auto data_type = CUTENSOR_C_32F;
            static auto           compute_desc() { return CUTENSOR_COMPUTE_DESC_32F; }
            static constexpr cx32 one() noexcept { return {1.0f, 0.0f}; }
            static constexpr cx32 zero() noexcept { return {0.0f, 0.0f}; }
        };

        template<>
        struct Traits<cx64> {
            static constexpr auto data_type = CUTENSOR_C_64F;
            static auto           compute_desc() { return CUTENSOR_COMPUTE_DESC_64F; }
            static constexpr cx64 one() noexcept { return {1.0, 0.0}; }
            static constexpr cx64 zero() noexcept { return {0.0, 0.0}; }
        };

        inline void check_cuda(cudaError_t err, const char *expr) {
            if(err != cudaSuccess) throw except::runtime_error("{} failed: {}", expr, cudaGetErrorString(err));
        }

        inline void check_cutensor(cutensorStatus_t err, const char *expr) {
            if(err != CUTENSOR_STATUS_SUCCESS) throw except::runtime_error("{} failed: {}", expr, cutensorGetErrorString(err));
        }

        template<std::size_t Rank>
        auto to_i64(const std::array<long, Rank> &dims) {
            std::array<int64_t, Rank> out{};
            for(std::size_t idx = 0; idx < Rank; ++idx) out[idx] = static_cast<int64_t>(dims[idx]);
            return out;
        }

        template<std::size_t Rank>
        auto to_strides(const std::array<long, Rank> &dims) {
            std::array<int64_t, Rank> strides{};
            int64_t                   stride = 1;
            for(std::size_t idx = 0; idx < Rank; ++idx) {
                strides[idx] = stride;
                stride *= static_cast<int64_t>(dims[idx]);
            }
            return strides;
        }

        template<std::size_t Rank>
        std::size_t size_bytes(const std::array<long, Rank> &dims, std::size_t elem_size) {
            return static_cast<std::size_t>(std::accumulate(dims.begin(), dims.end(), 1l, std::multiplies<long>{})) * elem_size;
        }

        template<typename Scalar>
        struct Cache {
            std::mutex          mutex;
            bool                signature_ready = false;
            bool                buffers_ready   = false;
            bool                handle_ready    = false;
            bool                plan1_ready     = false;
            bool                plan2_ready     = false;
            bool                plan3_ready     = false;
            int                 device          = -1;
            std::array<long, 3> mps_dims        = {0, 0, 0};
            std::array<long, 4> mpo_dims        = {0, 0, 0, 0};
            std::array<long, 3> envL_dims       = {0, 0, 0};
            std::array<long, 3> envR_dims       = {0, 0, 0};
            std::array<long, 4> t1_dims         = {0, 0, 0, 0};
            std::array<long, 4> t2_dims         = {0, 0, 0, 0};
            cutensorHandle_t    handle          = {};
            cutensorPlan_t      plan1           = {};
            cutensorPlan_t      plan2           = {};
            cutensorPlan_t      plan3           = {};
            void               *workspace       = nullptr;
            uint64_t            workspace_size  = 0;
            std::size_t         required_bytes  = 0;
            void               *d_mps           = nullptr;
            void               *d_mpo           = nullptr;
            void               *d_envL          = nullptr;
            void               *d_envR          = nullptr;
            void               *d_t1            = nullptr;
            void               *d_t2            = nullptr;
            void               *d_res           = nullptr;

            ~Cache() { reset(); }

            void release_buffers() {
                if(device >= 0) cudaSetDevice(device);
                if(d_res) cudaFree(d_res);
                if(d_t2) cudaFree(d_t2);
                if(d_t1) cudaFree(d_t1);
                if(d_envR) cudaFree(d_envR);
                if(d_envL) cudaFree(d_envL);
                if(d_mpo) cudaFree(d_mpo);
                if(d_mps) cudaFree(d_mps);
                if(workspace) cudaFree(workspace);

                workspace     = nullptr;
                d_mps         = nullptr;
                d_mpo         = nullptr;
                d_envL        = nullptr;
                d_envR        = nullptr;
                d_t1          = nullptr;
                d_t2          = nullptr;
                d_res         = nullptr;
                buffers_ready = false;
            }

            void reset() {
                release_buffers();
                if(device >= 0) cudaSetDevice(device);
                if(plan3_ready) cutensorDestroyPlan(plan3);
                if(plan2_ready) cutensorDestroyPlan(plan2);
                if(plan1_ready) cutensorDestroyPlan(plan1);
                if(handle_ready) cutensorDestroy(handle);

                signature_ready = false;
                handle_ready    = false;
                plan1_ready     = false;
                plan2_ready     = false;
                plan3_ready     = false;
                workspace_size  = 0;
                required_bytes  = 0;
                device          = -1;
            }
        };

        template<typename Scalar>
        Cache<Scalar> &get_cache() {
            static Cache<Scalar> cache;
            return cache;
        }

        template<typename Scalar, std::size_t Rank>
        auto create_tensor_desc(const cutensorHandle_t handle, const std::array<long, Rank> &dims) {
            cutensorTensorDescriptor_t desc{};
            auto                       extents = to_i64(dims);
            auto                       strides = to_strides(dims);
            check_cutensor(cutensorCreateTensorDescriptor(handle, &desc, static_cast<uint32_t>(Rank), extents.data(), strides.data(), Traits<Scalar>::data_type, 256u),
                           "cutensorCreateTensorDescriptor");
            return desc;
        }

        template<typename Scalar, std::size_t RankA, std::size_t RankB, std::size_t RankC, std::size_t RankD>
        auto create_plan(const cutensorHandle_t                    handle,
                         const std::array<long, RankA>            &dimsA,
                         const std::array<int32_t, RankA>         &modesA,
                         const std::array<long, RankB>            &dimsB,
                         const std::array<int32_t, RankB>         &modesB,
                         const std::array<long, RankC>            &dimsC,
                         const std::array<int32_t, RankC>         &modesC,
                         const std::array<long, RankD>            &dimsD,
                         const std::array<int32_t, RankD>         &modesD,
                         uint64_t                                 &workspace_size) {
            auto descA = create_tensor_desc<Scalar>(handle, dimsA);
            auto descB = create_tensor_desc<Scalar>(handle, dimsB);
            auto descC = create_tensor_desc<Scalar>(handle, dimsC);
            auto descD = create_tensor_desc<Scalar>(handle, dimsD);

            cutensorOperationDescriptor_t op_desc{};
            cutensorPlanPreference_t      pref{};
            cutensorPlan_t                plan{};

            check_cutensor(cutensorCreateContraction(handle, &op_desc, descA, modesA.data(), CUTENSOR_OP_IDENTITY, descB, modesB.data(), CUTENSOR_OP_IDENTITY, descC,
                                                     modesC.data(), CUTENSOR_OP_IDENTITY, descD, modesD.data(), Traits<Scalar>::compute_desc()),
                           "cutensorCreateContraction");
            check_cutensor(cutensorCreatePlanPreference(handle, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), "cutensorCreatePlanPreference");

            uint64_t stage_workspace = 0;
            check_cutensor(cutensorEstimateWorkspaceSize(handle, op_desc, pref, CUTENSOR_WORKSPACE_MIN, &stage_workspace), "cutensorEstimateWorkspaceSize");
            check_cutensor(cutensorCreatePlan(handle, &plan, op_desc, pref, stage_workspace), "cutensorCreatePlan");

            uint64_t required_workspace = 0;
            check_cutensor(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &required_workspace, sizeof(required_workspace)),
                           "cutensorPlanGetAttribute(CUTENSOR_PLAN_REQUIRED_WORKSPACE)");
            workspace_size = std::max(workspace_size, required_workspace);

            cutensorDestroyPlanPreference(pref);
            cutensorDestroyOperationDescriptor(op_desc);
            cutensorDestroyTensorDescriptor(descD);
            cutensorDestroyTensorDescriptor(descC);
            cutensorDestroyTensorDescriptor(descB);
            cutensorDestroyTensorDescriptor(descA);
            return plan;
        }

        template<typename Scalar>
        std::size_t compute_required_bytes(const Cache<Scalar> &cache) {
            return size_bytes(cache.mps_dims, sizeof(Scalar)) + size_bytes(cache.mpo_dims, sizeof(Scalar)) + size_bytes(cache.envL_dims, sizeof(Scalar)) +
                   size_bytes(cache.envR_dims, sizeof(Scalar)) + size_bytes(cache.t1_dims, sizeof(Scalar)) + size_bytes(cache.t2_dims, sizeof(Scalar)) +
                   size_bytes(cache.mps_dims, sizeof(Scalar)) + static_cast<std::size_t>(cache.workspace_size);
        }

        template<typename Scalar>
        void ensure_signature(Cache<Scalar>                          &cache,
                              int                                     device,
                              const std::array<long, 3>             &mps_dims,
                              const std::array<long, 4>             &mpo_dims,
                              const std::array<long, 3>             &envL_dims,
                              const std::array<long, 3>             &envR_dims) {
            if(cache.signature_ready and cache.device == device and cache.mps_dims == mps_dims and cache.mpo_dims == mpo_dims and cache.envL_dims == envL_dims and
               cache.envR_dims == envR_dims)
                return;

            cache.reset();
            cache.device    = device;
            cache.mps_dims  = mps_dims;
            cache.mpo_dims  = mpo_dims;
            cache.envL_dims = envL_dims;
            cache.envR_dims = envR_dims;
            cache.t1_dims   = {mps_dims[0], mps_dims[1], envR_dims[1], envR_dims[2]};
            cache.t2_dims   = {mpo_dims[0], mpo_dims[3], mps_dims[1], envR_dims[1]};

            check_cutensor(cutensorCreate(&cache.handle), "cutensorCreate");
            cache.handle_ready = true;

            // The matvec repeats the same local dimensions for many Krylov steps.
            // We therefore separate plan construction from buffer allocation so the
            // dispatcher can ask for the true device footprint before committing to
            // the GPU path, while still reusing the finished plan across calls.
            constexpr std::array<int32_t, 3> mps_modes  = {'s', 'l', 'r'};
            constexpr std::array<int32_t, 3> envR_modes = {'r', 'R', 'w'};
            constexpr std::array<int32_t, 4> t1_modes   = {'s', 'l', 'R', 'w'};
            constexpr std::array<int32_t, 4> mpo_modes  = {'W', 'w', 's', 'o'};
            constexpr std::array<int32_t, 4> t2_modes   = {'W', 'o', 'l', 'R'};
            constexpr std::array<int32_t, 3> envL_modes = {'l', 'L', 'W'};
            constexpr std::array<int32_t, 3> res_modes  = {'o', 'L', 'R'};

            cache.plan1 = create_plan<Scalar>(cache.handle, cache.mps_dims, mps_modes, cache.envR_dims, envR_modes, cache.t1_dims, t1_modes, cache.t1_dims, t1_modes,
                                              cache.workspace_size);
            cache.plan1_ready = true;
            cache.plan2 = create_plan<Scalar>(cache.handle, cache.mpo_dims, mpo_modes, cache.t1_dims, t1_modes, cache.t2_dims, t2_modes, cache.t2_dims, t2_modes,
                                              cache.workspace_size);
            cache.plan2_ready = true;
            cache.plan3 = create_plan<Scalar>(cache.handle, cache.t2_dims, t2_modes, cache.envL_dims, envL_modes, cache.mps_dims, res_modes, cache.mps_dims,
                                              res_modes, cache.workspace_size);
            cache.plan3_ready    = true;
            cache.required_bytes = compute_required_bytes(cache);
            cache.signature_ready = true;
        }

        template<typename Scalar>
        void ensure_buffers(Cache<Scalar> &cache) {
            if(cache.buffers_ready) return;

            // Buffer allocation is delayed until the dispatcher has already
            // decided that the operation fits comfortably on the active GPU.
            check_cuda(cudaMalloc(&cache.d_mps, size_bytes(cache.mps_dims, sizeof(Scalar))), "cudaMalloc(d_mps)");
            check_cuda(cudaMalloc(&cache.d_mpo, size_bytes(cache.mpo_dims, sizeof(Scalar))), "cudaMalloc(d_mpo)");
            check_cuda(cudaMalloc(&cache.d_envL, size_bytes(cache.envL_dims, sizeof(Scalar))), "cudaMalloc(d_envL)");
            check_cuda(cudaMalloc(&cache.d_envR, size_bytes(cache.envR_dims, sizeof(Scalar))), "cudaMalloc(d_envR)");
            check_cuda(cudaMalloc(&cache.d_t1, size_bytes(cache.t1_dims, sizeof(Scalar))), "cudaMalloc(d_t1)");
            check_cuda(cudaMalloc(&cache.d_t2, size_bytes(cache.t2_dims, sizeof(Scalar))), "cudaMalloc(d_t2)");
            check_cuda(cudaMalloc(&cache.d_res, size_bytes(cache.mps_dims, sizeof(Scalar))), "cudaMalloc(d_res)");
            if(cache.workspace_size > 0) check_cuda(cudaMalloc(&cache.workspace, cache.workspace_size), "cudaMalloc(workspace)");

            cache.buffers_ready = true;
        }
#endif
    }

    template<typename Scalar>
    std::size_t get_cutensor_operation_bytes([[maybe_unused]] std::array<long, 3> mps_dims,
                                             [[maybe_unused]] std::array<long, 4> mpo_dims,
                                             [[maybe_unused]] std::array<long, 3> envL_dims,
                                             [[maybe_unused]] std::array<long, 3> envR_dims) {
#if defined(DMRG_ENABLE_CUTENSOR)
        static_assert(cutensor_supported_v<Scalar>);
        const auto device = config::cuda::active_gpu_id();
        if(device < 0) return 0;

        auto &cache = get_cache<Scalar>();
        std::scoped_lock lock(cache.mutex);
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        ensure_signature(cache, device, mps_dims, mpo_dims, envL_dims, envR_dims);
        return cache.required_bytes;
#else
        return 0;
#endif
    }

    template<typename Scalar>
    bool cutensor_can_fit([[maybe_unused]] std::array<long, 3> mps_dims,
                          [[maybe_unused]] std::array<long, 4> mpo_dims,
                          [[maybe_unused]] std::array<long, 3> envL_dims,
                          [[maybe_unused]] std::array<long, 3> envR_dims) {
#if defined(DMRG_ENABLE_CUTENSOR)
        static_assert(cutensor_supported_v<Scalar>);
        const auto device = config::cuda::active_gpu_id();
        if(device < 0) return false;

        auto &cache = get_cache<Scalar>();
        std::scoped_lock lock(cache.mutex);
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        ensure_signature(cache, device, mps_dims, mpo_dims, envL_dims, envR_dims);
        // A cache entry with live buffers is stronger evidence than a fresh
        // free-memory query: this exact tensor shape has already been admitted
        // and its device storage is still resident.
        if(cache.buffers_ready) return true;

        return cache.required_bytes > 0 and config::cuda::query_memory(cache.required_bytes).fits;
#else
        return false;
#endif
    }

    template<typename Scalar>
    void contract_with_cutensor([[maybe_unused]] Scalar *res_ptr,
                                [[maybe_unused]] std::array<long, 3> res_dims,
                                [[maybe_unused]] const Scalar *mps_ptr,
                                [[maybe_unused]] std::array<long, 3> mps_dims,
                                [[maybe_unused]] const Scalar *mpo_ptr,
                                [[maybe_unused]] std::array<long, 4> mpo_dims,
                                [[maybe_unused]] const Scalar *envL_ptr,
                                [[maybe_unused]] std::array<long, 3> envL_dims,
                                [[maybe_unused]] const Scalar *envR_ptr,
                                [[maybe_unused]] std::array<long, 3> envR_dims) {
#if defined(DMRG_ENABLE_CUTENSOR)
        static_assert(cutensor_supported_v<Scalar>);
        const auto device = config::cuda::active_gpu_id();
        if(device < 0) throw except::runtime_error("cuTENSOR contraction requested without an active CUDA device");

        auto &cache = get_cache<Scalar>();
        std::scoped_lock lock(cache.mutex);

        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        ensure_signature(cache, device, mps_dims, mpo_dims, envL_dims, envR_dims);
        ensure_buffers(cache);

        // The cache keeps the persistent device buffers for one tensor shape.
        // The actual tensor contents still change from call to call, so each
        // matvec refreshes the four inputs, runs the three planned contractions,
        // and copies the result back to host memory.
        check_cuda(cudaMemcpy(cache.d_mps, mps_ptr, size_bytes(cache.mps_dims, sizeof(Scalar)), cudaMemcpyHostToDevice), "cudaMemcpy(mps)");
        check_cuda(cudaMemcpy(cache.d_mpo, mpo_ptr, size_bytes(cache.mpo_dims, sizeof(Scalar)), cudaMemcpyHostToDevice), "cudaMemcpy(mpo)");
        check_cuda(cudaMemcpy(cache.d_envL, envL_ptr, size_bytes(cache.envL_dims, sizeof(Scalar)), cudaMemcpyHostToDevice), "cudaMemcpy(envL)");
        check_cuda(cudaMemcpy(cache.d_envR, envR_ptr, size_bytes(cache.envR_dims, sizeof(Scalar)), cudaMemcpyHostToDevice), "cudaMemcpy(envR)");
        check_cuda(cudaMemset(cache.d_t1, 0, size_bytes(cache.t1_dims, sizeof(Scalar))), "cudaMemset(t1)");
        check_cuda(cudaMemset(cache.d_t2, 0, size_bytes(cache.t2_dims, sizeof(Scalar))), "cudaMemset(t2)");
        check_cuda(cudaMemset(cache.d_res, 0, size_bytes(cache.mps_dims, sizeof(Scalar))), "cudaMemset(res)");

        const auto alpha = Traits<Scalar>::one();
        const auto beta  = Traits<Scalar>::zero();

        check_cutensor(cutensorContract(cache.handle, cache.plan1, &alpha, cache.d_mps, cache.d_envR, &beta, cache.d_t1, cache.d_t1, cache.workspace,
                                        cache.workspace_size, nullptr),
                       "cutensorContract(stage1)");
        check_cutensor(cutensorContract(cache.handle, cache.plan2, &alpha, cache.d_mpo, cache.d_t1, &beta, cache.d_t2, cache.d_t2, cache.workspace,
                                        cache.workspace_size, nullptr),
                       "cutensorContract(stage2)");
        check_cutensor(cutensorContract(cache.handle, cache.plan3, &alpha, cache.d_t2, cache.d_envL, &beta, cache.d_res, cache.d_res, cache.workspace,
                                        cache.workspace_size, nullptr),
                       "cutensorContract(stage3)");

        check_cuda(cudaMemcpy(res_ptr, cache.d_res, size_bytes(cache.mps_dims, sizeof(Scalar)), cudaMemcpyDeviceToHost), "cudaMemcpy(res)");
#else
        throw except::runtime_error("cuTENSOR contraction requested, but DMRG_ENABLE_CUTENSOR is disabled");
#endif
    }

    template std::size_t get_cutensor_operation_bytes<fp32>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                                            std::array<long, 3> envR_dims);
    template std::size_t get_cutensor_operation_bytes<fp64>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                                            std::array<long, 3> envR_dims);
    template std::size_t get_cutensor_operation_bytes<cx32>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                                            std::array<long, 3> envR_dims);
    template std::size_t get_cutensor_operation_bytes<cx64>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                                            std::array<long, 3> envR_dims);

    template bool cutensor_can_fit<fp32>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                         std::array<long, 3> envR_dims);
    template bool cutensor_can_fit<fp64>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                         std::array<long, 3> envR_dims);
    template bool cutensor_can_fit<cx32>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                         std::array<long, 3> envR_dims);
    template bool cutensor_can_fit<cx64>(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                         std::array<long, 3> envR_dims);

    template void contract_with_cutensor(fp32 *res_ptr, std::array<long, 3> res_dims, const fp32 *mps_ptr, std::array<long, 3> mps_dims, const fp32 *mpo_ptr,
                                         std::array<long, 4> mpo_dims, const fp32 *envL_ptr, std::array<long, 3> envL_dims, const fp32 *envR_ptr,
                                         std::array<long, 3> envR_dims);
    template void contract_with_cutensor(fp64 *res_ptr, std::array<long, 3> res_dims, const fp64 *mps_ptr, std::array<long, 3> mps_dims, const fp64 *mpo_ptr,
                                         std::array<long, 4> mpo_dims, const fp64 *envL_ptr, std::array<long, 3> envL_dims, const fp64 *envR_ptr,
                                         std::array<long, 3> envR_dims);
    template void contract_with_cutensor(cx32 *res_ptr, std::array<long, 3> res_dims, const cx32 *mps_ptr, std::array<long, 3> mps_dims, const cx32 *mpo_ptr,
                                         std::array<long, 4> mpo_dims, const cx32 *envL_ptr, std::array<long, 3> envL_dims, const cx32 *envR_ptr,
                                         std::array<long, 3> envR_dims);
    template void contract_with_cutensor(cx64 *res_ptr, std::array<long, 3> res_dims, const cx64 *mps_ptr, std::array<long, 3> mps_dims, const cx64 *mpo_ptr,
                                         std::array<long, 4> mpo_dims, const cx64 *envL_ptr, std::array<long, 3> envL_dims, const cx64 *envR_ptr,
                                         std::array<long, 3> envR_dims);
}
