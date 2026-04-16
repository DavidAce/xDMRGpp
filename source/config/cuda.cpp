#include "cuda.h"
#include "debug/exceptions.h"
#include "settings.h"
#include "tools/common/log.h"
#include <cmath>
#include <fmt/core.h>
#include <mutex>
#include <string>

#if defined(DMRG_ENABLE_CUTENSOR)
    #include <cuda_runtime_api.h>
    #include <cutensor.h>
#endif

namespace config::cuda {
    namespace {
        struct CudaState {
            bool        initialized      = false;
            bool        available        = false;
            int         requested_device = -1;
            int         active_device    = -1;
            std::string description      = "CUDA/cuTENSOR support is disabled in this build";
        };

        CudaState      cudaState;
        std::once_flag init_flag;

#if defined(DMRG_ENABLE_CUTENSOR)
        inline void check_cuda(cudaError_t err, const char *expr) {
            if(err != cudaSuccess) throw except::runtime_error("{} failed: {}", expr, cudaGetErrorString(err));
        }

        bool probe_device(int device, std::string &desc) {
            if(auto err = cudaSetDevice(device); err != cudaSuccess) return false;
            if(auto err = cudaFree(nullptr); err != cudaSuccess) return false;

            cudaDeviceProp prop{};
            if(auto err = cudaGetDeviceProperties(&prop, device); err != cudaSuccess) return false;

            cutensorHandle_t handle{};
            if(auto err = cutensorCreate(&handle); err != CUTENSOR_STATUS_SUCCESS) return false;
            cutensorDestroy(handle);

            desc = fmt::format("CUDA/cuTENSOR active on device {}: {} | cc {}.{} | gpu_switchsize {} | gpu_max_alloc_fraction {:.2f}", device, prop.name,
                               prop.major, prop.minor, settings::cuda::gpu_switchsize, settings::cuda::gpu_max_alloc_fraction);
            return true;
        }
#endif
    }

    bool compiled() noexcept {
#if defined(DMRG_ENABLE_CUTENSOR)
        return true;
#else
        return false;
#endif
    }

    void initialize() {
        std::call_once(init_flag, [] {
            cudaState.initialized      = true;
            cudaState.requested_device = settings::cuda::device;

#if defined(DMRG_ENABLE_CUTENSOR)
            int device_count = 0;
            if(auto err = cudaGetDeviceCount(&device_count); err != cudaSuccess) {
                cudaState.description = fmt::format("CUDA/cuTENSOR unavailable: cudaGetDeviceCount failed: {}", cudaGetErrorString(err));
            } else if(device_count <= 0) {
                cudaState.description = "CUDA/cuTENSOR unavailable: no CUDA devices detected";
            } else if(cudaState.requested_device >= 0) {
                if(cudaState.requested_device >= device_count) {
                    throw except::runtime_error("Requested --gpuid={} but only {} CUDA device(s) are visible", cudaState.requested_device, device_count);
                }
                std::string desc;
                if(not probe_device(cudaState.requested_device, desc)) {
                    throw except::runtime_error("Requested --gpuid={} but the device failed CUDA/cuTENSOR initialization", cudaState.requested_device);
                }
                cudaState.available     = true;
                cudaState.active_device = cudaState.requested_device;
                cudaState.description   = desc;
            } else {
                for(int device = 0; device < device_count; ++device) {
                    std::string desc;
                    if(probe_device(device, desc)) {
                        cudaState.available     = true;
                        cudaState.active_device = device;
                        cudaState.description   = desc;
                        break;
                    }
                }
                if(not cudaState.available) cudaState.description = "CUDA/cuTENSOR unavailable: no working CUDA device passed the runtime probe";
            }
#endif

            if(tools::log) {
                if(cudaState.available)
                    tools::log->info("{}", cudaState.description);
                else
                    tools::log->warn("{}", cudaState.description);
            }
        });
    }

    bool available() {
        initialize();
        return cudaState.available;
    }

    MemoryStatus query_memory(std::size_t required_bytes) {
        initialize();
        MemoryStatus status;
        status.required_bytes = required_bytes;

#if defined(DMRG_ENABLE_CUTENSOR)
        if(not cudaState.available) return status;
        check_cuda(cudaSetDevice(cudaState.active_device), "cudaSetDevice");

        std::size_t free_bytes  = 0;
        std::size_t total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

        const auto clamped_fraction = std::clamp(settings::cuda::gpu_max_alloc_fraction, 0.0, 1.0);
        status.free_bytes           = free_bytes;
        status.total_bytes          = total_bytes;
        status.usable_bytes         = static_cast<std::size_t>(std::floor(static_cast<double>(free_bytes) * clamped_fraction));
        status.fits                 = required_bytes == 0 or required_bytes <= status.usable_bytes;
#endif
        return status;
    }

    int requested_device() noexcept { return cudaState.requested_device; }

    int active_device() {
        initialize();
        return cudaState.active_device;
    }

    const std::string &description() {
        initialize();
        return cudaState.description;
    }
}
