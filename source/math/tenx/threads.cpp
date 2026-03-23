#include "threads.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include <mutex> // std::once_flag, std::call_once

namespace tenx::threads {
    template<typename T>
    requires std::is_integral_v<T>
    void setNumThreads([[maybe_unused]] T num) noexcept {
#if defined(EIGEN_USE_THREADS)
        internal::num_threads = static_cast<unsigned int>(num);
#endif
    }
    template void setNumThreads(int num) noexcept;
    template void setNumThreads(long num) noexcept;
    template void setNumThreads(unsigned int num) noexcept;
    template void setNumThreads(unsigned long num) noexcept;

#if defined(EIGEN_USE_THREADS)
    std::unique_ptr<internal::ThreadPoolWrapper> internal::singleThreadWrapper;
    std::unique_ptr<internal::ThreadPoolWrapper> internal::multiThreadWrapper;

    namespace {
        std::once_flag    init_flag;
        std::atomic<bool> initialized{false};
    }

    int getNumThreads() noexcept {
    #if defined(_OPENMP)
        if(omp_in_parallel()) { return 1; } // Avoid simultaneous parallelization
    #endif
        return static_cast<int>(internal::num_threads);
    }

    internal::ThreadPoolWrapper::ThreadPoolWrapper(int nt)
        : tp(std::make_unique<Eigen::ThreadPool>(nt)), dev(std::make_unique<Eigen::ThreadPoolDevice>(tp.get(), nt)) {}

    const std::unique_ptr<internal::ThreadPoolWrapper> &get() noexcept {
    #if defined(_OPENMP)
        if(omp_in_parallel() && !initialized.load(std::memory_order_acquire)) {
            std::fprintf(stderr, "tenx::threads::get(): first-time initialization inside an OpenMP parallel region is forbidden");
            std::abort();
        }
    #endif

        std::call_once(init_flag, [] {
            // Create both wrappers once (simple, matches old semantics closely)
            internal::singleThreadWrapper = std::make_unique<internal::ThreadPoolWrapper>(1);

            // If num_threads==1 at init time, multiThreadWrapper may never be used,
            // but constructing it here keeps the logic very simple.
            const int nt                 = std::max(1, static_cast<int>(internal::num_threads));
            internal::multiThreadWrapper = (nt > 1) ? std::make_unique<internal::ThreadPoolWrapper>(nt) : nullptr;

            initialized.store(true, std::memory_order_release);
        });

        if(internal::num_threads == 1) return internal::singleThreadWrapper;

    #if defined(_OPENMP)
        if(omp_in_parallel()) return internal::singleThreadWrapper;
    #endif

        // If multiThreadWrapper wasn't created because nt==1 at init, fall back.
        return internal::multiThreadWrapper ? internal::multiThreadWrapper : internal::singleThreadWrapper;
    }

#else
    internal::DefaultDeviceWrapper::DefaultDeviceWrapper() : dev(std::make_unique<Eigen::DefaultDevice>()) {}

    void setNumThreads([[maybe_unused]] int num) {}

    namespace {
        std::once_flag    init_flag;
        std::atomic<bool> initialized{false};
    }

    const std::unique_ptr<internal::DefaultDeviceWrapper> &get() noexcept {
    #if defined(_OPENMP)
        if(omp_in_parallel() && !initialized.load(std::memory_order_acquire)) {
            std::fprintf(stderr, "tenx::threads::get(): first-time initialization inside an OpenMP parallel region is forbidden");
            std::abort();
        }
    #endif
        std::call_once(init_flag, [] {
            internal::defaultDeviceWrapper = std::make_unique<internal::DefaultDeviceWrapper>();
            initialized.store(true, std::memory_order_release);
        });
        return internal::defaultDeviceWrapper;
    }
#endif
}
