#pragma once
#include <memory>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <mutex>
namespace tenx {
    namespace threads {

#if defined(EIGEN_USE_THREADS)
        namespace internal {
            static std::once_flag init_flag;
            struct ThreadPoolWrapper {
                private:
                std::unique_ptr<Eigen::ThreadPool> tp;

                public:
                std::unique_ptr<Eigen::ThreadPoolDevice> dev;

                ThreadPoolWrapper(int nt);
            };
            inline unsigned int                       num_threads = 1;
            extern std::unique_ptr<ThreadPoolWrapper> singleThreadWrapper;
            extern std::unique_ptr<ThreadPoolWrapper> multiThreadWrapper;
        }

        template<typename T>
        requires std::is_integral_v<T>
        void       setNumThreads(T num) noexcept;
        extern int getNumThreads() noexcept;
        //        internal::ThreadPoolWrapper &get() noexcept;
        const std::unique_ptr<internal::ThreadPoolWrapper> &get() noexcept;
#else
        namespace internal {
            static std::once_flag init_flag;
            struct DefaultDeviceWrapper {
                std::unique_ptr<Eigen::DefaultDevice> dev;
                DefaultDeviceWrapper();
            };
            inline unsigned int                          num_threads = 1;
            extern std::unique_ptr<DefaultDeviceWrapper> defaultDeviceWrapper;
        }

        void                                                   setNumThreads([[maybe_unused]] int num);
        const std::unique_ptr<internal::DefaultDeviceWrapper> &get() noexcept;
#endif

    }
}