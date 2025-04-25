#pragma once

#include <memory>
namespace h5pp {
    class File;
}

class AlgorithmLauncher {
    private:
    std::shared_ptr<h5pp::File> h5file;

    public:
    AlgorithmLauncher();
    void start_h5file();
    void setup_temp_path();
    void run_algorithms();
    template <typename Scalar> void run_idmrg();
    template <typename Scalar> void run_fdmrg();
    template <typename Scalar> void run_flbit();
    template <typename Scalar> void run_xdmrg();
    template <typename Scalar> void run_itebd();
};
