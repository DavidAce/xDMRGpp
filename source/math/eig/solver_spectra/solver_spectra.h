#pragma once
#include "../enums.h"
#include "../settings.h"
#include "../sfinae.h"
#include "../solution.h"
#include <complex>
#include <memory>
#include <vector>

namespace eig {
    template<typename MatrixType>
    class solver_spectra {
        private:
        int nev_internal;
        int ncv_internal;

        public:
        using Scalar = typename MatrixType::Scalar;

        void eigs();

        MatrixType    &matrix;
        eig::settings &config;
        eig::solution &result;
        Scalar        *residual = nullptr;
        solver_spectra(MatrixType &matrix_, eig::settings &config_, eig::solution &result_);
    };

}
