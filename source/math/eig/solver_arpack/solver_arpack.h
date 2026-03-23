#pragma once
#include "../enums.h"

namespace eig {
    class settings;
    class solution;

    template<typename MatrixType>
    class solver_arpack {
        private:
        int nev_internal;
        int ncv_internal;

        void eigs_sym();
        void eigs_nsym();
        void eigs_comp();

        void eigs_sym_rc();
        void eigs_nsym_rc();
        void eigs_comp_rc();

        public:
        using Scalar = typename MatrixType::Scalar;

        void eigs();
        template<typename Derived>
        void find_solution(Derived &solver, int nev);
        template<typename Derived>
        void find_solution_rc(Derived &solver);
        template<typename Derived>
        void copy_solution(Derived &solver);

        template<typename eval_t, typename evec_t, Side side>
        void compute_residual_norms();

        MatrixType    &matrix;
        eig::settings &config;
        eig::solution &result;
        Scalar        *residual = nullptr;
        solver_arpack(MatrixType &matrix_, eig::settings &config_, eig::solution &result_);
    };

}
