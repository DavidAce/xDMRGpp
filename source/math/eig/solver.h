#pragma once
#include "enums.h"
#include "settings.h"
#include "solution.h"
#include <memory>

namespace spdlog {
    class logger;
}
struct primme_params;

namespace eig {
    int getBasisSize(long L, int nev, std::optional<int> basisSize);

    class solver {
        public:
        eig::settings                   config;
        eig::solution                   result;
        std::shared_ptr<spdlog::logger> log;
        solver();
        solver(const eig::settings &config);
        void setLogLevel(size_t loglevel);
        template<typename Scalar>
        void subtract_phase(std::vector<Scalar> &eigvecs, size_type L, int nev);

        // Functions for full diagonalization of explicit matrix
        int sgeev(fp32 *matrix, size_type L);
        int sggev(fp32 *matrixA, fp32 *matrixB, size_type L);
        int ssyevd(fp32 *matrix, size_type L);
        int ssyevr(fp32 *matrix, size_type L, char range, int il, int iu, fp32 vl, fp32 vu);
        int ssyevx(fp32 *matrix, size_type L, char range, int il, int iu, fp32 vl, fp32 vu);
        int ssygvd(fp32 *matrixA, fp32 *matrixB, size_type L);
        int ssygvx(fp32 *matrixA, fp32 *matrixB, size_type L, char range, int il, int iu, fp32 vl, fp32 vu);

        int dgeev(fp64 *matrix, size_type L);
        int dggev(fp64 *matrixA, fp64 *matrixB, size_type L);
        int dsyevd(fp64 *matrix, size_type L);
        int dsyevr(fp64 *matrix, size_type L, char range, int il, int iu, fp64 vl, fp64 vu);
        int dsyevx(fp64 *matrix, size_type L, char range, int il, int iu, fp64 vl, fp64 vu);
        int dsygvd(fp64 *matrixA, fp64 *matrixB, size_type L);
        int dsygvx(fp64 *matrixA, fp64 *matrixB, size_type L, char range, int il, int iu, fp64 vl, fp64 vu);

        int cheev(cx32 *matrix, size_type L);
        int cheevd(cx32 *matrix, size_type L);
        int cheevr(cx32 *matrix, size_type L, char range, int il, int iu, fp32 vl, fp32 vu);
        int cgeev(cx32 *matrix, size_type L);
        int chegv(cx32 *matrixA, cx32 *matrixB, size_type L);
        int chegvd(cx32 *matrixA, cx32 *matrixB, size_type L);
        int chegvx(cx32 *matrixA, cx32 *matrixB, size_type L, char range, int il, int iu, fp32 vl, fp32 vu);
        int cggev(cx32 *matrixA, cx32 *matrixB, size_type L);

        int zheev(cx64 *matrix, size_type L);
        int zheevd(cx64 *matrix, size_type L);
        int zheevr(cx64 *matrix, size_type L, char range, int il, int iu, double vl, double vu);
        int zgeev(cx64 *matrix, size_type L);
        int zhegv(cx64 *matrixA, cx64 *matrixB, size_type L);
        int zhegvd(cx64 *matrixA, cx64 *matrixB, size_type L);
        int zhegvx(cx64 *matrixA, cx64 *matrixB, size_type L, char range, int il, int iu, double vl, double vu);
        int zggev(cx64 *matrixA, cx64 *matrixB, size_type L);

        void eig_init(Form form, Type type, Vecs compute_eigvecs, Dephase remove_phase_);
        template<Form form = Form::SYMM, typename Scalar>
        void eig(Scalar *matrix, size_type L, Vecs compute_eigvecs = Vecs::ON, Dephase remove_phase_ = Dephase::OFF);

        template<Form form = Form::SYMM, typename Scalar>
        void eig(Scalar *matrixA, Scalar *matrixB, size_type L, Vecs compute_eigvecs = Vecs::ON, Dephase remove_phase_ = Dephase::OFF);

        template<Form form = Form::SYMM, typename Scalar>
        void eig(Scalar *matrix, size_type L, char range, int il, int iu, double vl, double vu, Vecs compute_eigvecs = Vecs::ON,
                 Dephase remove_phase_ = Dephase::OFF);
        template<Form form = Form::SYMM, typename Scalar>
        void eig(Scalar *matrixA, Scalar *matrixB, size_type L, char range, int il, int iu, double vl, double vu, Vecs compute_eigvecs = Vecs::ON,
                 Dephase remove_phase_ = Dephase::OFF);

        // Functions for few eigensolutions
        template<typename Scalar>
        void eigs_init(size_type L, int nev, int ncv, Ritz ritz = Ritz::LM, Form form = Form::SYMM, Type type = Type::FP64, Side side = Side::R,
                       std::optional<cx64> sigma = std::nullopt, Shinv shinv = Shinv::OFF, Storage storage = Storage::DENSE, Vecs compute_eigvecs_ = Vecs::OFF,
                       Dephase remove_phase_ = Dephase::OFF, Scalar *residual = nullptr, Lib lib = Lib::ARPACK);

        template<typename MatrixProductType>
        void set_default_config(const MatrixProductType &matrix);

        template<typename Scalar, Storage storage = Storage::DENSE>
        void eigs(const Scalar *matrix, size_type L, int nev, int ncv, Ritz ritz = Ritz::SR, Form form = Form::SYMM, Side side = Side::R,
                  std::optional<cx64> sigma = std::nullopt, Shinv shinv = Shinv::OFF, Vecs vecs = Vecs::ON, Dephase remove_phase = Dephase::OFF,
                  Scalar *residual = nullptr);

        template<typename MatrixProductType>
        void eigs(MatrixProductType &matrix, int nev, int ncv, Ritz ritz = Ritz::SR, Form form = Form::SYMM, Side side = Side::R,
                  std::optional<cx64> sigma = std::nullopt, Shinv shinv = Shinv::OFF, Vecs vecs = Vecs::ON, Dephase remove_phase = Dephase::OFF,
                  typename MatrixProductType::Scalar *residual = nullptr);

        template<typename MatrixProductType>
        void eigs(MatrixProductType &matrix);

        template<typename MatrixProductType>
        int eigs_primme(MatrixProductType &matrix);

        private:
        template<typename MatrixProductType>
        static void MultAx_wrapper(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr);
        template<typename MatrixProductType>
        static void MultOPv_wrapper(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr);
    };
}
