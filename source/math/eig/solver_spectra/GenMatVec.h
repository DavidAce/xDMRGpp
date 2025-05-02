#pragma once
#include <Spectra/Util/CompInfo.h>
template<typename MatVec>
class GenMatVec {
    public:
    using Scalar     = typename MatVec::Scalar;
    using RealScalar = typename MatVec::RealScalar;

    private:
    const MatVec             &mv;
    mutable Spectra::CompInfo m_info    = Spectra::CompInfo::NotComputed;
    long                      maxiters  = 500;
    RealScalar                tolerance = RealScalar{2e-5f};

    public:
    Spectra::CompInfo info() const { return m_info; }
    GenMatVec(const MatVec &mv_) : mv(mv_) {}

    int  rows() const { return mv.rows(); }
    int  cols() const { return mv.cols(); }
    void set_maxiters(long max) { maxiters = max; }
    void set_tolerance(RealScalar tol) { tolerance = tol; }
    // Perform the matrix-vector multiplication operation y=Bx.
    void perform_op(const Scalar *x_in, Scalar *y_out) const { mv.MultBx(x_in, y_out); }

    // Perform the solving operation y_out = inv(B) * x_in
    void solve(const Scalar *x_in, Scalar *y_out) const {
        mv.MultInvBx(x_in, y_out, maxiters, tolerance);
        // m_info = Spectra::CompInfo::Successful;
    }
};
