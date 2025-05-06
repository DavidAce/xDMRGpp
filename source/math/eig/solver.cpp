#include "solver.h"
#include "debug/exceptions.h"
#include "log.h"
#include "matvec/matvec_dense.h"
#include "matvec/matvec_mpo.h"
#include "matvec/matvec_mpos.h"
#include "matvec/matvec_sparse.h"
#include "matvec/matvec_zero.h"
#include "solver_arpack/solver_arpack.h"
#include "solver_spectra/solver_spectra.h"
#include "tid/tid.h"
#include <general/sfinae.h>

int eig::getBasisSize(long L, int nev, std::optional<int> basisSize) {
    if(not basisSize.has_value() or basisSize.value() <= 0) { basisSize = nev * safe_cast<int>(std::ceil(std::log2(L))); }
    return std::clamp(basisSize.value(), 2 * nev, safe_cast<int>(L));
}

eig::solver::solver() {
    if(config.loglevel)
        eig::setLevel(config.loglevel.value());
    else
        eig::setLevel(2);
    eig::setTimeStamp();
    log = eig::log;
}

eig::solver::solver(const eig::settings &config_) : eig::solver::solver() { config = config_; }

void eig::solver::setLogLevel(size_t loglevel) {
    config.loglevel = loglevel;
    eig::setLevel(loglevel);
}

template<typename Scalar>
void eig::solver::subtract_phase(std::vector<Scalar> &eigvecs, size_type L, int nev)
// The solution to  the eigenvalue equation Av = l*v is determined up to a constant phase factor, i.e., if v
// is a solution, so is v*exp(i*theta). By computing the complex angle of the first element in v, one can then
// remove it from all other elements of v.
{
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        if(eigvecs.empty()) return;
        using Real = typename Scalar::value_type;
        if(nev > 0) {
            for(int i = 0; i < nev; i++) {
                if(eigvecs[safe_cast<size_t>(i * L)].imag() == Real{0}) { continue; }
                Scalar inv_phase     = Scalar(Real{0}, Real{-1}) * std::arg(eigvecs[safe_cast<size_t>(i * L)]);
                auto   begin         = eigvecs.begin() + i * L;
                auto   end           = begin + L;
                Scalar exp_inv_phase = std::exp(inv_phase);
                std::transform(begin, end, begin, [exp_inv_phase](Scalar num) -> Scalar { return (num * exp_inv_phase); });
                std::transform(begin, end, begin, [](Scalar num) -> Scalar { return std::abs(num.imag()) > static_cast<Real>(1e-15) ? num : std::real(num); });
            }
        } else {
            throw std::logic_error("Subtract phase requires nev > 0");
        }
    }
}

template void eig::solver::subtract_phase(std::vector<fp32> &eigvecs, size_type L, int nev);
template void eig::solver::subtract_phase(std::vector<fp64> &eigvecs, size_type L, int nev);
template void eig::solver::subtract_phase(std::vector<fp128> &eigvecs, size_type L, int nev);
template void eig::solver::subtract_phase(std::vector<cx64> &eigvecs, size_type L, int nev);
template void eig::solver::subtract_phase(std::vector<cx32> &eigvecs, size_type L, int nev);
template void eig::solver::subtract_phase(std::vector<cx128> &eigvecs, size_type L, int nev);

void eig::solver::eig_init(Form form, Type type, Vecs compute_eigvecs, Dephase remove_phase) {
    eig::log->trace("eig init");
    result.reset();
    config.compute_eigvecs = config.compute_eigvecs.value_or(compute_eigvecs);
    config.remove_phase    = config.remove_phase.value_or(remove_phase);
    config.type            = config.type.value_or(type);
    config.form            = config.form.value_or(form);
    config.side            = config.side.value_or(Side::LR);
    config.storage         = config.storage.value_or(Storage::DENSE);
}

template<eig::Form form, typename Scalar>
void eig::solver::eig(Scalar *matrix, size_type L, Vecs compute_eigvecs_, Dephase remove_phase_) {
    static_assert(!std::is_const_v<Scalar>);
    auto t_eig = tid::tic_scope("eig");
    int  info  = 0;
    try {
        if constexpr(std::is_same_v<Scalar, fp32>) {
            eig_init(form, Type::FP32, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack ssyevd", config.tag, config.tag.empty() ? "" : " ");
                info       = ssyevd(matrix, L);
            } else if constexpr(form == Form::NSYM) {
                config.tag = fmt::format("{}{}lapack sgeev", config.tag, config.tag.empty() ? "" : " ");
                info       = sgeev(matrix, L);
            }
        } else if constexpr(std::is_same_v<Scalar, fp64>) {
            eig_init(form, Type::FP64, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack dsyevd", config.tag, config.tag.empty() ? "" : " ");
                info       = dsyevd(matrix, L);
            } else if constexpr(form == Form::NSYM) {
                config.tag = fmt::format("{}{}lapack dgeev", config.tag, config.tag.empty() ? "" : " ");
                info       = dgeev(matrix, L);
            }
        } else if constexpr(std::is_same_v<Scalar, cx32>) {
            eig_init(form, Type::CX32, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack cheevd", config.tag, config.tag.empty() ? "" : " ");
                info       = cheevd(matrix, L);
                //                if(config.tag.empty()) config.tag = "zheev";
                //                info = zheev(matrix, L);
            } else if constexpr(form == Form::NSYM) {
                config.tag = fmt::format("{}{}lapack cgeev", config.tag, config.tag.empty() ? "" : " ");
                info       = cgeev(matrix, L);
            }
        } else if constexpr(std::is_same_v<Scalar, cx64>) {
            eig_init(form, Type::CX64, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack zheevd", config.tag, config.tag.empty() ? "" : " ");
                info       = zheevd(matrix, L);
                //                if(config.tag.empty()) config.tag = "zheev";
                //                info = zheev(matrix, L);
            } else if constexpr(form == Form::NSYM) {
                config.tag = fmt::format("{}{}lapack zgeev", config.tag, config.tag.empty() ? "" : " ");
                info       = zgeev(matrix, L);
            }
        } else {
            throw except::runtime_error("Unknown type");
        }

    } catch(std::exception &ex) {
        eig::log->error("Eigenvalue solver failed: {}", ex.what());
        throw except::runtime_error("Eigenvalue solver Failed: {}", ex.what());
    }
    result.build_eigvals_cx32();
    result.build_eigvecs_cx32();
    result.build_eigvals_cx64();
    result.build_eigvecs_cx64();
    result.build_eigvals_cx128();
    result.build_eigvecs_cx128();
    if(info == 0 and config.remove_phase and config.remove_phase.value() == Dephase::OFF) {
        // The solution to  the eigenvalue equation Av = l*v is determined up to a constant phase factor, i.e., if v
        // is a solution, so is v*exp(i*theta). By computing the complex angle of the first element in v, one can then
        // remove it from all other elements of v.
        subtract_phase(result.eigvecsL_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsL_cx64, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx64, L, result.meta.nev);
        subtract_phase(result.eigvecsL_cx128, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx128, L, result.meta.nev);
    }
}
template void eig::solver::eig<eig::Form::SYMM>(fp32 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(fp32 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp64 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(fp64 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp128 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(fp128 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx32 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(cx32 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx64 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(cx64 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx128 *matrix, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(cx128 *matrix, size_type, Vecs, Dephase);

template<eig::Form form, typename Scalar>
void eig::solver::eig(Scalar *matrixA, Scalar *matrixB, size_type L, Vecs compute_eigvecs_, Dephase remove_phase_) {
    static_assert(!std::is_const_v<Scalar>);
    auto t_eig = tid::tic_scope("eig");
    int  info  = 0;
    try {
        if(matrixB == nullptr) {
            if constexpr(std::is_same_v<Scalar, fp32>) {
                eig_init(form, Type::FP32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack ssyevd", config.tag, config.tag.empty() ? "" : " ");
                    info       = ssyevd(matrixA, L);

                } else if constexpr(form == Form::NSYM) {
                    config.tag = fmt::format("{}{}lapack sgeev", config.tag, config.tag.empty() ? "" : " ");
                    info       = sgeev(matrixA, L);
                }
            } else if constexpr(std::is_same_v<Scalar, fp64>) {
                eig_init(form, Type::FP64, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack dsyevd", config.tag, config.tag.empty() ? "" : " ");
                    info       = dsyevd(matrixA, L);

                } else if constexpr(form == Form::NSYM) {
                    config.tag = fmt::format("{}{}lapack dgeev", config.tag, config.tag.empty() ? "" : " ");
                    info       = dgeev(matrixA, L);
                }
            } else if constexpr(std::is_same_v<Scalar, cx32>) {
                eig_init(form, Type::CX32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack cheevd", config.tag, config.tag.empty() ? "" : " ");
                    info       = cheevd(matrixA, L);
                } else if constexpr(form == Form::NSYM) {
                    config.tag = fmt::format("{}{}lapack cgeev", config.tag, config.tag.empty() ? "" : " ");
                    info       = cgeev(matrixA, L);
                }
            } else if constexpr(std::is_same_v<Scalar, cx64>) {
                eig_init(form, Type::CX64, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack zheevd", config.tag, config.tag.empty() ? "" : " ");
                    info       = zheevd(matrixA, L);
                } else if constexpr(form == Form::NSYM) {
                    config.tag = fmt::format("{}{}lapack zgeev", config.tag, config.tag.empty() ? "" : " ");
                    info       = zgeev(matrixA, L);
                }
            } else {
                throw except::runtime_error("Unknown type");
            }
        } else {
            if constexpr(std::is_same_v<Scalar, fp32>) {
                eig_init(form, Type::FP32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack ssygvd", config.tag, config.tag.empty() ? "" : " ");
                    info       = ssygvd(matrixA, matrixB, L);
                } else if constexpr(form == Form::NSYM) {
                    throw except::logic_error("dggev has not been implemented");
                    if(config.tag.empty()) config.tag = "sggev";
                    // info = sggev(matrixA, matrixB, L);
                }
            } else if constexpr(std::is_same_v<Scalar, fp64>) {
                eig_init(form, Type::FP64, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack dsygvd", config.tag, config.tag.empty() ? "" : " ");
                    info       = dsygvd(matrixA, matrixB, L);
                } else if constexpr(form == Form::NSYM) {
                    throw except::logic_error("dggev has not been implemented");
                    if(config.tag.empty()) config.tag = "dggev";
                    // info = dggev(matrixA, matrixB, L);
                }
            } else if constexpr(std::is_same_v<Scalar, cx32>) {
                eig_init(form, Type::CX32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    throw except::logic_error("chegvd has not been implemented");
                    if(config.tag.empty()) config.tag = "chegvd";
                    // info = zhegvd(matrixA, matrixB, L);
                    //                if(config.tag.empty()) config.tag = "zheev";
                    //                info = zheev(matrix, L);
                } else if constexpr(form == Form::NSYM) {
                    config.tag = fmt::format("{}{}lapack cggev", config.tag, config.tag.empty() ? "" : " ");
                    throw except::logic_error("cggev has not been implemented");
                    // info = zggev(matrixA, matrixB, L);
                }
            } else if constexpr(std::is_same_v<Scalar, cx64>) {
                eig_init(form, Type::CX64, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    throw except::logic_error("zhegvd has not been implemented");
                    if(config.tag.empty()) config.tag = "zhegvd";
                    // info = zhegvd(matrixA, matrixB, L);
                    //                if(config.tag.empty()) config.tag = "zheev";
                    //                info = zheev(matrix, L);
                } else if constexpr(form == Form::NSYM) {
                    config.tag = fmt::format("{}{}lapack zggev", config.tag, config.tag.empty() ? "" : " ");
                    throw except::logic_error("zggev has not been implemented");
                    // info = zggev(matrixA, matrixB, L);
                }
            } else {
                throw except::runtime_error("Unknown type");
            }
        }

    } catch(std::exception &ex) {
        eig::log->error("Eigenvalue solver failed: {}", ex.what());
        throw except::runtime_error("Eigenvalue solver Failed: {}", ex.what());
    }
    result.build_eigvals_cx32();
    result.build_eigvecs_cx32();
    result.build_eigvals_cx64();
    result.build_eigvecs_cx64();
    result.build_eigvals_cx128();
    result.build_eigvecs_cx128();
    if(info == 0 and config.remove_phase and config.remove_phase.value() == Dephase::OFF) {
        // The solution to  the eigenvalue equation Av = l*v is determined up to a constant phase factor, i.e., if v
        // is a solution, so is v*exp(i*theta). By computing the complex angle of the first element in v, one can then
        // remove it from all other elements of v.
        subtract_phase(result.eigvecsL_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsL_cx64, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx64, L, result.meta.nev);
    }
}
template void eig::solver::eig<eig::Form::SYMM>(fp32 *matrixA, fp32 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(fp32 *matrixA, fp32 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp64 *matrixA, fp64 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(fp64 *matrixA, fp64 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp128 *matrixA, fp128 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(fp128 *matrixA, fp128 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx32 *matrixA, cx32 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(cx32 *matrixA, cx32 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx64 *matrixA, cx64 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(cx64 *matrixA, cx64 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx128 *matrixA, cx128 *matrixB, size_type, Vecs, Dephase);
template void eig::solver::eig<eig::Form::NSYM>(cx128 *matrixA, cx128 *matrixB, size_type, Vecs, Dephase);

template<eig::Form form, typename Scalar, typename RealScalar>
void eig::solver::eig(Scalar *matrix, size_type L, char range, int il, int iu, RealScalar vl, RealScalar vu, Vecs compute_eigvecs_, Dephase remove_phase_) {
    static_assert(form == Form::SYMM);
    static_assert(!std::is_const_v<Scalar>);
    auto t_eig = tid::tic_scope("eig");
    int  info  = 0;
    try {
        if constexpr(std::is_same_v<Scalar, fp32>) {
            eig_init(form, Type::FP32, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack ssyevr", config.tag, config.tag.empty() ? "" : " ");
                info       = ssyevr(matrix, L, range, il, iu, vl, vu);
            }
            if constexpr(form == Form::NSYM) throw std::logic_error("?sygvr not implemented");
        } else if constexpr(std::is_same_v<Scalar, fp64>) {
            eig_init(form, Type::FP64, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack dsyevr", config.tag, config.tag.empty() ? "" : " ");
                info       = dsyevr(matrix, L, range, il, iu, vl, vu);
            }
            if constexpr(form == Form::NSYM) throw std::logic_error("?sygvr not implemented");
        } else if constexpr(std::is_same_v<Scalar, cx32>) {
            eig_init(form, Type::CX32, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack cheevr", config.tag, config.tag.empty() ? "" : " ");
                info       = cheevr(matrix, L, range, il, iu, vl, vu);
            }
            if constexpr(form == Form::NSYM) throw std::logic_error("?sygvx not implemented");
        } else if constexpr(std::is_same_v<Scalar, cx64>) {
            eig_init(form, Type::CX64, compute_eigvecs_, remove_phase_);
            if constexpr(form == Form::SYMM) {
                config.tag = fmt::format("{}{}lapack zheevr", config.tag, config.tag.empty() ? "" : " ");
                info       = zheevr(matrix, L, range, il, iu, vl, vu);
            }
            if constexpr(form == Form::NSYM) throw std::logic_error("?sygvx not implemented");
        } else {
            throw except::runtime_error("Unknown type");
        }

    } catch(std::exception &ex) {
        eig::log->error("Eigenvalue solver failed: {}", ex.what());
        throw except::runtime_error("Eigenvalue solver Failed: {}", ex.what());
    }
    result.build_eigvals_cx32();
    result.build_eigvecs_cx32();
    result.build_eigvals_cx64();
    result.build_eigvecs_cx64();

    if(info == 0 and config.remove_phase and config.remove_phase.value() == Dephase::OFF) {
        // The solution to  the eigenvalue equation Av = l*v is determined up to a constant phase factor, i.e., if v
        // is a solution, so is v*exp(i*theta). By computing the complex angle of the first element in v, one can then
        // remove it from all other elements of v.
        subtract_phase(result.eigvecsL_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsL_cx64, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx64, L, result.meta.nev);
        subtract_phase(result.eigvecsL_cx128, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx128, L, result.meta.nev);
    }
}
template void eig::solver::eig<eig::Form::SYMM>(fp32 *matrix, size_type, char, int, int, fp32, fp32, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp64 *matrix, size_type, char, int, int, fp64, fp64, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp128 *matrix, size_type, char, int, int, fp128, fp128, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx32 *matrix, size_type, char, int, int, fp32, fp32, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx64 *matrix, size_type, char, int, int, fp64, fp64, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx128 *matrix, size_type, char, int, int, fp128, fp128, Vecs, Dephase);

template<eig::Form form, typename Scalar, typename RealScalar>
void eig::solver::eig(Scalar *matrixA, Scalar *matrixB, size_type L, char range, int il, int iu, RealScalar vl, RealScalar vu, Vecs compute_eigvecs_,
                      Dephase remove_phase_) {
    static_assert(form == Form::SYMM);
    static_assert(!std::is_const_v<Scalar>);
    auto t_eig = tid::tic_scope("eig");
    int  info  = 0;
    try {
        if(matrixB == nullptr) {
            if constexpr(std::is_same_v<Scalar, fp32>) {
                eig_init(form, Type::FP32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack ssyevr", config.tag, config.tag.empty() ? "" : " ");
                    info       = ssyevr(matrixA, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvr not implemented");
            } else if constexpr(std::is_same_v<Scalar, fp64>) {
                eig_init(form, Type::FP64, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack dsyevr", config.tag, config.tag.empty() ? "" : " ");
                    info       = dsyevr(matrixA, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvr not implemented");
            } else if constexpr(std::is_same_v<Scalar, cx32>) {
                eig_init(form, Type::CX32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack cheevr", config.tag, config.tag.empty() ? "" : " ");
                    info       = cheevr(matrixA, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvx not implemented");
            } else if constexpr(std::is_same_v<Scalar, cx64>) {
                eig_init(form, Type::CX64, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack zheevr", config.tag, config.tag.empty() ? "" : " ");
                    info       = zheevr(matrixA, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvx not implemented");
            } else {
                throw except::runtime_error("Unknown type");
            }
        } else {
            if constexpr(std::is_same_v<Scalar, fp32>) {
                eig_init(form, Type::FP32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack ssygvx", config.tag, config.tag.empty() ? "" : " ");
                    info       = ssygvx(matrixA, matrixB, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvr not implemented");
            }
            if constexpr(std::is_same_v<Scalar, fp64>) {
                eig_init(form, Type::FP32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack dsygvx", config.tag, config.tag.empty() ? "" : " ");
                    info       = dsygvx(matrixA, matrixB, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvr not implemented");
            } else if constexpr(std::is_same_v<Scalar, cx32>) {
                eig_init(form, Type::CX32, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack chegvx", config.tag, config.tag.empty() ? "" : " ");
                    throw except::logic_error("The generalized solvers have not been implemented (chegvx)");
                    // info = zheevx(matrix, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvx not implemented");
            } else if constexpr(std::is_same_v<Scalar, cx64>) {
                eig_init(form, Type::CX64, compute_eigvecs_, remove_phase_);
                if constexpr(form == Form::SYMM) {
                    config.tag = fmt::format("{}{}lapack zhegvx", config.tag, config.tag.empty() ? "" : " ");
                    throw except::logic_error("The generalized solvers have not been implemented (zhegvx)");
                    // info = zheevx(matrix, L, range, il, iu, vl, vu);
                }
                if constexpr(form == Form::NSYM) throw std::logic_error("?sygvx not implemented");
            } else {
                throw except::runtime_error("Unknown type");
            }
        }

    } catch(std::exception &ex) {
        eig::log->error("Eigenvalue solver failed: {}", ex.what());
        throw except::runtime_error("Eigenvalue solver Failed: {}", ex.what());
    }
    result.build_eigvals_cx32();
    result.build_eigvecs_cx32();
    result.build_eigvals_cx64();
    result.build_eigvecs_cx64();
    result.build_eigvals_cx128();
    result.build_eigvecs_cx128();

    if(info == 0 and config.remove_phase and config.remove_phase.value() == Dephase::OFF) {
        // The solution to  the eigenvalue equation Av = l*v is determined up to a constant phase factor, i.e., if v
        // is a solution, so is v*exp(i*theta). By computing the complex angle of the first element in v, one can then
        // remove it from all other elements of v.
        subtract_phase(result.eigvecsL_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx32, L, result.meta.nev);
        subtract_phase(result.eigvecsL_cx64, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx64, L, result.meta.nev);
        subtract_phase(result.eigvecsL_cx128, L, result.meta.nev);
        subtract_phase(result.eigvecsR_cx128, L, result.meta.nev);
    }
}
template void eig::solver::eig<eig::Form::SYMM>(fp32 *matrixA, fp32 *matrixB, size_type, char, int, int, fp32, fp32, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp64 *matrixA, fp64 *matrixB, size_type, char, int, int, fp64, fp64, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(fp128 *matrixA, fp128 *matrixB, size_type, char, int, int, fp128, fp128, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx32 *matrixA, cx32 *matrixB, size_type, char, int, int, fp32, fp32, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx64 *matrixA, cx64 *matrixB, size_type, char, int, int, fp64, fp64, Vecs, Dephase);
template void eig::solver::eig<eig::Form::SYMM>(cx128 *matrixA, cx128 *matrixB, size_type, char, int, int, fp128, fp128, Vecs, Dephase);

template<typename Scalar>
void eig::solver::eigs_init(size_type L, int nev, int ncv, Ritz ritz, Form form, Type type, Side side, std::optional<cx64> sigma, Shinv shift_invert,
                            Storage storage, Vecs compute_eigvecs, Dephase remove_phase, Scalar *residual, Lib lib) {
    if(config.loglevel) eig::setLevel(config.loglevel.value());
    eig::log->trace("eigs init");
    result.reset();
    config.lib = config.lib.value_or(lib);

    // Precision settings which are overridden manually
    config.tol     = config.tol.value_or(1e-12);
    config.maxIter = config.maxIter.value_or(1000);

    // Other settings that we pass on each invocation of eigs
    config.compute_eigvecs = config.compute_eigvecs.value_or(compute_eigvecs);
    config.remove_phase    = config.remove_phase.value_or(remove_phase);
    config.maxNev          = config.maxNev.value_or(nev);
    config.maxNcv          = config.maxNcv.value_or(ncv);
    config.sigma           = config.sigma.value_or(sigma.value_or(cx64(0.0, 0.0)));
    config.shift_invert    = config.shift_invert.value_or(shift_invert);
    config.type            = config.type.value_or(type);
    config.form            = config.form.value_or(form);
    config.ritz            = config.ritz.value_or(ritz);
    config.side            = config.side.value_or(side);
    config.storage         = config.storage.value_or(storage);

    if(config.initial_guess.empty()) config.initial_guess.push_back({residual, 0});
    config.maxNev.value() = std::clamp(config.maxNev.value(), 1, safe_cast<int>(L));
    if(config.form == Form::NSYM) {
        if(config.maxNev.value() == 1) { config.maxNev = 2; }
    }

    config.maxNcv = getBasisSize(L, config.maxNev.value(), config.maxNcv); // Adjust ncv if <= 0 (autoselect) <= 2nev(clamp) or > L (clamp)
    assert(1 <= config.maxNev.value() and config.maxNev.value() <= L);
    assert(config.maxNev <= config.maxNcv.value() and config.maxNcv.value() <= L);

    if(config.shift_invert == Shinv::ON and not config.sigma) throw std::runtime_error("Sigma must be set to use shift-invert mode");
    config.checkRitz();

    if(not config.logTime) {
        if(eig::log->level() == spdlog::level::trace) config.logTime = 10.0;
        if(eig::log->level() == spdlog::level::debug) config.logTime = 60.0;
        if(eig::log->level() >= spdlog::level::info) config.logTime = 60.0 * 10;
    }
    if(not config.logIter) {
        if(eig::log->level() == spdlog::level::trace) config.logIter = 100;
        if(eig::log->level() == spdlog::level::debug) config.logIter = 1000;
        if(eig::log->level() >= spdlog::level::info) config.logIter = 5000;
    }
}
/* clang-format off */
template void eig::solver::eigs_init(size_type L, int nev, int ncv, Ritz ritz, Form form, Type type, Side side, std::optional<cx64> sigma, Shinv shift_invert, Storage storage, Vecs compute_eigvecs, Dephase remove_phase, fp32 *residual, Lib lib);
template void eig::solver::eigs_init(size_type L, int nev, int ncv, Ritz ritz, Form form, Type type, Side side, std::optional<cx64> sigma, Shinv shift_invert, Storage storage, Vecs compute_eigvecs, Dephase remove_phase, fp64 *residual, Lib lib);
template void eig::solver::eigs_init(size_type L, int nev, int ncv, Ritz ritz, Form form, Type type, Side side, std::optional<cx64> sigma, Shinv shift_invert, Storage storage, Vecs compute_eigvecs, Dephase remove_phase, fp128 *residual, Lib lib);
template void eig::solver::eigs_init(size_type L, int nev, int ncv, Ritz ritz, Form form, Type type, Side side, std::optional<cx64> sigma, Shinv shift_invert, Storage storage, Vecs compute_eigvecs, Dephase remove_phase, cx32 *residual, Lib lib);
template void eig::solver::eigs_init(size_type L, int nev, int ncv, Ritz ritz, Form form, Type type, Side side, std::optional<cx64> sigma, Shinv shift_invert, Storage storage, Vecs compute_eigvecs, Dephase remove_phase, cx64 *residual, Lib lib);
template void eig::solver::eigs_init(size_type L, int nev, int ncv, Ritz ritz, Form form, Type type, Side side, std::optional<cx64> sigma, Shinv shift_invert, Storage storage, Vecs compute_eigvecs, Dephase remove_phase, cx128 *residual, Lib lib);
/* clang-format on */

template<typename MatrixProductType>
void eig::solver::set_default_config(const MatrixProductType &matrix) {
    if(config.loglevel) eig::setLevel(config.loglevel.value());
    eig::log->trace("eigs init");

    if(not config.lib) config.lib = eig::Lib::PRIMME;

    // Precision settings which are overridden manually
    config.tol     = config.tol.value_or(1e-12);
    config.maxIter = config.maxIter.value_or(2000);

    // Other settings that we pass on each invocation of eigs
    config.compute_eigvecs = config.compute_eigvecs.value_or(eig::Vecs::OFF);
    config.remove_phase    = config.remove_phase.value_or(eig::Dephase::OFF);
    config.maxNev          = config.maxNev.value_or(1);
    config.maxNcv          = config.maxNcv.value_or(-1);
    config.shift_invert    = config.shift_invert.value_or(Shinv::OFF);
    config.type            = config.type.value_or(matrix.get_type());
    config.form            = config.form.value_or(matrix.get_form());
    config.side            = config.side.value_or(matrix.get_side());
    config.ritz            = config.ritz.value_or(eig::Ritz::SA);
    config.storage         = config.storage.value_or(MatrixProductType::storage);

    config.maxNev.value() = std::clamp(config.maxNev.value(), 1, safe_cast<int>(matrix.rows()));
    if(config.form == Form::NSYM) {
        if(config.maxNev.value() == 1) { config.maxNev = 2; }
    }

    config.maxNcv = getBasisSize(matrix.rows(), config.maxNev.value(), config.maxNcv); // Adjust ncv if <= 0 (autoselect) <= 2nev(clamp) or > L (clamp)
    assert(1 <= config.maxNev.value() and config.maxNev.value() <= matrix.rows());
    assert(config.maxNev <= config.maxNcv.value() and config.maxNcv.value() <= matrix.rows());

    if(config.shift_invert == Shinv::ON and not config.sigma) throw std::runtime_error("Sigma must be set to use shift-invert mode");
    config.checkRitz();

    if(not config.logTime) {
        if(eig::log->level() == spdlog::level::trace) config.logTime = 10.0;
        if(eig::log->level() == spdlog::level::debug) config.logTime = 60.0;
        if(eig::log->level() >= spdlog::level::info) config.logTime = 60.0 * 10;
    }
    if(not config.logIter) {
        if(eig::log->level() == spdlog::level::trace) config.logIter = 100;
        if(eig::log->level() == spdlog::level::debug) config.logIter = 1000;
        if(eig::log->level() == spdlog::level::info) config.logIter = 5000;
    }
}
template void eig::solver::set_default_config(const MatVecMPOS<fp32> &matrix);
template void eig::solver::set_default_config(const MatVecMPOS<fp64> &matrix);
template void eig::solver::set_default_config(const MatVecMPOS<fp128> &matrix);
template void eig::solver::set_default_config(const MatVecMPOS<cx32> &matrix);
template void eig::solver::set_default_config(const MatVecMPOS<cx64> &matrix);
template void eig::solver::set_default_config(const MatVecMPOS<cx128> &matrix);

template void eig::solver::set_default_config(const MatVecMPO<fp32> &matrix);
template void eig::solver::set_default_config(const MatVecMPO<fp64> &matrix);
template void eig::solver::set_default_config(const MatVecMPO<fp128> &matrix);
template void eig::solver::set_default_config(const MatVecMPO<cx32> &matrix);
template void eig::solver::set_default_config(const MatVecMPO<cx64> &matrix);
template void eig::solver::set_default_config(const MatVecMPO<cx128> &matrix);

template void eig::solver::set_default_config(const MatVecDense<fp32> &matrix);
template void eig::solver::set_default_config(const MatVecDense<fp64> &matrix);
template void eig::solver::set_default_config(const MatVecDense<fp128> &matrix);
template void eig::solver::set_default_config(const MatVecDense<cx32> &matrix);
template void eig::solver::set_default_config(const MatVecDense<cx64> &matrix);
template void eig::solver::set_default_config(const MatVecDense<cx128> &matrix);

template void eig::solver::set_default_config(const MatVecSparse<fp32> &matrix);
template void eig::solver::set_default_config(const MatVecSparse<fp64> &matrix);
template void eig::solver::set_default_config(const MatVecSparse<fp128> &matrix);
template void eig::solver::set_default_config(const MatVecSparse<cx32> &matrix);
template void eig::solver::set_default_config(const MatVecSparse<cx64> &matrix);
template void eig::solver::set_default_config(const MatVecSparse<cx128> &matrix);

template void eig::solver::set_default_config(const MatVecZero<fp32> &matrix);
template void eig::solver::set_default_config(const MatVecZero<fp64> &matrix);
template void eig::solver::set_default_config(const MatVecZero<fp128> &matrix);
template void eig::solver::set_default_config(const MatVecZero<cx32> &matrix);
template void eig::solver::set_default_config(const MatVecZero<cx64> &matrix);
template void eig::solver::set_default_config(const MatVecZero<cx128> &matrix);

template<typename Scalar, eig::Storage storage>
void eig::solver::eigs(const Scalar *matrix, size_type L, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert,
                       Vecs compute_eigvecs, Dephase remove_phase, Scalar *residual) {
    auto t_eig = tid::tic_scope("eig");
    static_assert(storage != Storage::MPS and "Can't build MatVecMPO from a pointer");
    if constexpr(storage == Storage::DENSE) {
        auto matrix_dense = MatVecDense<Scalar>(matrix, L, true, form, side);
        eigs(matrix_dense, nev, ncv, ritz, form, side, sigma, shift_invert, compute_eigvecs, remove_phase, residual);
    } else if constexpr(storage == Storage::SPARSE) {
        auto matrix_sparse = MatVecSparse<Scalar, false>(matrix, L, true);
        eigs(matrix_sparse, nev, ncv, ritz, form, side, sigma, shift_invert, compute_eigvecs, remove_phase, residual);
    }
}

/* clang-format off */
template void eig::solver::eigs(const fp32 *matrix, size_type L, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp32 *residual);
template void eig::solver::eigs(const fp64 *matrix, size_type L, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp64 *residual);
template void eig::solver::eigs(const fp128 *matrix, size_type L, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp128 *residual);
template void eig::solver::eigs(const cx32 *matrix, size_type L, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx32 *residual);
template void eig::solver::eigs(const cx64 *matrix, size_type L, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx64 *residual);
template void eig::solver::eigs(const cx128 *matrix, size_type L, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx128 *residual);
/* clang-format on */

template<typename MatrixProductType>
void eig::solver::eigs(MatrixProductType &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert,
                       Vecs compute_eigvecs, Dephase remove_phase, typename MatrixProductType::Scalar *residual) {
    auto t_eig   = tid::tic_scope("eig");
    using Scalar = typename MatrixProductType::Scalar;
    Type type    = eig::ScalarToType<Scalar>();
    eigs_init(matrix.rows(), nev, ncv, ritz, form, type, side, sigma, shift_invert, matrix.storage, compute_eigvecs, remove_phase, residual);
    switch(config.lib.value()) {
        case Lib::ARPACK: {
            if constexpr(sfinae::is_any_v<Scalar, fp32, fp64, cx32, cx64>) {
                solver_arpack<MatrixProductType> solver(matrix, config, result);
                solver.eigs();
                break;
            } else {
                eig::log->warn("Type {} has not been implemented for ARPACK. Trying with with another solver ...", sfinae::type_name<Scalar>());
                [[fallthrough]];
            }
        }
        case Lib::PRIMME: {
            if constexpr(sfinae::is_any_v<Scalar, fp32, fp64, cx32, cx64>) {
                eigs_primme(matrix);
                break;
            } else {
                eig::log->warn("Type {} has not been implemented for PRIMME. Trying with with another solver ...", sfinae::type_name<Scalar>());
                [[fallthrough]];
            }
        }
        case Lib::SPECTRA: {
            solver_spectra<MatrixProductType> solver(matrix, config, result);
            solver.eigs();
            break;
        }
    }
}
/* clang-format off */
template void eig::solver::eigs(MatVecMPOS<fp32> &matrix,  int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp32 *residual);
template void eig::solver::eigs(MatVecMPOS<fp64> &matrix,  int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp64 *residual);
template void eig::solver::eigs(MatVecMPOS<fp128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp128 *residual);
template void eig::solver::eigs(MatVecMPOS<cx32> &matrix,  int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx32 *residual);
template void eig::solver::eigs(MatVecMPOS<cx64> &matrix,  int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx64 *residual);
template void eig::solver::eigs(MatVecMPOS<cx128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx128 *residual);

template void eig::solver::eigs(MatVecMPO<fp32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp32 *residual);
template void eig::solver::eigs(MatVecMPO<fp64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp64 *residual);
template void eig::solver::eigs(MatVecMPO<fp128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp128 *residual);
template void eig::solver::eigs(MatVecMPO<cx32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx32 *residual);
template void eig::solver::eigs(MatVecMPO<cx64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx64 *residual);
template void eig::solver::eigs(MatVecMPO<cx128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx128 *residual);

template void eig::solver::eigs(MatVecDense<fp32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp32 *residual);
template void eig::solver::eigs(MatVecDense<fp64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp64 *residual);
template void eig::solver::eigs(MatVecDense<fp128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp128 *residual);
template void eig::solver::eigs(MatVecDense<cx32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx32 *residual);
template void eig::solver::eigs(MatVecDense<cx64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx64 *residual);
template void eig::solver::eigs(MatVecDense<cx128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx128 *residual);

template void eig::solver::eigs(MatVecSparse<fp32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp32 *residual);
template void eig::solver::eigs(MatVecSparse<fp64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp64 *residual);
template void eig::solver::eigs(MatVecSparse<fp128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp128 *residual);
template void eig::solver::eigs(MatVecSparse<cx32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx32 *residual);
template void eig::solver::eigs(MatVecSparse<cx64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx64 *residual);
template void eig::solver::eigs(MatVecSparse<cx128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx128 *residual);

template void eig::solver::eigs(MatVecZero<fp32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp32 *residual);
template void eig::solver::eigs(MatVecZero<fp64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp64 *residual);
template void eig::solver::eigs(MatVecZero<fp128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, fp128 *residual);
template void eig::solver::eigs(MatVecZero<cx32> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx32 *residual);
template void eig::solver::eigs(MatVecZero<cx64> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx64 *residual);
template void eig::solver::eigs(MatVecZero<cx128> &matrix, int nev, int ncv, Ritz ritz, Form form, Side side, std::optional<cx64> sigma, Shinv shift_invert, Vecs compute_eigvecs, Dephase remove_phase, cx128 *residual);
/* clang-format on */

template<typename MatrixProductType>
void eig::solver::eigs(MatrixProductType &matrix) {
    auto t_eig   = tid::tic_scope("eig");
    using Scalar = typename MatrixProductType::Scalar;
    result.reset();
    set_default_config(matrix);
    switch(config.lib.value()) {
        case Lib::ARPACK: {
            if constexpr(sfinae::is_any_v<Scalar, fp32, fp64, cx32, cx64>) {
                config.tag = fmt::format("{}{}arpack", config.tag, config.tag.empty() ? "" : " ");
                solver_arpack<MatrixProductType> solver(matrix, config, result);
                solver.eigs();
                break;
            } else {
                eig::log->warn("Type {} has not been implemented for ARPACK. Trying with with another solver ...", sfinae::type_name<Scalar>());
                [[fallthrough]];
            }
        }
        case Lib::PRIMME: {
            if constexpr(sfinae::is_any_v<Scalar, fp32, fp64, cx32, cx64>) {
                config.tag = fmt::format("{}{}primme", config.tag, config.tag.empty() ? "" : " ");
                eigs_primme(matrix);
                break;
            } else {
                eig::log->warn("Type {} has not been implemented for PRIMME. Trying with with another solver ...", sfinae::type_name<Scalar>());
                [[fallthrough]];
            }
        }
        case Lib::SPECTRA: {
            config.tag = fmt::format("{}{}spectra", config.tag, config.tag.empty() ? "" : " ");
            solver_spectra<MatrixProductType> solver(matrix, config, result);
            solver.eigs();
            break;
        }
    }
}

template void eig::solver::eigs(MatVecMPOS<fp32> &matrix);
template void eig::solver::eigs(MatVecMPOS<fp64> &matrix);
template void eig::solver::eigs(MatVecMPOS<fp128> &matrix);
template void eig::solver::eigs(MatVecMPOS<cx32> &matrix);
template void eig::solver::eigs(MatVecMPOS<cx64> &matrix);
template void eig::solver::eigs(MatVecMPOS<cx128> &matrix);

template void eig::solver::eigs(MatVecMPO<fp32> &matrix);
template void eig::solver::eigs(MatVecMPO<fp64> &matrix);
template void eig::solver::eigs(MatVecMPO<fp128> &matrix);
template void eig::solver::eigs(MatVecMPO<cx32> &matrix);
template void eig::solver::eigs(MatVecMPO<cx64> &matrix);
template void eig::solver::eigs(MatVecMPO<cx128> &matrix);

template void eig::solver::eigs(MatVecDense<fp32> &matrix);
template void eig::solver::eigs(MatVecDense<fp64> &matrix);
template void eig::solver::eigs(MatVecDense<fp128> &matrix);
template void eig::solver::eigs(MatVecDense<cx32> &matrix);
template void eig::solver::eigs(MatVecDense<cx64> &matrix);
template void eig::solver::eigs(MatVecDense<cx128> &matrix);

template void eig::solver::eigs(MatVecSparse<fp32> &matrix);
template void eig::solver::eigs(MatVecSparse<fp64> &matrix);
template void eig::solver::eigs(MatVecSparse<fp128> &matrix);
template void eig::solver::eigs(MatVecSparse<cx32> &matrix);
template void eig::solver::eigs(MatVecSparse<cx64> &matrix);
template void eig::solver::eigs(MatVecSparse<cx128> &matrix);

template void eig::solver::eigs(MatVecZero<fp32> &matrix);
template void eig::solver::eigs(MatVecZero<fp64> &matrix);
template void eig::solver::eigs(MatVecZero<fp128> &matrix);
template void eig::solver::eigs(MatVecZero<cx32> &matrix);
template void eig::solver::eigs(MatVecZero<cx64> &matrix);
template void eig::solver::eigs(MatVecZero<cx128> &matrix);