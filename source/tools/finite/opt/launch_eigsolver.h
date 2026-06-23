#pragma once

#include "math/eig/enums.h"
#include "report.h"
#include "tensors/TensorsFinite.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
#include <vector>

namespace tools::finite::opt::internal {

    template<typename CalcType, typename Scalar>
    bool launch_eigsolver_folded_spectrum(eig::Lib lib, const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                           const OptMeta &meta, reports::eigs_log<Scalar> &elog, std::vector<opt_mps<Scalar>> &results);

    template<typename CalcType, typename Scalar>
    bool launch_eigsolver_generalized_shift_invert(eig::Lib lib, const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                                   const OptMeta &meta, reports::eigs_log<Scalar> &elog, std::vector<opt_mps<Scalar>> &results);

}
