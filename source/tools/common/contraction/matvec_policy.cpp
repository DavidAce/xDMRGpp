
#include "matvec_policy.h"
namespace tools::common::contraction::internal {
    MatVecOptions &matvec_options_active() {
        static thread_local MatVecOptions opts{};
        return opts;
    }
}