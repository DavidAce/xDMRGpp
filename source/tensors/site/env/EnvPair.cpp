
#include "EnvPair.h"
#include "EnvEne.h"
#include "EnvVar.h"

template<typename env_type>
void env_pair<env_type>::assert_validity() const {
    if constexpr(requires(const env_type &e) { e.assert_validity(); }) {
        L.assert_validity();
        R.assert_validity();
    }
}
