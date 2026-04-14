#include "runner.h"
#include "unit_ids.h"

#include <string>
#include <vector>

namespace fmtab {
    #define FMTAB_DECLARE_WRAP_UNIT(i) std::string wrap_unit_##i();
    FMTAB_FOR_EACH_UNIT(FMTAB_DECLARE_WRAP_UNIT)
    #undef FMTAB_DECLARE_WRAP_UNIT

    std::vector<std::string> collect_wrap_outputs() {
        auto outputs = std::vector<std::string>{};
        outputs.reserve(16);
        #define FMTAB_PUSH_WRAP_UNIT(i) outputs.push_back(wrap_unit_##i());
        FMTAB_FOR_EACH_UNIT(FMTAB_PUSH_WRAP_UNIT)
        #undef FMTAB_PUSH_WRAP_UNIT
        return outputs;
    }
}
