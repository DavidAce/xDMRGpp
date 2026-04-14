#include "runner.h"
#include "unit_ids.h"

#include <string>
#include <vector>

namespace fmtab {
    #define FMTAB_DECLARE_STD_UNIT(i) std::string std_unit_##i();
    FMTAB_FOR_EACH_UNIT(FMTAB_DECLARE_STD_UNIT)
    #undef FMTAB_DECLARE_STD_UNIT

    std::vector<std::string> collect_std_outputs() {
        auto outputs = std::vector<std::string>{};
        outputs.reserve(16);
        #define FMTAB_PUSH_STD_UNIT(i) outputs.push_back(std_unit_##i());
        FMTAB_FOR_EACH_UNIT(FMTAB_PUSH_STD_UNIT)
        #undef FMTAB_PUSH_STD_UNIT
        return outputs;
    }
}
