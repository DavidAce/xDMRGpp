#pragma once
#include <vector>
namespace qm {
    template<typename GateType>
    class Circuit {
        std::vector<std::vector<GateType>> circuit;
    };

}
