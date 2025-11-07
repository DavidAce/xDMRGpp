#pragma once
#include <vector>
namespace qm {
    class Gate;
    class SwapGate;

    template<typename GateType>
    class Circuit {
        std::vector<std::vector<GateType>> circuit;
    };

}
