#include "enums.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace test::enumcppgen_demo;

    const auto algo = sv2enum<AlgorithmType>("xDMRG");
    assert(algo == AlgorithmType::xDMRG);
    assert(enum2sv(algo) == "xDMRG");

    const auto compress = sv2enum<MpoCompress>("DPL");
    assert(compress == MpoCompress::DPL);
    assert(enum2sv(compress) == "DPL");

    const auto policy = sv2enum<ProjectionPolicy>("INIT|STUCK|CONVERGED");
    assert(has_flag(policy, ProjectionPolicy::INIT));
    assert(has_flag(policy, ProjectionPolicy::STUCK));
    assert(has_flag(policy, ProjectionPolicy::CONVERGED));
    assert(flag2sv(policy) == "INIT|STUCK|CONVERGED");

    const auto alias = sv2enum<ProjectionPolicy>("DEFAULT");
    assert(enum2sv(alias) == "DEFAULT");
    assert(flag2sv(alias) == "INIT|STUCK|CONVERGED");

    bool rejected_whitespace = false;
    try {
        static_cast<void>(sv2enum<ProjectionPolicy>("INIT | STUCK | CONVERGED"));
    } catch(const std::runtime_error &) { rejected_whitespace = true; }
    assert(rejected_whitespace);

    std::cout << "enumcppgen demo\n";
    std::cout << "  enum2sv(xDMRG)           : " << enum2sv(algo) << '\n';
    std::cout << "  sv2enum<MpoCompress>     : " << enum2sv(compress) << '\n';
    std::cout << "  sv2enum<ProjectionPolicy>: " << flag2sv(policy) << '\n';
    std::cout << "  alias expansion          : " << flag2sv(alias) << '\n';
    std::cout << "  whitespace rejected      : " << std::boolalpha << rejected_whitespace << '\n';
    return 0;
}
