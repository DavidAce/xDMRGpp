#include "demo_enums_generated.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace test::enum_support;
    using namespace test::enumgen_demo;

    const auto algo = from_string<AlgorithmType>("fLBIT");
    assert(algo.has_value());
    assert(to_string(algo.value()) == "fLBIT");
    assert(doc_of(algo.value()) == "Finite-system l-bit evolution");

    const auto compress = from_string<MpoCompress>("AUTO");
    assert(compress.has_value());
    assert(doc_of(compress.value()) == "Select based on global setting");

    const auto flags = parse_flags<ProjectionPolicy>("INIT | FORCE");
    assert(flags.has_value());
    assert(has_flag(flags.value(), ProjectionPolicy::INIT));
    assert(has_flag(flags.value(), ProjectionPolicy::FORCE));
    assert(format_flags(flags.value()) == "INIT|FORCE");

    const auto alias = from_string<ProjectionPolicy>("DEFAULT");
    assert(alias.has_value());
    assert(to_string(alias.value()) == "DEFAULT");
    assert(format_flags(alias.value()) == "INIT|STUCK|CONVERGED");

    std::cout << "enumgen demo\n";
    std::cout << "  AlgorithmType doc        : " << enum_doc<AlgorithmType>() << '\n';
    std::cout << "  fLBIT                    : " << to_string(algo.value()) << " -> " << doc_of(algo.value()) << '\n';
    std::cout << "  MpoCompress::AUTO        : " << to_string(compress.value()) << " -> " << doc_of(compress.value()) << '\n';
    std::cout << "  ProjectionPolicy tokens  : " << format_flags(flags.value()) << '\n';
    std::cout << "  ProjectionPolicy alias   : exact=" << to_string(alias.value()) << " expanded=" << format_flags(alias.value()) << '\n';
    return 0;
}
