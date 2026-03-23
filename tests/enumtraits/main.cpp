#include "demo_enums.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace test::enum_support;
    using namespace test::enumtraits_demo;

    const auto algo = from_string<AlgorithmType>("xDMRG");
    assert(algo.has_value());
    assert(algo.value() == AlgorithmType::xDMRG);
    assert(to_string(algo.value()) == "xDMRG");

    const auto compression = from_string<MpoCompress>("DPL");
    assert(compression.has_value());
    assert(doc_of(compression.value()) == "Deparallelization: removes parallel columns/rows from each mpo");

    const auto parsed_flags = parse_flags<ProjectionPolicy>("INIT | STUCK | CONVERGED");
    assert(parsed_flags.has_value());
    assert(has_flag(parsed_flags.value(), ProjectionPolicy::INIT));
    assert(has_flag(parsed_flags.value(), ProjectionPolicy::STUCK));
    assert(has_flag(parsed_flags.value(), ProjectionPolicy::CONVERGED));
    assert(format_flags(parsed_flags.value()) == "INIT|STUCK|CONVERGED");

    const auto alias = from_string<ProjectionPolicy>("DEFAULT");
    assert(alias.has_value());
    assert(to_string(alias.value()) == "DEFAULT");
    assert(format_flags(alias.value()) == "INIT|STUCK|CONVERGED");

    const auto invalid = parse_flags<ProjectionPolicy>("INIT|UNKNOWN");
    assert(!invalid.has_value());

    std::cout << "enumtraits demo\n";
    std::cout << "  AlgorithmType doc        : " << enum_doc<AlgorithmType>() << '\n';
    std::cout << "  xDMRG                    : " << to_string(algo.value()) << " -> " << doc_of(algo.value()) << '\n';
    std::cout << "  MpoCompress::DPL         : " << to_string(compression.value()) << " -> " << doc_of(compression.value()) << '\n';
    std::cout << "  ProjectionPolicy tokens  : " << format_flags(parsed_flags.value()) << '\n';
    std::cout << "  ProjectionPolicy alias   : exact=" << to_string(alias.value()) << " expanded=" << format_flags(alias.value()) << '\n';
    std::cout << "  Canonical flag spellings :";
    for(const auto name : names<ProjectionPolicy>(true)) std::cout << ' ' << name;
    std::cout << '\n';
    return 0;
}
