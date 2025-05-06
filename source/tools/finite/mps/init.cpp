#include "../mps.h"
bool tools::finite::mps::init::bitfield_is_valid(size_t bitfield) { return bitfield != -1ul and init::used_bitfields.count(bitfield) == 0; }

std::string tools::get_bitfield(size_t nbits, const std::string &pattern, BitOrder bitOrder) {
    if(pattern.empty()) return {};
    std::string bitfield;
    if(pattern.front() == 'b') {
        // We have a bit string pattern
        bitfield = pattern.substr(1, std::string::npos);
    } else if(std::isdigit(pattern.front())) {
        bitfield = fmt::format("{0:0>{1}b}", std::stoull(pattern), nbits);
    } else {
        throw except::runtime_error("Unrecognized initial state pattern: [{}]\n"
                                    "Hint: use a pattern 'b<bitfield>' or give the bitfield as a non-negative integer\n",
                                    pattern);
    }
    if(bitfield.size() != nbits)
        throw except::runtime_error("The parsed pattern gives a bitfield that is shorter than the state length.\n"
                                    "    Pattern         : {}\n"
                                    "    Bitfield        : {} (size {})\n"
                                    "    Number of bits  : {}\n",
                                    pattern, bitfield, bitfield.size(), nbits);
    if(bitOrder == BitOrder::Reverse) { std::reverse(bitfield.begin(), bitfield.end()); }
    return bitfield;
}

std::string tools::get_bitfield(size_t nbits, size_t pattern, BitOrder bitOrder) {
    std::string bitfield;
    bitfield = fmt::format("{0:0>{1}b}", pattern, nbits);

    if(bitfield.size() != nbits)
        throw except::runtime_error("The parsed pattern gives a bitfield that is shorter than the state length.\n"
                                    "    Pattern         : {}\n"
                                    "    Bitfield        : {} (size {})\n"
                                    "    Number of bits  : {}\n",
                                    pattern, bitfield, bitfield.size(), nbits);
    if(bitOrder == BitOrder::Reverse) { std::reverse(bitfield.begin(), bitfield.end()); }
    return bitfield;
}