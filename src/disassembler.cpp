#include "disassembler.h"

#include <iostream>

std::ostream& rvdisasm::disassemble(std::ostream& os, std::span<uint32_t> buf, uint64_t start_addr) {
    std::ios_base::fmtflags f{ os.flags() };

    os << std::dec << buf.size() << " instructions, starting at 0x" << std::hex << start_addr << std::dec << '\n';

    os.flags(f);

    return os;
}

