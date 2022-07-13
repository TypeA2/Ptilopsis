#pragma once

#include <iosfwd>
#include <span>
#include <string>

namespace rvdisasm {
    std::ostream& disassemble(std::ostream& os, std::span<uint32_t> buf, uint64_t start_addr = 0x0);

    [[nodiscard]] std::string instruction(uint32_t instr);
}
