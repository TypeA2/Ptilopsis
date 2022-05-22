#pragma once

#include <iosfwd>
#include <span>

namespace rvdisasm {
    std::ostream& disassemble(std::ostream& os, std::span<uint32_t> buf, uint64_t start_addr = 0x0);
}
