#pragma once

#include <iosfwd>
#include <span>
#include <string>

namespace rvdisasm {
    namespace color {
        constexpr std::string_view white = "\033[37m";
        constexpr std::string_view instr = "\033[38;5;186m";
        constexpr std::string_view reg = "\033[38;5;117m";
        constexpr std::string_view imm = "\033[38;5;114m";
        constexpr std::string_view extra = "\033[38;5;114m";
        constexpr std::string_view index = "\033[38;5;94m";
    }

    std::ostream& disassemble(std::ostream& os, std::span<uint32_t> buf, uint64_t start_addr = 0x0);

    [[nodiscard]] std::string instruction(uint32_t instr, bool pad = false);
}
