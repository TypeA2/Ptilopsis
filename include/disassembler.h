#pragma once

#include <iosfwd>
#include <span>
#include <string>
#include <optional>

namespace rvdisasm {
    namespace color {
        constexpr std::string_view white = "\033[37m";
        constexpr std::string_view instr = "\033[38;5;186m";
        constexpr std::string_view reg = "\033[38;5;117m";
        constexpr std::string_view imm = "\033[38;5;114m";
        constexpr std::string_view extra = "\033[38;5;114m";
        constexpr std::string_view index = "\033[38;5;94m";
    }

    enum class instruction_type {
        unknown, r, i, s, b, u, j, r4,
    };

    enum class rv_register : int16_t {
        x0, x1, x2, x3, x4, x5, x6, x7,
        x8, x9, x10, x11, x12, x13, x14, x15,
        x16, x17, x18, x19, x20, x21, x22, x23,
        x24, x25, x26, x27, x28, x29, x30, x31
    };

    std::ostream& disassemble(std::ostream& os, std::span<uint32_t> buf, uint64_t start_addr = 0x0, std::span<uint64_t> func_starts = {});

    [[nodiscard]] std::string instruction(uint32_t instr, bool pad = false);
}

std::ostream& operator<<(std::ostream& os, rvdisasm::rv_register reg);
std::ostream& operator<<(std::ostream& os, std::optional<rvdisasm::rv_register> reg);
