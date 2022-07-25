#include "disassembler.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <ranges>

#include <magic_enum.hpp>

namespace color = rvdisasm::color;

bool g_color_regs = true;

std::ostream& operator<<(std::ostream& os, rvdisasm::rv_register reg) {
    extern bool g_color_regs;

    static constexpr std::array<std::string_view, magic_enum::enum_count<rvdisasm::rv_register>()> abi_names {
        "zero",
        "ra", "sp", "gp", "tp",
        "t0", "t1", "t2",
        "s0", "s1",
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
        "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        "t3", "t4", "t5", "t6"

    };

    os << (g_color_regs ? rvdisasm::color::reg : "") << abi_names[magic_enum::enum_integer(reg)] << (g_color_regs ? rvdisasm::color::white : "");

    return os;
}

std::ostream& operator<<(std::ostream& os, std::optional<rvdisasm::rv_register> reg) {
    return reg.has_value() ? (os << reg.value()) : os;
}

[[nodiscard]] rvdisasm::instruction_type instr_type(uint32_t instr) {
    using enum rvdisasm::instruction_type;
    uint8_t opcode = instr & 0b1111111;
    switch (opcode) {
        case 0b0110111: return u;
        case 0b0010111: return u;
        case 0b1101111: return j;
        case 0b1100111: return i;
        case 0b1100011: return b;
        case 0b0000011: return i;
        case 0b0100011: return s;
        case 0b0010011: return i;
        case 0b0110011: return r;
        case 0b0001111: return i;
        case 0b1110011: return i;
        case 0b0000111: return i;
        case 0b0100111: return s;
        case 0b1000011: return r4;
        case 0b1000111: return r4;
        case 0b1001011: return r4;
        case 0b1001111: return r4;
        case 0b1010011: return r;
    }
    return unknown;
}

[[nodiscard]] std::string_view instr_name(uint32_t instr) {
    uint8_t opcode = instr & 0b1111111;
    uint8_t funct3 = (instr >> 12) & 0b111;
    uint8_t funct7 = (instr >> 25) & 0b1111111;
    switch (opcode) {
        case 0b0110111: return "lui";
        case 0b0010111: return "auipc";
        case 0b1101111: return "jal";
        case 0b1100111: return "jalr";
        case 0b1100011:
            switch (funct3) {
                case 0b000: return "beq";
                case 0b001: return "bne";
                case 0b100: return "blt";
                case 0b101: return "bge";
                case 0b110: return "bltu";
                case 0b111: return "bgeu";
            }
        case 0b0000011:
            switch (funct3) {
                case 0b000: return "lb";
                case 0b001: return "lh";
                case 0b010: return "lw";
                case 0b100: return "lbu";
                case 0b101: return "lhu";
            }
        case 0b0100011:
            switch (funct3) {
                case 0b000: return "sb";
                case 0b001: return "sh";
                case 0b010: return "sw";
            }
        case 0b0010011:
            switch (funct3) {
                case 0b000: return "addi";
                case 0b001: return "slli";
                case 0b010: return "slti";
                case 0b011: return "sltiu";
                case 0b100: return "xori";
                case 0b101:
                    switch (funct7) {
                        case 0b0000000: return "srli";
                        case 0b0100000: return "srai";
                    }
                case 0b110: return "ori";
                case 0b111: return "andi";
            }
        case 0b0110011:
            switch (funct3) {
                case 0b000:
                    switch (funct7) {
                        case 0b0000000: return "add";
                        case 0b0000001: return "mul";
                        case 0b0100000: return "sub";
                    }
                case 0b001:
                    switch (funct7) {
                        case 0: return "sll";
                        case 1: return "mulh";
                    }
                case 0b010:
                    switch (funct7) {
                        case 0: return "slt";
                        case 1: return "mulhsu";
                    }
                case 0b011:
                    switch (funct7) {
                        case 0: return "sltu";
                        case 1: return "mulhu";
                    }
                case 0b100:
                    switch (funct7) {
                        case 0: return "xor";
                        case 1: return "div";
                    }
                case 0b101:
                    switch (funct7) {
                        case 0b0000000: return "srl";
                        case 0b0000001: return "divu";
                        case 0b0100000: return "sra";
                    }
                case 0b110:
                    switch (funct7) {
                        case 0: return "or";
                        case 1: return "rem";
                    }
                case 0b111:
                    switch (funct7) {
                        case 0: return "and";
                        case 1: return "remu";
                    }
            }
        case 0b0001111: return "fence";
        case 0b1110011:
            switch (funct7) {
                case 0: return "ecall";
                case 1: return "ebreak";
            }
        case 0b0000111: return "flw";
        case 0b0100111: return "fsw";
        case 0b1000011: return "fmadd.s";
        case 0b1000111: return "fmsub.s";
        case 0b1001011: return "fnmsub.s";
        case 0b1001111: return "fnmadd.s";
        case 0b1010011:
            switch (funct7) {
                case 0b0000000: return "fadd.s";
                case 0b0000100: return "fsub.s";
                case 0b0001000: return "fmul.s";
                case 0b0001100: return "fdiv.s";
                case 0b0101100: return "fsqrt.s";
                case 0b0010000:
                    switch (funct3) {
                        case 0b000: return "fsgnj.s";
                        case 0b001: return "fsgnjn.s";
                        case 0b010: return "fsgnjx.s";
                    }
                case 0b0010100:
                    switch (funct3) {
                        case 0b000: return "fmin.s";
                        case 0b001: return "fmax.s";
                    }
                case 0b1100000:
                    switch ((instr >> 20) & 0b11111) {
                        case 0: return "fcvt.w.s";
                        case 1: return "fcvt.wu.s";
                    }
                case 0b1110000:
                    switch (funct3) {
                        case 0b000: return "fmv.x.w";
                        case 0b001: return "fclass.s";
                    }
                case 0b1010000:
                    switch (funct3) {
                        case 0b000: return "fle.s";
                        case 0b001: return "flt.s";
                        case 0b010: return "feq.s";
                    }
                case 0b1101000:
                    switch ((instr >> 20) & 0b11111) {
                        case 0: return "fcvt.s.w";
                        case 1: return "fcvt.s.wu";
                    }
                case 0b1111000: return "fmv.w.x";
            }
    }

    return instr ? "unknown" : "(zero)";
}

std::ostream& format_args(std::ostream& os, uint32_t instr, bool color, uint64_t addr) {
    auto flags = os.flags();
    bool old_color_regs = g_color_regs;
    g_color_regs = color;

    os << std::dec;
    auto type = instr_type(instr);

    std::string_view color_imm = color ? color::imm : "";
    std::string_view color_white = color ? color::white : "";

    using namespace rvdisasm;

    switch (type) {
        case instruction_type::r: {
            auto rd = magic_enum::enum_cast<rv_register>((instr >> 7) & 0b11111);
            auto rs1 = magic_enum::enum_cast<rv_register>((instr >> 15) & 0b11111);
            auto rs2 = magic_enum::enum_cast<rv_register>((instr >> 20) & 0b11111);

            os << rd << ", " << rs1 << ", " << rs2;
            break;
        }
        case instruction_type::i: {
            auto rd = magic_enum::enum_cast<rv_register>((instr >> 7) & 0b11111);
            auto rs1 = magic_enum::enum_cast<rv_register>((instr >> 15) & 0b11111);
            uint32_t imm = instr >> 20;
            if (imm >> 11) {
                imm |= 0xFFFFF000;
            }

            auto imm_signed = static_cast<int32_t>(imm);

            switch (instr & 0b1111111) {
                case 0b0000011:
                case 0b0000111:
                case 0b1100111:
                    os << rd << ", " << color_imm << imm_signed << color_white << "(" << rs1 << ")";
                    break;

                default:
                    os << rd << ", " << rs1 << ", " << color_imm << imm_signed << color_white;
            }

            break;
        }
        case instruction_type::s: {
            auto rs1 = magic_enum::enum_cast<rv_register>((instr >> 15) & 0b11111);
            auto rs2 = magic_enum::enum_cast<rv_register>((instr >> 20) & 0b11111);
            uint32_t imm = (instr >> 7) & 0b11111;
            imm |= (instr >> 20) & 0b111111100000;
            if (imm >> 11) {
                imm |= 0xFFFFF000;
            }

            auto imm_signed = static_cast<int32_t>(imm);
            os << rs2 << ", " << color_imm << imm_signed << color_white << "(" << rs1 << ")";
            break;
        }
        case instruction_type::b: {
            auto rs1 = magic_enum::enum_cast<rv_register>((instr >> 15) & 0b11111);
            auto rs2 = magic_enum::enum_cast<rv_register>((instr >> 20) & 0b11111);
            uint32_t imm = (instr >> 7) & 0b11110;
            imm |= (instr >> 20) & 0b1111100000;
            imm |= ((instr >> 7) & 0b1) << 11;
            if (instr >> 31) {
                imm |= 0xFFFFF800;
            }
            auto imm_signed = static_cast<int32_t>(imm);

            os << rs1 << ", " << rs2 << ", " << color_imm << std::hex << std::showbase << (addr + imm_signed) << std::dec << color_white;
            break;
        }
        case instruction_type::u: {
            auto rd = magic_enum::enum_cast<rv_register>((instr >> 7) & 0b11111);
            int32_t signed_imm = instr & 0xFFFFF000;

            os << rd << ", " << color_imm << signed_imm << color_white;
            break;
        }
        case instruction_type::j: {
            auto rd = magic_enum::enum_cast<rv_register>((instr >> 7) & 0b11111);
            uint32_t imm = (instr >> 20) & 0b11111111110;
            imm |= ((instr >> 20) & 0b1) << 11;
            imm |= (instr & 0xFF000);
            if (instr >> 31) {
                imm |= 0xFFF80000;
            }
            auto imm_signed = static_cast<int32_t>(imm);

            os << rd << ", " << color_imm << imm_signed << color_white;
            break;
        }
        case instruction_type::r4: {
            auto rd = magic_enum::enum_cast<rv_register>((instr >> 7) & 0b11111);
            auto rs1 = magic_enum::enum_cast<rv_register>((instr >> 15) & 0b11111);
            auto rs2 = magic_enum::enum_cast<rv_register>((instr >> 20) & 0b11111);
            auto rs3 = magic_enum::enum_cast<rv_register>((instr >> 27) & 0b11111);

            os << rd << ", " << rs1 << ", " << rs2 << ", " << rs3;
            break;
        }
        default:
            break;
    }

    g_color_regs = old_color_regs;
    os.flags(flags);
    return os;
}

std::ostream& rvdisasm::disassemble(std::ostream& os, std::span<uint32_t> buf, uint64_t start_addr, std::span<uint64_t> func_starts) {
    std::ios_base::fmtflags f{ os.flags() };

    size_t words = buf.size();

    size_t digits = static_cast<size_t>(std::ceil(std::log(words) / std::log(16))) + 1;
    size_t count_digits = static_cast<size_t>(std::log10(words)) + 1;

    os << color::extra << std::setfill('0')
       << std::dec << buf.size() << color::white << " instructions, starting at "
       << color::imm << "0x" << std::hex << std::setw(digits) << start_addr << color::white << std::dec <<'\n';

    size_t func_index = 0;

    for (uint32_t instr : buf) {
        if (std::ranges::find(func_starts, (start_addr / 4)) != func_starts.end()) {
            for (size_t i = 0; i < (count_digits + 3 + digits + 2 + 12 + 1); ++i) {
                os << ' ';
            }
            os << "func" << func_index++ << ":\n";
        }
        os  << color::extra << std::dec << std::setw(count_digits) << std::setfill(' ') << std::right << (start_addr / 4) << color::white << "   "
            << std::hex << std::setw(digits) << std::setfill('0') << std::right << start_addr << ": "
            << ' ' << color::extra << std::setw(2) << ((instr >> 24) & 0xFF) << color::white
            << ' ' << color::extra << std::setw(2) << ((instr >> 16) & 0xFF) << color::white
            << ' ' << color::extra << std::setw(2) << ((instr >> 8 ) & 0xFF) << color::white
            << ' ' << color::extra << std::setw(2) << (instr & 0xFF)         << color::white
            << "   " << color::instr << std::setw(9) << std::setfill(' ') << std::left
            << instr_name(instr) << color::white << ' ';

        format_args(os, instr, true, start_addr);

        os << '\n';

        start_addr += 4;
    }

    os.flags(f);

    return os;
}

std::string rvdisasm::instruction(uint32_t instr, bool pad) {
    std::stringstream res;
    if (pad) {
        res << std::setw(9) << std::setfill(' ') << std::left;
    }
    res << instr_name(instr) << ' ';
    format_args(res, instr, false, 0);

    return res.str();
}

