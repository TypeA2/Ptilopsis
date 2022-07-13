#include "disassembler.h"

#include <iostream>
#include <sstream>

std::ostream& rvdisasm::disassemble(std::ostream& os, std::span<uint32_t> buf, uint64_t start_addr) {
    std::ios_base::fmtflags f{ os.flags() };

    os << std::dec << buf.size() << " instructions, starting at 0x" << std::hex << start_addr << std::dec << '\n';

    os.flags(f);

    return os;
}

void decode_u_immediate(std::ostream& os, uint32_t instr) {

}

[[nodiscard]] std::string_view instr_name(uint32_t instr) {
    uint8_t opcode =  instr        & 0b1111111;
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

    return "unknown";
}

std::string rvdisasm::instruction(uint32_t instr) {
    return std::string { instr_name(instr) };
    std::stringstream res;
    uint8_t opcode = instr & 0b1111111;
    switch (opcode) {

    }

    return res.str();
}

