#pragma once

#include <array>

#include "codegen/rv_generator.hpp"

/* SIMD-based generator */
class rv_generator_avx : public rv_generator_st {
    public:
    using rv_generator_st::rv_generator_st;

    [[nodiscard]] std::span<uint32_t> get_instructions() const override;

    private:
    avx_buffer<uint32_t> instr;
    avx_buffer<uint32_t> rd_avx;
    avx_buffer<uint32_t> rs1_avx;
    avx_buffer<uint32_t> rs2_avx;
    avx_buffer<uint32_t> jt_avx;

    avx_buffer<uint32_t> used_instrs_avx;

    void dump_instrs() override;

    void preprocess() override;
    void isn_cnt() override;
    void isn_gen() override;
    void optimize() override;
    void regalloc() override;
    void fix_func_tab_avx(std::span<uint32_t> instr_offsets);
    void fix_jumps() override;
    void postprocess() override;

    void scatter_regalloc_fixup(const std::span<int, 8> valid_mask, const m256i indices, const std::span<int, 8> opwords);
};
