#pragma once

#include "codegen/rv_generator.hpp"

/* SIMD-based generator */
class rv_generator_avx : public rv_generator_st {
    public:
    using rv_generator_st::rv_generator_st;

    private:
    avx_buffer<uint32_t> instr;
    avx_buffer<uint32_t> rd_avx;
    avx_buffer<uint32_t> rs1_avx;
    avx_buffer<uint32_t> rs2_avx;
    avx_buffer<uint32_t> jt_avx;

    void preprocess() override;
    void isn_cnt() override;
    void isn_gen() override;
};

