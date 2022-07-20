#pragma once

#include <span>
#include <memory>
#include <thread>
#include <ostream>

#include <cstdint>

#include <codegen/datatype.hpp>

#include "rv_nodetype.hpp"
#include "simd.hpp"
#include "avx_buffer.hpp"

class DepthTree;

/* RISC-V machine code generator */
class rv_generator {
    protected:
    size_t nodes;
    size_t max_depth;

    avx_buffer<rv_node_type> node_types;
    avx_buffer<DataType> result_types;
    avx_buffer<int32_t> parents;
    avx_buffer<int32_t> depth;
    avx_buffer<int32_t> child_idx;
    avx_buffer<uint32_t> node_data;

    /* Generated RISC-V instructions, in 32-bit words */
    avx_buffer<uint32_t> instructions;

    /* Per-instruction data */
    avx_buffer<int64_t> rd;
    avx_buffer<int64_t> rs1;
    avx_buffer<int64_t> rs2;
    avx_buffer<uint32_t> jt; /* jump target */

    /* Number of instructions for each node */
    avx_buffer<uint32_t> node_sizes;

    /* Location indices for each node */
    avx_buffer<uint32_t> node_locations;

    /* Start, end and size indices of all functions */
    avx_buffer<uint32_t> func_starts;
    avx_buffer<uint32_t> func_ends;
    avx_buffer<uint32_t> function_sizes;
    avx_buffer<uint32_t> stack_sizes;
    avx_buffer<uint32_t> function_ids;

    /* Which instructions are actually enabled */
    avx_buffer<bool> used_instrs;

    public:

    explicit rv_generator(const DepthTree& tree);

    rv_generator(const rv_generator&) = delete;
    rv_generator& operator=(const rv_generator&) = delete;
    rv_generator(rv_generator&&) noexcept = delete;
    rv_generator& operator=(rv_generator&&) = delete;
    
    virtual ~rv_generator() = default;

    virtual void process() = 0;

    std::ostream& print(std::ostream& os) const;
};

inline std::ostream& operator<<(std::ostream& os, const rv_generator& gen) {
    return gen.print(os);
}


/* Simple single-threaded proof-of-concept */
class rv_generator_st : public rv_generator {
    public:
    using rv_generator::rv_generator;

    void process() override;

    private:
    void preprocess();
    void isn_cnt();
    void isn_gen();
    void optimize();
    void regalloc();
    void fix_func_tab(std::span<int64_t> instr_offsets);
};

