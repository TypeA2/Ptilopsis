#pragma once

#include <span>
#include <memory>
#include <thread>
#include <ostream>

#include <cstdint>

#include <codegen/datatype.hpp>

#include "rv_nodetype.hpp"
#include "simd.hpp"
#include "swap_buffer.hpp"

class DepthTree;

/* RISC-V machine code generator */
class rv_generator {
    protected:
    size_t nodes;

    avx_buffer<rv_node_type> node_types;
    avx_buffer<DataType> result_types;
    avx_buffer<int32_t> parents;
    avx_buffer<int32_t> depth;
    avx_buffer<int32_t> child_idx;
    avx_buffer<uint32_t> node_data;

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
};

