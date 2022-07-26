#pragma once

#include "codegen/rv_generator.hpp"

/* SIMD-based generator */
class rv_generator_avx : public rv_generator_st {
    public:
    using rv_generator_st::rv_generator_st;

    private:
    void preprocess() override;
};

