#include "codegen/rv_generator_avx.hpp"

#include <iostream>

#include <magic_enum.hpp>

#include "simd.hpp"

void rv_generator_avx::preprocess() {
    using enum rv_node_type;

    /* Work in steps of 8 32-bit elements at a time */
    for (size_t i = 0; i < (node_types.size() / 8); ++i) {
        /* Retrieve all func_arg and func_call_arg nodes */
        __m256i types = _mm256_load_si256(node_types.m256i() + i);
        
        const __m256i func_arg_mask = _mm256_set1_epi32(magic_enum::enum_integer(func_arg));
        types = _mm256_and_epi32(types, func_arg_mask);
        __m256i func_args_mask = _mm256_cmpeq_epi32(types, func_arg_mask);

        /* Only continue if func args are found */
        if (_mm256_testz_si256(func_arg_mask, func_arg_mask) == 0) {
            /* Retrieve all float nodes */
            __m256i return_types = _mm256_load_si256(result_types.m256i() + i);
            /* All nodes that are either FLOAT or FLOAT_REF */
            __m256i floats = _mm256_cmpeq_epi32(return_types, _mm256_set1_epi32(magic_enum::enum_integer(rv_data_type::FLOAT)));
            floats = _mm256_or_si256(floats, _mm256_cmpeq_epi32(return_types, _mm256_set1_epi32(magic_enum::enum_integer(rv_data_type::INT))));

        }

        std::cout << format(func_args_mask) << '\n';
    }
}
