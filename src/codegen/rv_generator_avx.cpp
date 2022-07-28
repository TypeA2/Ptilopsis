#include "codegen/rv_generator_avx.hpp"

#include <iostream>
#include <bitset>

#include <magic_enum.hpp>

#include "simd.hpp"

void rv_generator_avx::preprocess() {
    using enum rv_node_type;

    //std::cout << node_data << '\n';
    //std::cout << result_types << '\n';

    /* Work in steps of 8 32-bit elements at a time */
    for (size_t i = 0; i < (node_types.size() / 8); ++i) {
        /* Retrieve all func_arg and func_call_arg nodes */
        __m256i types = _mm256_load_si256(node_types.m256i() + i);
        auto orig_types = types;
        
        const __m256i func_arg_mask = _mm256_set1_epi32(magic_enum::enum_integer(func_arg));
        const __m256i types_masked = _mm256_and_si256(types, func_arg_mask);
        __m256i func_args = _mm256_cmpeq_epi32(types_masked, func_arg_mask);

        /* Only continue if func args are found */
        if (_mm256_testz_si256(func_args, func_args) != 1) {
            /* Retrieve all float nodes */
            __m256i return_types = _mm256_load_si256(result_types.m256i() + i);

            /* All nodes that are either FLOAT or FLOAT_REF */
            __m256i float_func_args = _mm256_cmpeq_epi32(return_types, _mm256_set1_epi32(magic_enum::enum_integer(rv_data_type::FLOAT)));
            float_func_args = _mm256_or_si256(float_func_args, _mm256_cmpeq_epi32(return_types, _mm256_set1_epi32(magic_enum::enum_integer(rv_data_type::FLOAT_REF))));
            float_func_args = _mm256_and_si256(float_func_args, func_args);

            /* All func args that are not float*/
            __m256i int_func_args = _mm256_and_si256(func_args, _mm256_xor_epi32(float_func_args, _mm256_set1_epi32(-1)));

            /* Node data = argument index of this type */
            __m256i node_data = _mm256_load_si256(this->node_data.m256i() + i);

            /* Child idx = argument index */
            __m256i child_idx = _mm256_load_si256(this->child_idx.m256i() + i);

            {
                /* All node data for float func args = arg index among floats */
                __m256i float_data = _mm256_and_si256(node_data, float_func_args);

                /* Mask of all float data larger than 7, so passed outside of a float register */
                __m256i comp = _mm256_cmpgt_epi32(float_data, _mm256_set1_epi32(7));

                /* Node data for all float args not passed in a float register, effectively the number of float args
                 *   Note that non-float-arg elements are calculated too, but these are just ignored at a later moment
                 */
                __m256i int_args = _mm256_sub_epi32(child_idx, float_data);

                /* Calculate reg_offset = num_int_args + num_float_args - 8 */
                float_data = _mm256_sub_epi32(float_data, _mm256_set1_epi32(8));
                __m256i reg_offset = _mm256_add_epi32(int_args, float_data);

                /* on_stack_mask = all reg_offsets >= 8 -> passed on stack */
                __m256i on_stack_mask = _mm256_cmpgt_epi32(reg_offset, _mm256_set1_epi32(7));
                /* as_int_mask = all reg_offsets < 8 and originally >8 -> passed in integer register */
                __m256i as_int_mask = _mm256_and_epi32(comp, _mm256_xor_epi32(on_stack_mask, _mm256_set1_epi32(-1)));

                /* All nodes with reg_offset >= 8 go on the stack, so set bit 2 */
                types = _mm256_or_si256(types, _mm256_and_si256(on_stack_mask, _mm256_set1_epi32(0b10)));
                /* All nodes with reg_offset < 8 but node_data >= 8 go into integer registers, set lowest bit */
                types = _mm256_or_si256(types, _mm256_and_si256(as_int_mask, _mm256_set1_epi32(1)));

                /* node_data = reg_offset - 8 for the nodes on the stack */
                float_data = _mm256_sub_epi32(float_data, _mm256_and_si256(on_stack_mask, _mm256_set1_epi32(8)));
                /* node_data = reg_offset for the nodes in integer registers */

                /* Write the stack nodes */
                node_data = _mm256_blendv_epi8(node_data, float_data, on_stack_mask);
                /* And the float-as-int nodes */
                node_data = _mm256_blendv_epi8(node_data, reg_offset, as_int_mask);
            }

            {
                /* All node data that are func args but not floats */
                __m256i int_data = _mm256_and_si256(node_data, int_func_args);

                /* Number of args of the other type -> float args */
                __m256i float_args = _mm256_sub_epi32(child_idx, int_data);

                /* If there's more than 8 float args already, some integer registers will be used for floats */
                __m256i float_args_count_mask = _mm256_cmpgt_epi32(float_args, _mm256_set1_epi32(7));

                /* Absolute register offset */
                __m256i reg_offset = _mm256_and_si256(float_args_count_mask, _mm256_sub_epi32(float_args, _mm256_set1_epi32(8)));
                reg_offset = _mm256_add_epi32(reg_offset, int_data);

                /* Offsets of 8 and above are put on the stack*/
                __m256i on_stack_mask = _mm256_cmpgt_epi32(reg_offset, _mm256_set1_epi32(7));

                /* On the stack, so set 2nd bit */
                types = _mm256_or_si256(types, _mm256_and_si256(on_stack_mask, _mm256_set1_epi32(0b10)));

                /* On-stack, subtract 8 from offset to get stack offset */
                int_data = _mm256_sub_epi32(int_data, _mm256_and_si256(on_stack_mask, _mm256_set1_epi32(8)));

                /* Write node data for nodes on stack */
                node_data = _mm256_blendv_epi8(node_data, int_data, on_stack_mask);

            }

            /* Write types and node data back to buffer */
            _mm256_store_si256(this->node_types.m256i() + i, types);
            _mm256_store_si256(this->node_data.m256i() + i, node_data);
        }
    }
}
