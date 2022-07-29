#include "codegen/rv_generator_avx.hpp"

#include <iostream>
#include <bitset>

#include <magic_enum.hpp>

#include "simd.hpp"

using namespace simd::epi32_operators;

namespace epi32 = simd::epi32;

void rv_generator_avx::preprocess() {
    using enum rv_node_type;

    /* Work in steps of 8 32-bit elements at a time */
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        /* Retrieve all func_arg and func_call_arg nodes */
        __m256i types = _mm256_load_si256(node_types.m256i(i));

        const __m256i func_arg_mask_source = epi32::from_enum(func_arg);
        const __m256i types_masked = types & func_arg_mask_source;
        /* Mask for all nodes that are func args */
        __m256i func_args_mask = (types_masked == func_arg_mask_source);

        /* Only continue if func args are found */
        if (!epi32::is_zero(func_args_mask)) {
            /* Retrieve all float nodes */
            __m256i return_types = epi32::load(result_types.m256i(i));

            /* All nodes that are either FLOAT or FLOAT_REF */
            __m256i float_func_args_mask = (return_types == epi32::from_enum(rv_data_type::FLOAT)) | (return_types == epi32::from_enum(rv_data_type::FLOAT_REF));

            /* Only take the func args node that are float */
            float_func_args_mask = float_func_args_mask & func_args_mask;

            /* All func args that are not float*/
            __m256i int_func_args_mask = func_args_mask & ~float_func_args_mask;

            /* Node data = argument index of this type */
            __m256i node_data = epi32::load(this->node_data.m256i(i));

            /* Child idx = argument index */
            __m256i child_idx = epi32::load(this->child_idx.m256i(i));

            {
                /* All node data for float func args = arg index among floats */
                __m256i float_data = node_data & float_func_args_mask;

                /* Mask of all float data larger than 7, so passed outside of a float register */
                __m256i comp_mask = float_data > 7;

                /* Node data for all float args not passed in a float register, effectively the number of float args
                 *   Note that non-float-arg elements are calculated too, but these are just ignored at a later moment
                 */
                __m256i int_args = child_idx - float_data;

                /* Calculate reg_offset = num_int_args + num_float_args - 8 */
                __m256i reg_offset = int_args + float_data - 8;

                /* on_stack_mask = all reg_offsets >= 8 -> passed on stack */
                __m256i on_stack_mask = (reg_offset > 7) & comp_mask;
                /* as_int_mask = all reg_offsets < 8 and originally >8 -> passed in integer register */
                __m256i as_int_mask = ~on_stack_mask & comp_mask; 

                /* All nodes with reg_offset >= 8 go on the stack, so set bit 2 */
                types = types | (on_stack_mask & 0b10);
                /* All nodes with reg_offset < 8 but node_data >= 8 go into integer registers, set lowest bit */
                types = types | (as_int_mask & 0b01);

                /* node_data = reg_offset - 8 for the nodes on the stack */
                reg_offset = reg_offset - (8_m256i & on_stack_mask);
                /* node_data = reg_offset for all float nodes with node data > 7 */
                node_data = _mm256_blendv_epi8(node_data, reg_offset, comp_mask);
            }

            {
                /* All node data that are func args but not floats */
                __m256i int_data = node_data & int_func_args_mask;

                /* Number of args of the other type -> float args */
                __m256i float_args = child_idx - int_data;

                /* reg_offset = int_data + (float_args > 7 ? (float_args - 8) : float_args) */
                __m256i reg_offset = int_data + (float_args - (8_m256i & (float_args > 7)));

                /* Offsets of 8 and above are put on the stack*/
                __m256i on_stack_mask = (reg_offset > 7);

                /* On the stack, so set 2nd bit */
                types = types | (on_stack_mask & 0b10);
                /* Write node data for nodes on stack */
                reg_offset = reg_offset - (8_m256i & on_stack_mask);
                node_data = _mm256_blendv_epi8(node_data, reg_offset, int_func_args_mask);
            }

            /* Write types and node data back to buffer */
            epi32::store(this->node_types.m256i(i), types);
            epi32::store(this->node_data.m256i(i), node_data);
        }
    }

    /* For all func_call_arg_list, set the node data to the number of stack args
     *   If the last (immediately preceding) argument is a stack arg, this already contains the needed number.
     *   If the last argument is a normal integer node, all arguments are in registers.
     *   If the last argument is a normal float argument, all floats fit in registers, but the int arguments may have overflowed
     */
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        __m256i types = epi32::load(node_types.m256i(i));

        /* NOTE: Pareas has a i > 0 check, but in any valid program for i == 0, this mask will be 0 */
        __m256i func_call_arg_list_mask = (types == epi32::from_enum(func_call_arg_list));

        if (!epi32::is_zero(func_call_arg_list_mask)) {
            /* Unaligned load, use pointer arithmetic to prevent overflows */
            __m256i prev_types = epi32::loadu(&node_types[i * 8] - 1);

            /* Load the previous node data, unaligned */
            __m256i prev_node_data = epi32::loadu(&node_data[i * 8] - 1);

            /* Mask of all stack func call args */
            __m256i func_call_stack_arg_mask = (prev_types == epi32::from_enum(func_call_arg_on_stack));
            if (!epi32::is_zero(func_call_stack_arg_mask)) {
                __m256i adjusted_node_data = prev_node_data + 1;

                /* Store the relevant updated fields */
                epi32::maskstore(node_data.m256i(i), func_call_stack_arg_mask, adjusted_node_data);
            }

            /* Mask all normal func call args that are floats */
            __m256i float_func_call_arg_mask = (prev_types == epi32::from_enum(func_call_arg));
            if (!epi32::is_zero(float_func_call_arg_mask)) {
                __m256i prev_result_types = epi32::loadu(&result_types[i * 8] - 1);

                float_func_call_arg_mask = float_func_call_arg_mask & (prev_result_types == epi32::from_enum(rv_data_type::FLOAT));
                if (!epi32::is_zero(float_func_call_arg_mask)) {
                    __m256i prev_child_idx = epi32::loadu(&child_idx[i * 8] - 1);

                    __m256i int_args = prev_child_idx - prev_node_data;
                    int_args = int_args - 8;
                    int_args = epi32::max(int_args, 0_m256i);

                    /* Store the updated fields */
                    epi32::maskstore(node_data.m256i(i), float_func_call_arg_mask, int_args);
                }
            }
        }
    }

    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        const __m256i iota = epi32::from_value(static_cast<int>(i << 3)) + epi32::from_values(0, 1, 2, 3, 4, 5, 6, 7);

        __m256i parents = epi32::load(this->parents.m256i(i));

        /* All nodes with parents. There is only ever 1 node per program without a parent, so this mask is always nonzero */
        __m256i valid_parents_mask = (parents > -1);
        __m256i parent_types = epi32::gather(node_types.data(), parents & valid_parents_mask);

        /* All nodes of which the parent is a comparison node */
        __m256i parent_eq_expr_mask = ((parent_types & epi32::from_enum(eq_expr)) == epi32::from_enum(eq_expr));

        /* All targeted nodes */
        __m256i result_types = epi32::load(this->result_types.m256i(i));
        result_types = result_types & parent_eq_expr_mask;

        /* The ones that are of data type float */
        __m256i result_types_mask = (result_types == epi32::from_enum(rv_data_type::FLOAT));

        if (!epi32::is_zero(result_types_mask)) {

            /* The parents of the nodes in result_types_mask should be se to data type float
             * AVX512F required for scatter, for now just scalar:
             * https://newbedev.com/what-do-you-do-without-fast-gather-and-scatter-in-avx2-instructions
             */
            AVX_ALIGNED int should_store_mask[8];
            epi32::store(should_store_mask, result_types_mask);

            AVX_ALIGNED uint32_t parent_indices[8];
            epi32::store(parent_indices, parents);

            for (size_t i = 0; i < 8; ++i) {
                if (should_store_mask) {
                    this->result_types[parent_indices[i]] = rv_data_type::FLOAT;
                }
            }
        }
    }
}

void rv_generator_avx::isn_cnt() {
    using enum rv_node_type;

    /* Map node_count onto all nodes */
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        __m256i node_types = epi32::load(this->node_types.m256i(i));
        /* node_types is the in the outer array, so multiply by _mm256_set1_epi32 to get the actual index */
        // TODO maybe pad data_type_array_size to 8 elements and use a shift instead, also maybe pre-scale to 4
        __m256i node_types_indices = node_types * data_type_count;

        __m256i result_types = epi32::load(this->result_types.m256i(i));
        /* result_types is the index in the inner array, so just add to the previously calculated offsets */
        node_types_indices = node_types_indices + result_types;

        /* Use the calculated indices to gather the base node counts */
        __m256i base = epi32::gather(node_size_mapping.data(), node_types_indices);

        /* Offset is calculated and applied to base value */
        __m256i delta = epi32::zero();

        __m256i func_call_arg_list_mask = (node_types == epi32::from_enum(func_call_arg_list));
        if (!epi32::is_zero(func_call_arg_list_mask)) {
            /* Already add 1 for every arg list */
            delta = delta + (1_m256i & func_call_arg_list_mask);

            __m256i prev_node_types = epi32::loadu(&this->node_types[i * 8] - 1);
            
            // TODO is this branch even necessary?
            __m256i func_call_arg_mask = func_call_arg_list_mask | ((prev_node_types & epi32::from_enum(func_call_arg)) == epi32::from_enum(func_call_arg));
            /* In a valid program we're basically guaranteed to get at least 1 here, so skip the possible branch */
            __m256i prev_child_idx = epi32::loadu(&this->child_idx[i * 8] - 1);

            /* Add the child idx + 1 of the previous node (so the last arg) to the func call arg list */
            delta = delta + ((prev_child_idx + 1) & func_call_arg_mask);
        }

        __m256i parents = epi32::load(this->parents.m256i(i));
        __m256i valid_parents_mask = ~(parents == -1);
        __m256i parent_types = epi32::gather(this->node_types.data(), parents & valid_parents_mask);
        __m256i child_idx = epi32::load(this->child_idx.m256i(i));

        /* Add 1 for the conditional nodes of if/if_else/while, and add 1 for the if-branch of an if_else */
        __m256i if_else_mask = (parent_types == epi32::from_enum(if_else_statement));
        __m256i if_statement_conditional_mask = ((child_idx == 0) & (if_else_mask | (parent_types == epi32::from_enum(if_statement))));

        /* All nodes with child_idx == 0 and that are if or if_else */
        delta = delta + (1_m256i & if_statement_conditional_mask);

        __m256i if_else_while_2nd_node_mask = ((child_idx == 1) & (if_else_mask | (parent_types == epi32::from_enum(while_statement))));
        delta = delta + (1_m256i & if_else_while_2nd_node_mask);

        base = base + delta;

        epi32::store(this->node_sizes.m256i(i), base);
    }
}
