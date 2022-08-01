#include "codegen/rv_generator_avx.hpp"

#include <iostream>
#include <bitset>
#include <iomanip>

#include <magic_enum.hpp>

#include "simd.hpp"
#include "disassembler.h"

using namespace simd::epi32_operators;

namespace epi32 = simd::epi32;

void rv_generator_avx::dump_instrs() {
    std::cout << rvdisasm::color::extra << " == " << instr.size() << " instructions ==             rd rs1 rs2 jt\n" << rvdisasm::color::white;
    size_t digits = static_cast<size_t>(std::log10(instr.size()) + 1);
    for (size_t i = 0; i < instr.size(); ++i) {
        std::string instr = rvdisasm::instruction(this->instr[i], true);
        std::cout << rvdisasm::color::extra << std::dec << std::setw(digits) << std::setfill(' ') << i << rvdisasm::color::white << ": " << instr;
        int64_t pad = std::max<int64_t>(0, 32 - static_cast<int64_t>(instr.size()));
        for (int64_t j = 0; j < pad; ++j) {
            std::cout << ' ';
        }

        std::cout << std::setw(2) << std::setfill(' ') << rd_avx[i] << ' '
            << std::setw(2) << std::setfill(' ') << rs1_avx[i] << ' '
            << std::setw(2) << std::setfill(' ') << rs2_avx[i] << ' '
            << std::setw(2) << std::setfill(' ') << jt_avx[i] << '\n';
    }
}

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
                node_data = epi32::blendv(node_data, reg_offset, comp_mask);
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
                node_data = epi32::blendv(node_data, reg_offset, int_func_args_mask);
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
            AVX_ALIGNED auto should_store_mask = epi32::extract(result_types_mask);
            AVX_ALIGNED auto parent_indices = epi32::extract(parents);

            for (size_t j = 0; j < 8; ++j) {
                if (should_store_mask[j]) {
                    this->result_types[parent_indices[j]] = rv_data_type::FLOAT;
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
            __m256i func_call_arg_mask = func_call_arg_list_mask & ((prev_node_types & epi32::from_enum(func_call_arg)) == epi32::from_enum(func_call_arg));
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

    /* Prefix sum from ADMS20_05 */
    
    /* Accumulator */
    __m256i offset = epi32::zero();
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        __m256i node_size = epi32::loadu(&this->node_sizes[i * 8] - 1);

        /* Compute the actual prefix sum */
        node_size = node_size + _mm256_slli_si256_dual<4>(node_size);
        node_size = node_size + _mm256_slli_si256_dual<8>(node_size);
        node_size = node_size + _mm256_slli_si256_dual<16>(node_size);

        node_size = node_size + offset;

        /* Broadcast last value to every value of offset */
        offset = _mm256_permutevar8x32_epi32(node_size, epi32::from_value(7));

        epi32::store(this->node_locations.m256i(i), node_size);
    }

    /* instr_count_fix not needed, we shouldn't be seeing invalid nodes */

    /* instr_count_fix_post */
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        __m256i parents = epi32::load(this->parents.m256i(i));
        __m256i valid_parents_mask = (parents > -1);

        __m256i child_idx = epi32::load(this->child_idx.m256i(i));
        __m256i parent_types = epi32::gather(this->node_types.data(), parents & valid_parents_mask);

        /* All conditional nodes of if and if_else */
        __m256i conditionals_mask = ((parent_types & epi32::from_enum(if_statement)) == epi32::from_enum(if_statement));
        /* Lowest bit represents the conditional node child index, so check if it's equal */
        conditionals_mask = conditionals_mask & (child_idx == (parent_types & 1));

        if (!epi32::is_zero(conditionals_mask)) {
            /* For all conditional nodes, set the offset of the parent to past the current node */
            __m256i indices = epi32::blendv(epi32::from_value(-1), parents, conditionals_mask);
            __m256i values = epi32::load(this->node_locations.m256i(i));

            /* Get base node size for all nodes */
            __m256i node_types_indices = (epi32::load(this->node_types.m256i(i)) * data_type_count) + epi32::load(this->result_types.m256i(i));
            values = values + (epi32::maskgatherz(node_size_mapping.data(), node_types_indices, conditionals_mask));

            AVX_ALIGNED auto should_store_mask = epi32::extract(conditionals_mask);
            AVX_ALIGNED auto fixed_indices = epi32::extract(indices);
            AVX_ALIGNED auto fixed_values = epi32::extract(values);

            for (size_t j = 0; j < 8; ++j) {
                if (should_store_mask[j]) {
                    node_locations[fixed_indices[j]] = fixed_values[j];
                }
            }
        }

        /* Func call args modify their own args, so no complicated stores, just masks */
        // TODO possible double load in this loop
        __m256i func_call_arg_mask = epi32::load(this->node_types.m256i(i));
        
        func_call_arg_mask = (epi32::from_enum(func_call_arg) == (func_call_arg_mask & epi32::from_enum(func_call_arg)));

        if (!epi32::is_zero(func_call_arg_mask)) {
            /* New location is the location of the parent plus the child idx plus 1 */
            __m256i parent_locations = epi32::maskgatherz(this->node_locations.data(), parents, func_call_arg_mask);
            __m256i new_locs = parent_locations + epi32::load(this->child_idx.m256i(i)) + 1;

            epi32::maskstore(this->node_locations.m256i(i), func_call_arg_mask, new_locs);
        }
    }

    /* Function table generation */
    // TODO benchmark 1 vs 2 pass method (aka with vs without reallocations)
    size_t func_count = 0;
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        __m256i node_types = epi32::load(this->node_types.m256i(i));

        /* Count the number of function declarations*/
        AVX_ALIGNED auto func_decl_mask = epi32::extract<uint32_t>(node_types == epi32::from_enum(func_decl));
        for (uint32_t f : func_decl_mask) {
            if (f) {
                func_count += 1;
            }
        }
    }

    avx_buffer<uint32_t> func_decls { func_count };
    auto func_decls_ptr = func_decls.data();
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        __m256i node_types = epi32::load(this->node_types.m256i(i));

        /* Get their indices */
        AVX_ALIGNED auto func_decl_mask = epi32::extract<uint32_t>(node_types == epi32::from_enum(func_decl));
        for (size_t j = 0; j < 8; ++j) {
            if (func_decl_mask[j]) {
                (*func_decls_ptr++) = static_cast<uint32_t>((i * 8) + j);
            }
        }
    }
    
    /* Gather node data from all func decls */
    function_ids = avx_buffer<uint32_t> { func_count };
    func_starts = avx_buffer<uint32_t> { func_count };
    avx_buffer<uint32_t> func_offsets { func_count };

    // TODO load/loadu and store/storeu pairs here, not needed maybe? (data should be cached anyway though)
    for (size_t i = 0; i < func_decls.size_m256i(); ++i) {
        __m256i func_decl_indices = epi32::load(func_decls.m256i(i));

        /* Retrieve the node data of every func decl */
        epi32::store(this->function_ids.m256i(i), epi32::gather(this->node_data.data(), func_decl_indices));

        /* Calculate the offset of each function */
        __m256i offset = 6_m256i + epi32::gather(this->node_locations.data(), func_decl_indices);
        epi32::storeu(&func_starts[i * 8] + 1, offset);
        epi32::store(func_offsets.m256i(i), offset);
    }

    function_sizes = avx_buffer<uint32_t> { func_count };
    for (size_t i = 0; i < func_decls.size_m256i(); ++i) {
        /* Load elements and shifted elements directly */
        __m256i shifted_offsets = epi32::loadu(&func_offsets[i * 8] - 1);
        __m256i offsets = epi32::load(func_offsets.m256i(i));

        epi32::store(function_sizes.m256i(i), offsets - shifted_offsets);
    }

    func_ends = avx_buffer<uint32_t> { func_count };
    for (size_t i = 0; i < func_decls.size_m256i(); ++i) {
        __m256i start = epi32::load(func_starts.m256i(i));
        __m256i size = epi32::load(function_sizes.m256i(i));

        epi32::store(func_ends.m256i(i), start + size);
    }
}

void rv_generator_avx::isn_gen() {
    using enum rv_node_type;

    // TODO how realistic is SIMD'ing this? Sorting and removing takes ~8% of the original isn_gen time
    auto idx_array = avx_buffer<uint32_t>::iota(nodes);
    std::ranges::sort(idx_array, std::ranges::less {}, [this](uint32_t i) {
        return depth[i];
    });

    auto depth_starts = avx_buffer<uint32_t>::iota(nodes);

    auto removed = std::ranges::remove_if(depth_starts, [this, &idx_array](uint32_t i) {
        return !(i == 0 || depth[idx_array[i]] != depth[idx_array[i - 1]]);
    });

    depth_starts.shrink_to(std::distance(depth_starts.begin(), removed.begin()));

    uint32_t word_count = node_locations.back();

    /* Back to AVX... */
    avx_buffer<uint32_t> registers { nodes * parent_idx_per_node };
    instr = avx_buffer<uint32_t> { word_count };
    rd_avx = avx_buffer<uint32_t> { word_count };
    rs1_avx = avx_buffer<uint32_t> { word_count };
    rs2_avx = avx_buffer<uint32_t> { word_count };
    jt_avx = avx_buffer<uint32_t> { word_count };

    for (uint32_t i = 0; i < max_depth + 1; ++i) {
        uint32_t current_depth = static_cast<uint32_t>(max_depth) - i;

        uint32_t start_idx = depth_starts[current_depth];
        uint32_t end_idx = (current_depth == max_depth) ? static_cast<uint32_t>(nodes) : depth_starts[current_depth + 1];

        uint32_t level_size = end_idx - start_idx;
        uint32_t level_steps = (level_size + 1) / 2;

        bool is_uneven = level_size & 1;

        /* Every node can produce up to 4 instructions, we can work on 4 integers at a time, so work in steps of 2 instructions */
        for (uint32_t j = 0; j < level_steps; ++j) {
            bool skip_last = is_uneven && (level_steps == j + 1);
            const uint32_t idx_0 = idx_array[start_idx + (2 * j)];
            const uint32_t idx_1 = skip_last ? idx_0 : idx_array[start_idx + (2 * j) + 1];

            /* lo = idx_0, hi = idx_1 */
            // TODO maybe load twice + broadcast when gathering using these as indices
            const __m256i indices = _mm256_set_m128i(_mm_set1_epi32(idx_1), _mm_set1_epi32(idx_0));
            const __m256i iota = _mm256_broadcastsi128_si256(_mm_set_epi32(3 * data_type_count, 2 * data_type_count, data_type_count, 0));

            // TODO this as a cached load or maybe only perform 2 loads and shuffle?
            const __m256i result_types = epi32::gather(this->result_types.data(), indices);

            // TODO is gather even faster than 2 individual loads here?
            const __m256i node_types = epi32::gather(this->node_types.data(), indices);
            __m256i has_instr_indices = node_types * (data_type_count * 4);
            has_instr_indices = has_instr_indices + iota + result_types;

            // TODO masking SHOULD be faster:
            // Hopefully the gather caches the first 4 elements, but still this would be >1 (likely ~4-5 for L1d) cycles of latency
            // _mm256_or_si256 has a latency of 1 and CPI of 0.33, so should be faster than a cached data access.
            // Question is whether this holds for larger datasets...
            const __m128i mask_true = _mm_set1_epi32(0xFFFFFFFF);
            const __m256i has_inst_load_mask = _mm256_set_m128i(skip_last ? _mm_setzero_si128() : mask_true, mask_true);
            const __m256i has_instr_mask = epi32::maskgatherz(has_instr_mapping.data(), has_instr_indices, has_inst_load_mask);

            /* If has_instr OR if i == 0, propagate data */
            const __m256i propagate_data_mask = (has_instr_mask | epi32::from_values(-1, 0, 0, 0, skip_last ? 0 : -1, 0, 0, 0));
            /* Never 0, since the first value will always be propagated */
            
            const __m256i parents = epi32::maskgatherz(this->parents.data(), indices, propagate_data_mask);
            const __m256i no_parent_mask = (parents == -1);
            const __m256i load_from_parents_mask = (~no_parent_mask & propagate_data_mask);
            const __m256i parent_types = epi32::maskgatherz(this->node_types.data(), parents, load_from_parents_mask);

            /* child_idx value of all non-parentless nodes */
            const __m256i child_idx = epi32::maskgatherz(this->child_idx.data(), indices, load_from_parents_mask);
                
            /* Non-conditional nodes of branching instructions */
            __m256i non_conditionals_mask = ((parent_types & epi32::from_enum(if_statement)) == epi32::from_enum(if_statement));
            non_conditionals_mask = non_conditionals_mask & ~(child_idx == (parent_types & 1));

            /* Mask for which nodes to call parent_arg_idx on, at least on non-conditionals */
            __m256i parent_arg_idx_call_mask = non_conditionals_mask;

            /* Load parent_arg_idx_lookup for parentless or conditional nodes */
            const __m256i parent_arg_idx_lookup_load_mask = (no_parent_mask | ~non_conditionals_mask) & propagate_data_mask;
            if (!epi32::is_zero(parent_arg_idx_lookup_load_mask)) {
                /* Same shape as has_instr_mapping */
                const __m256i calc_type = epi32::maskgatherz(parent_arg_idx_lookup.data(), has_instr_indices, parent_arg_idx_lookup_load_mask);

                /* calc_type == 1 means a sub-call */
                parent_arg_idx_call_mask = parent_arg_idx_call_mask | (calc_type == 1);

                /* If result_type > 1 for calc_type 2, also call */
                const __m256i calc_type_2_mask = (calc_type == 2);
                if (!epi32::is_zero(calc_type_2_mask)) {
                    const __m256i result_types = epi32::maskgatherz(this->result_types.data(), indices, calc_type_2_mask);
                    parent_arg_idx_call_mask = parent_arg_idx_call_mask | (result_types > 1);
                }
            }

            /* For all nodes in parent_arg_idx_call_mask... */
            __m256i parent_arg_idx = ((parents * parent_idx_per_node) + child_idx) & parent_arg_idx_call_mask;
            parent_arg_idx = parent_arg_idx | (epi32::from_value(-1) & ~parent_arg_idx_call_mask);
            
            /* Last argument to get_data_prop_value */
            const __m256i instr_in_buf = epi32::from_values(0, 1, 2, 3, 0, 1, 2, 3) + epi32::maskgatherz(this->node_locations.data(), indices, propagate_data_mask);
            __m256i rd = epi32::zero();

            /* Calculate get_data_prop_value */
            if (!epi32::is_zero(has_instr_mask)) {
                /* Need rd for instructions with register
                 * 
                 * get_output_table also has the same shape as has_instr_mapping
                 * 
                 * Mask ungathered values to -1, so check_has_output_mask doesn't pick them up 
                 */
                const __m256i calc_type = epi32::maskgather(epi32::from_value(-1), get_output_table.data(), has_instr_indices, has_instr_mask);

                rd = 32_m256i & (calc_type == 3);
                rd = rd | (10_m256i & (calc_type == 4));

                // TODO cmpeq has CPI of 0.5 but bitwise AND has CPI of 0.33, maybe use that instead?
                const __m256i calc_type_1_mask = (calc_type == 1);
                const __m256i calc_type_2_mask = (calc_type == 2);
                const __m256i use_node_data_mask = calc_type_1_mask | calc_type_2_mask;
                if (!epi32::is_zero(use_node_data_mask)) {
                    const  __m256i node_data = epi32::maskgatherz(this->node_data.data(), indices, use_node_data_mask);

                    rd = rd | ((node_data + 10) & calc_type_1_mask);
                    rd = rd | ((node_data + 42) & calc_type_2_mask);
                }

                /* Default, use parent_arg_idx and has_output */
                __m256i check_has_output_mask = (calc_type == 0);
                /* Exceedingly unlikely to be zero, so continue */

                /* If there is an output index or explicitly can have an output  */
                __m256i has_output_val_mask = (parent_arg_idx_call_mask & has_instr_mask);
                has_output_val_mask = has_output_val_mask |  epi32::maskgatherz(has_output.data(), has_instr_indices, check_has_output_mask);

                rd = rd | ((instr_in_buf + 64) & has_output_val_mask);
            }

            /* Use rd unless node is the non-conditional node of a if/else/while, in which case use instr_no */
            const __m256i use_rd_mask = no_parent_mask | (~non_conditionals_mask & load_from_parents_mask);

            const __m256i node_size_mapping_indices = (node_types * data_type_count) + result_types;
            const __m256i instr_no = instr_in_buf + epi32::maskgatherz(node_size_mapping.data(), node_size_mapping_indices, non_conditionals_mask);

            const __m256i data_prop_value = (rd & use_rd_mask) | (instr_no & non_conditionals_mask);

            // TODO this value is loaded a few times, is that needed, or are we being nicer to the compiler
            const __m256i relative_offset = epi32::from_values(0, 1, 2, 3, 0, 1, 2, 3);

            /* Default to instr_in_buf */
            __m256i instr_loc = instr_in_buf & has_instr_mask;

            // TODO is this faster than a set_epi32? could be
            const __m256i relative_offset_1_mask = (relative_offset == 1);
            const __m256i if_else_mask = relative_offset_1_mask & (node_types == epi32::from_enum(if_else_statement));
            const __m256i while_mask = relative_offset_1_mask & (node_types == epi32::from_enum(while_statement));

            if (!epi32::is_zero(if_else_mask) || !epi32::is_zero(while_mask)) {
                __m256i register_indices = indices * parent_idx_per_node;
                register_indices = register_indices + (1_m256i & if_else_mask) + (2_m256i & while_mask);

                const __m256i load_from_reg_mask = if_else_mask | while_mask;

                /* Replace by registers if any are present */
                instr_loc = (instr_loc & ~load_from_reg_mask) | epi32::maskgatherz(registers.data(), register_indices, load_from_reg_mask);
            }

            /* func_decl or >=2 func_decl_dummy*/
            const __m256i instr_in_buf_add_two_mask = ((node_types == epi32::from_enum(func_decl)) | ((relative_offset > 1) & (node_types == epi32::from_enum(func_decl_dummy))));
            const __m256i func_call_arg_list_mask = relative_offset_1_mask & (node_types == epi32::from_enum(func_call_arg_list));

            /* Add 2 now and add child_idx later */
            instr_loc = instr_loc + (2_m256i & (instr_in_buf_add_two_mask | func_call_arg_list_mask));

            {
                /* We get -1, but this results in type 0 anyway, so it's ignored */
                const __m256i prev_indices = indices - 1;
                const __m256i prev_node_types = epi32::maskgatherz(this->node_types.data(), prev_indices, func_call_arg_list_mask);

                /* If the previous node is any kind of func call arg... */
                const __m256i func_call_arg_mask = func_call_arg_list_mask & ((prev_node_types & epi32::from_enum(func_call_arg) == epi32::from_enum(func_call_arg)));
                if (!epi32::is_zero(func_call_arg_mask)) {
                    /* add child_idx to res */
                    const __m256i prev_child_idx = epi32::maskgatherz(this->child_idx.data(), prev_indices, func_call_arg_mask);
                    instr_loc = instr_loc + (prev_child_idx & func_call_arg_mask);
                }
            }

            /* Gather actual instruction words */
            __m256i instr_words = epi32::maskgatherz(instr_table.data(), has_instr_indices, has_instr_mask);

            /* Re-used for jt */
            const __m256i instr_constant_indices = (node_types * 4) + relative_offset;

            // TODO node data is loaded once previously, not needed?
            /* Load all node data of nodes with calc_type != 0 */
            const __m256i node_data = epi32::maskgatherz(this->node_data.data(), indices, has_instr_mask);
            {
                /* Calculate instruction constants */
                const __m256i calc_type = epi32::maskgatherz(instr_constant_table.data(), instr_constant_indices, has_instr_mask);

                /* lui -> upper 20 bits of the constant, don't mode */
                instr_words = instr_words | ((node_data & epi32::from_value(0xFFFFF000)) & (calc_type == 1));
                /* addi -> lower 12 bits, shifted left by 20 */
                instr_words = instr_words | (_mm256_slli_epi32(node_data & epi32::from_value(0xFFF), 20) & (calc_type == 2));

                /* Multiple ones use node_data*4, pre-calculate */
                // TODO leftshift?
                const __m256i node_data_mul4 = node_data * 4;
                /* func call arg list, (-4 * (node_data + 2)) << 20 -> (-1 * (node_data_mul4 + 8)) << 20 */
                instr_words = instr_words | (_mm256_slli_epi32((node_data_mul4 + 8) * -1, 20) & (calc_type == 3));
                /* Func arg on stack */
                instr_words = instr_words | (_mm256_slli_epi32(node_data_mul4, 20) & (calc_type == 4));
                /* func arg list */
                instr_words = instr_words | (_mm256_slli_epi32(node_data_mul4 * -1, 20) & (calc_type == 5));
            }

            /* Calculate jump targets*/
            __m256i jump_targets = epi32::zero();
            {
                const __m256i calc_type = epi32::maskgatherz(instr_jt_table.data(), instr_constant_indices, has_instr_mask);

                /* calc_types 1-5 load from registers array*/
                const __m256i use_registers_mask = (~(calc_type > 5) & (calc_type > 0));
                
                /* idx * 3 + 1 for 1 and 2, +2 for 3 and 4 */
                const __m256i register_indices = (indices * parent_idx_per_node) + 1 + (1_m256i & (calc_type > 2));
                if (!epi32::is_zero(register_indices)) {
                    __m256i register_vals = epi32::maskgatherz(registers.data(), register_indices, use_registers_mask);

                    /* If calc_type is even (lowest bit is 0), add 1 */
                    register_vals = register_vals + (1_m256i & ((calc_type & 1) == epi32::zero()));

                    jump_targets = register_vals & use_registers_mask;
                }

                /* for 6 and 7, use func_ends and func_starts respectively */
                const __m256i use_func_ends_mask = (calc_type == 6);
                if (!epi32::is_zero(use_func_ends_mask)) {
                    const __m256i func_ends_vals = epi32::maskgatherz(this->func_ends.data(), node_data, use_func_ends_mask);

                    jump_targets = jump_targets | ((func_ends_vals - 6) & use_func_ends_mask);
                }

                const __m256i use_func_starts_mask = (calc_type == 7);
                if (!epi32::is_zero(use_func_starts_mask)) {
                    const __m256i func_starts_vals = epi32::maskgatherz(this->func_starts.data(), node_data, use_func_starts_mask);

                    jump_targets = jump_targets | (func_starts_vals & use_func_starts_mask);
                }
            }

            /* Calculate rs1 and rs2 */
            __m256i rs1 = epi32::zero();
            __m256i rs2 = epi32::zero();
            {
                // TODO convert to uint16_t array and merge gather calls?
                for (uint32_t i = 0; i < 2; ++i) {
                    const __m256i operand_table_indices = (has_instr_indices * 2) + i;
                    const __m256i calc_type = epi32::maskgatherz(operand_table.data(), operand_table_indices, has_instr_mask);

                    __m256i instr_arg = epi32::zero();

                    // TODO reorder to be more efficient?
                    const __m256i calc_type_1_mask = (calc_type == 1);
                    const __m256i calc_type_4_mask = (calc_type == 4);
                    const __m256i calc_type_5_mask = (calc_type == 5);
                    const __m256i calc_type_6_mask = (calc_type == 6);
                    const __m256i use_registers_mask = calc_type_1_mask | calc_type_4_mask | calc_type_5_mask | calc_type_6_mask;
                    if (!epi32::is_zero(use_registers_mask)) {
                        __m256i register_indices = indices * parent_idx_per_node;
                        register_indices = register_indices + (epi32::from_value(i) & calc_type_1_mask);
                        register_indices = register_indices + (1_m256i & (calc_type_4_mask | calc_type_5_mask));
                        register_indices = register_indices - (epi32::from_value(i) & calc_type_5_mask);

                        instr_arg = epi32::maskgatherz(registers.data(), register_indices, use_registers_mask);
                    }

                    instr_arg = instr_arg | ((node_data + 10) & (calc_type == 2)) | ((node_data + 42) & (calc_type == 3)) | ((instr_in_buf + 63) & (calc_type == 7));

                    if (i == 0) {
                        rs1 = instr_arg;
                    } else {
                        rs2 = instr_arg;
                    }
                }
            }

            AVX_ALIGNED auto has_instr_mask_array = epi32::extract(has_instr_mask);
            AVX_ALIGNED auto instr_loc_array = epi32::extract(instr_loc);
            AVX_ALIGNED auto instr_words_array = epi32::extract(instr_words);
            AVX_ALIGNED auto rd_array = epi32::extract(rd);
            AVX_ALIGNED auto rs1_array = epi32::extract(rs1);
            AVX_ALIGNED auto rs2_array = epi32::extract(rs2);
            AVX_ALIGNED auto jt_array = epi32::extract(jump_targets);

            for (uint32_t k = 0; k < 8; ++k) {
                if (has_instr_mask_array[k]) {
                    uint32_t idx = instr_loc_array[k];
                    instr[idx] = instr_words_array[k];
                    rd_avx[idx] = rd_array[k];
                    rs1_avx[idx] = rs1_array[k];
                    rs2_avx[idx] = rs2_array[k];
                    jt_avx[idx] = jt_array[k];
                }
            }

            /* Scalar-ly scatter because AVX2... */
            AVX_ALIGNED auto parent_indices = epi32::extract(parent_arg_idx);
            AVX_ALIGNED auto new_regs = epi32::extract(data_prop_value);

            // TODO does scattering here cause issues? possibly need to wait till end of loop or level
            for (uint32_t k = 0; k < 8; ++k) {
                // TODO >=0 or extract parent_arg_idx_call_mask too?
                if (int idx = parent_indices[k]; idx >= 0) {
                    registers[idx] = new_regs[k];
                }
            }
        }
    }
}
