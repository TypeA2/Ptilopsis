#include "codegen/rv_generator_avx.hpp"

#include <iostream>
#include <bitset>
#include <iomanip>
#include <cmath>
#include <bit>

#include <magic_enum.hpp>

#include "simd.hpp"
#include "disassembler.hpp"
#include "utils.hpp"

using namespace simd::epi32_operators;

namespace epi32 = simd::epi32;
namespace epi64 = simd::epi64;

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
        m256i types = _mm256_load_si256(node_types.m256i(i));

        const m256i func_arg_mask_source = epi32::from_enum(func_arg);
        const m256i types_masked = types & func_arg_mask_source;
        /* Mask for all nodes that are func args */
        m256i func_args_mask = (types_masked == func_arg_mask_source);

        /* Only continue if func args are found */
        if (!epi32::is_zero(func_args_mask)) {
            /* Retrieve all float nodes */
            m256i return_types = epi32::load(result_types.m256i(i));

            /* All nodes that are either FLOAT or FLOAT_REF */
            m256i float_func_args_mask = (return_types == epi32::from_enum(rv_data_type::FLOAT)) | (return_types == epi32::from_enum(rv_data_type::FLOAT_REF));

            /* Only take the func args node that are float */
            float_func_args_mask = float_func_args_mask & func_args_mask;

            /* All func args that are not float*/
            m256i int_func_args_mask = func_args_mask & ~float_func_args_mask;

            /* Node data = argument index of this type */
            m256i node_data = epi32::load(this->node_data.m256i(i));

            /* Child idx = argument index */
            m256i child_idx = epi32::load(this->child_idx.m256i(i));

            {
                /* All node data for float func args = arg index among floats */
                m256i float_data = node_data & float_func_args_mask;

                /* Mask of all float data larger than 7, so passed outside of a float register */
                m256i comp_mask = float_data > 7;

                /* Node data for all float args not passed in a float register, effectively the number of float args
                 *   Note that non-float-arg elements are calculated too, but these are just ignored at a later moment
                 */
                m256i int_args = child_idx - float_data;

                /* Calculate reg_offset = num_int_args + num_float_args - 8 */
                m256i reg_offset = int_args + float_data - 8;

                /* on_stack_mask = all reg_offsets >= 8 -> passed on stack */
                m256i on_stack_mask = (reg_offset > 7) & comp_mask;
                /* as_int_mask = all reg_offsets < 8 and originally >8 -> passed in integer register */
                m256i as_int_mask = ~on_stack_mask & comp_mask; 

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
                m256i int_data = node_data & int_func_args_mask;

                /* Number of args of the other type -> float args */
                m256i float_args = child_idx - int_data;

                /* reg_offset = int_data + (float_args > 7 ? (float_args - 8) : float_args) */
                m256i reg_offset = int_data + (float_args - (8_m256i & (float_args > 7)));

                /* Offsets of 8 and above are put on the stack*/
                m256i on_stack_mask = (reg_offset > 7);

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
        m256i types = epi32::load(node_types.m256i(i));

        /* NOTE: Pareas has a i > 0 check, but in any valid program for i == 0, this mask will be 0 */
        m256i func_call_arg_list_mask = (types == epi32::from_enum(func_call_arg_list));

        if (!epi32::is_zero(func_call_arg_list_mask)) {
            /* Unaligned load, use pointer arithmetic to prevent overflows */
            m256i prev_types = epi32::loadu(&node_types[i * 8] - 1);

            /* Load the previous node data, unaligned */
            m256i prev_node_data = epi32::loadu(&node_data[i * 8] - 1);

            /* Mask of all stack func call args */
            m256i func_call_stack_arg_mask = (prev_types == epi32::from_enum(func_call_arg_on_stack));
            if (!epi32::is_zero(func_call_stack_arg_mask)) {
                m256i adjusted_node_data = prev_node_data + 1;

                /* Store the relevant updated fields */
                epi32::maskstore(node_data.m256i(i), func_call_stack_arg_mask, adjusted_node_data);
            }

            /* Mask all normal func call args that are floats */
            m256i float_func_call_arg_mask = (prev_types == epi32::from_enum(func_call_arg));
            if (!epi32::is_zero(float_func_call_arg_mask)) {
                m256i prev_result_types = epi32::loadu(&result_types[i * 8] - 1);

                float_func_call_arg_mask = float_func_call_arg_mask & (prev_result_types == epi32::from_enum(rv_data_type::FLOAT));
                if (!epi32::is_zero(float_func_call_arg_mask)) {
                    m256i prev_child_idx = epi32::loadu(&child_idx[i * 8] - 1);

                    m256i int_args = prev_child_idx - prev_node_data;
                    int_args = int_args - 8;
                    int_args = epi32::max(int_args, 0_m256i);

                    /* Store the updated fields */
                    epi32::maskstore(node_data.m256i(i), float_func_call_arg_mask, int_args);
                }
            }
        }
    }

    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        //const m256i iota = epi32::from_value(static_cast<int>(i << 3)) + epi32::from_values(0, 1, 2, 3, 4, 5, 6, 7);

        m256i parents = epi32::load(this->parents.m256i(i));

        /* All nodes with parents. There is only ever 1 node per program without a parent, so this mask is always nonzero */
        m256i valid_parents_mask = (parents > -1);
        m256i parent_types = epi32::gather(node_types.data(), parents & valid_parents_mask);

        /* All nodes of which the parent is a comparison node */
        m256i parent_eq_expr_mask = ((parent_types & epi32::from_enum(eq_expr)) == epi32::from_enum(eq_expr));

        /* All targeted nodes */
        m256i result_types = epi32::load(this->result_types.m256i(i));
        result_types = result_types & parent_eq_expr_mask;

        /* The ones that are of data type float */
        m256i result_types_mask = (result_types == epi32::from_enum(rv_data_type::FLOAT));

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
        m256i node_types = epi32::load(this->node_types.m256i(i));
        /* node_types is the in the outer array, so multiply by _mm256_set1_epi32 to get the actual index */
        // TODO maybe pad data_type_array_size to 8 elements and use a shift instead, also maybe pre-scale to 4
        m256i node_types_indices = node_types * data_type_count;

        m256i result_types = epi32::load(this->result_types.m256i(i));
        /* result_types is the index in the inner array, so just add to the previously calculated offsets */
        node_types_indices = node_types_indices + result_types;

        /* Use the calculated indices to gather the base node counts */
        m256i base = epi32::gather(node_size_mapping.data(), node_types_indices);

        /* Offset is calculated and applied to base value */
        m256i delta = epi32::zero();

        m256i func_call_arg_list_mask = (node_types == epi32::from_enum(func_call_arg_list));
        if (!epi32::is_zero(func_call_arg_list_mask)) {
            /* Already add 1 for every arg list */
            delta = delta + (1_m256i & func_call_arg_list_mask);

            m256i prev_node_types = epi32::loadu(&this->node_types[i * 8] - 1);
            
            // TODO is this branch even necessary?
            m256i func_call_arg_mask = func_call_arg_list_mask & ((prev_node_types & epi32::from_enum(func_call_arg)) == epi32::from_enum(func_call_arg));
            /* In a valid program we're basically guaranteed to get at least 1 here, so skip the possible branch */
            m256i prev_child_idx = epi32::loadu(&this->child_idx[i * 8] - 1);

            /* Add the child idx + 1 of the previous node (so the last arg) to the func call arg list */
            delta = delta + ((prev_child_idx + 1) & func_call_arg_mask);
        }

        m256i parents = epi32::load(this->parents.m256i(i));
        m256i valid_parents_mask = ~(parents == -1);
        m256i parent_types = epi32::gather(this->node_types.data(), parents & valid_parents_mask);
        m256i child_idx = epi32::load(this->child_idx.m256i(i));

        /* Add 1 for the conditional nodes of if/if_else/while, and add 1 for the if-branch of an if_else */
        m256i if_else_mask = (parent_types == epi32::from_enum(if_else_statement));
        m256i if_statement_conditional_mask = ((child_idx == 0) & (if_else_mask | (parent_types == epi32::from_enum(if_statement))));

        /* All nodes with child_idx == 0 and that are if or if_else */
        delta = delta + (1_m256i & if_statement_conditional_mask);

        m256i if_else_while_2nd_node_mask = ((child_idx == 1) & (if_else_mask | (parent_types == epi32::from_enum(while_statement))));
        delta = delta + (1_m256i & if_else_while_2nd_node_mask);

        base = base + delta;

        epi32::store(this->node_sizes.m256i(i), base);
    }

    /* Prefix sum from ADMS20_05 */
    
    /* Accumulator */
    m256i offset = epi32::zero();
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        m256i node_size = epi32::loadu(&this->node_sizes[i * 8] - 1);

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
        m256i parents = epi32::load(this->parents.m256i(i));
        m256i valid_parents_mask = (parents > -1);

        m256i child_idx = epi32::load(this->child_idx.m256i(i));
        m256i parent_types = epi32::gather(this->node_types.data(), parents & valid_parents_mask);

        /* All conditional nodes of if and if_else */
        m256i conditionals_mask = ((parent_types & epi32::from_enum(if_statement)) == epi32::from_enum(if_statement));
        /* Lowest bit represents the conditional node child index, so check if it's equal */
        conditionals_mask = conditionals_mask & (child_idx == (parent_types & 1));

        if (!epi32::is_zero(conditionals_mask)) {
            /* For all conditional nodes, set the offset of the parent to past the current node */
            m256i indices = epi32::blendv(epi32::from_value(-1), parents, conditionals_mask);
            m256i values = epi32::load(this->node_locations.m256i(i));

            /* Get base node size for all nodes */
            m256i node_types_indices = (epi32::load(this->node_types.m256i(i)) * data_type_count) + epi32::load(this->result_types.m256i(i));
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
        m256i func_call_arg_mask = epi32::load(this->node_types.m256i(i));
        
        func_call_arg_mask = (epi32::from_enum(func_call_arg) == (func_call_arg_mask & epi32::from_enum(func_call_arg)));

        if (!epi32::is_zero(func_call_arg_mask)) {
            /* New location is the location of the parent plus the child idx plus 1 */
            m256i parent_locations = epi32::maskgatherz(this->node_locations.data(), parents, func_call_arg_mask);
            m256i new_locs = parent_locations + epi32::load(this->child_idx.m256i(i)) + 1;

            epi32::maskstore(this->node_locations.m256i(i), func_call_arg_mask, new_locs);
        }
    }

    /* Function table generation */
    // TODO benchmark 1 vs 2 pass method (aka with vs without reallocations)
    size_t func_count = 0;
    for (size_t i = 0; i < node_types.size_m256i(); ++i) {
        m256i node_types = epi32::load(this->node_types.m256i(i));

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
        m256i node_types = epi32::load(this->node_types.m256i(i));

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
        m256i func_decl_indices = epi32::load(func_decls.m256i(i));

        /* Retrieve the node data of every func decl */
        epi32::store(this->function_ids.m256i(i), epi32::gather(this->node_data.data(), func_decl_indices));

        /* Calculate the offset of each function */
        m256i offset = 6_m256i + epi32::gather(this->node_locations.data(), func_decl_indices);
        epi32::storeu(&func_starts[i * 8] + 1, offset);
        epi32::store(func_offsets.m256i(i), offset);
    }

    function_sizes = avx_buffer<uint32_t> { func_count };
    for (size_t i = 0; i < func_decls.size_m256i(); ++i) {
        /* Load elements and shifted elements directly */
        m256i shifted_offsets = epi32::loadu(&func_offsets[i * 8] - 1);
        m256i offsets = epi32::load(func_offsets.m256i(i));

        epi32::store(function_sizes.m256i(i), offsets - shifted_offsets);
    }

    func_ends = avx_buffer<uint32_t> { func_count };
    for (size_t i = 0; i < func_decls.size_m256i(); ++i) {
        m256i start = epi32::load(func_starts.m256i(i));
        m256i size = epi32::load(function_sizes.m256i(i));

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
            const m256i indices = _mm256_set_m128i(_mm_set1_epi32(idx_1), _mm_set1_epi32(idx_0));
            const m256i iota = _mm256_broadcastsi128_si256(_mm_set_epi32(3 * data_type_count, 2 * data_type_count, data_type_count, 0));

            // TODO this as a cached load or maybe only perform 2 loads and shuffle?
            const m256i result_types = epi32::gather(this->result_types.data(), indices);

            // TODO is gather even faster than 2 individual loads here?
            const m256i node_types = epi32::gather(this->node_types.data(), indices);
            m256i has_instr_indices = node_types * (data_type_count * 4);
            has_instr_indices = has_instr_indices + iota + result_types;

            // TODO masking SHOULD be faster:
            // Hopefully the gather caches the first 4 elements, but still this would be >1 (likely ~4-5 for L1d) cycles of latency
            // _mm256_or_si256 has a latency of 1 and CPI of 0.33, so should be faster than a cached data access.
            // Question is whether this holds for larger datasets...
            const __m128i mask_true = _mm_set1_epi32(0xFFFFFFFF);
            const m256i has_inst_load_mask = _mm256_set_m128i(skip_last ? _mm_setzero_si128() : mask_true, mask_true);
            const m256i has_instr_mask = epi32::maskgatherz(has_instr_mapping.data(), has_instr_indices, has_inst_load_mask);

            /* If has_instr OR if i == 0, propagate data */
            const m256i propagate_data_mask = (has_instr_mask | epi32::from_values(-1, 0, 0, 0, skip_last ? 0 : -1, 0, 0, 0));
            /* Never 0, since the first value will always be propagated */
            
            const m256i parents = epi32::maskgatherz(this->parents.data(), indices, propagate_data_mask);
            const m256i no_parent_mask = (parents == -1);
            const m256i load_from_parents_mask = (~no_parent_mask & propagate_data_mask);
            const m256i parent_types = epi32::maskgatherz(this->node_types.data(), parents, load_from_parents_mask);

            /* child_idx value of all non-parentless nodes */
            const m256i child_idx = epi32::maskgatherz(this->child_idx.data(), indices, load_from_parents_mask);
                
            /* Non-conditional nodes of branching instructions */
            m256i non_conditionals_mask = ((parent_types & epi32::from_enum(if_statement)) == epi32::from_enum(if_statement));
            non_conditionals_mask = non_conditionals_mask & ~(child_idx == (parent_types & 1));

            /* Mask for which nodes to call parent_arg_idx on, at least on non-conditionals */
            m256i parent_arg_idx_call_mask = non_conditionals_mask;

            /* Load parent_arg_idx_lookup for parentless or conditional nodes */
            const m256i parent_arg_idx_lookup_load_mask = (no_parent_mask | ~non_conditionals_mask) & propagate_data_mask;
            if (!epi32::is_zero(parent_arg_idx_lookup_load_mask)) {
                /* Same shape as has_instr_mapping */
                const m256i calc_type = epi32::maskgatherz(parent_arg_idx_lookup.data(), has_instr_indices, parent_arg_idx_lookup_load_mask);

                /* calc_type == 1 means a sub-call */
                parent_arg_idx_call_mask = parent_arg_idx_call_mask | (calc_type == 1);

                /* If result_type > 1 for calc_type 2, also call */
                const m256i calc_type_2_mask = (calc_type == 2);
                if (!epi32::is_zero(calc_type_2_mask)) {
                    const m256i result_types = epi32::maskgatherz(this->result_types.data(), indices, calc_type_2_mask);
                    parent_arg_idx_call_mask = parent_arg_idx_call_mask | (result_types > 1);
                }
            }

            /* For all nodes in parent_arg_idx_call_mask... */
            m256i parent_arg_idx = ((parents * parent_idx_per_node) + child_idx) & parent_arg_idx_call_mask;
            parent_arg_idx = parent_arg_idx | (epi32::from_value(-1) & ~parent_arg_idx_call_mask);
            
            /* Last argument to get_data_prop_value */
            const m256i instr_in_buf = epi32::from_values(0, 1, 2, 3, 0, 1, 2, 3) + epi32::maskgatherz(this->node_locations.data(), indices, propagate_data_mask);
            m256i rd = epi32::zero();

            /* Calculate get_data_prop_value */
            if (!epi32::is_zero(has_instr_mask)) {
                /* Need rd for instructions with register
                 * 
                 * get_output_table also has the same shape as has_instr_mapping
                 * 
                 * Mask ungathered values to -1, so check_has_output_mask doesn't pick them up 
                 */
                const m256i calc_type = epi32::maskgather(epi32::from_value(-1), get_output_table.data(), has_instr_indices, has_instr_mask);

                rd = 32_m256i & (calc_type == 3);
                rd = rd | (10_m256i & (calc_type == 4));

                // TODO cmpeq has CPI of 0.5 but bitwise AND has CPI of 0.33, maybe use that instead?
                const m256i calc_type_1_mask = (calc_type == 1);
                const m256i calc_type_2_mask = (calc_type == 2);
                const m256i use_node_data_mask = calc_type_1_mask | calc_type_2_mask;
                if (!epi32::is_zero(use_node_data_mask)) {
                    const  m256i node_data = epi32::maskgatherz(this->node_data.data(), indices, use_node_data_mask);

                    rd = rd | ((node_data + 10) & calc_type_1_mask);
                    rd = rd | ((node_data + 42) & calc_type_2_mask);
                }

                /* Default, use parent_arg_idx and has_output */
                m256i check_has_output_mask = (calc_type == 0);
                /* Exceedingly unlikely to be zero, so continue */

                /* If there is an output index or explicitly can have an output  */
                m256i has_output_val_mask = (parent_arg_idx_call_mask & has_instr_mask);
                has_output_val_mask = has_output_val_mask |  epi32::maskgatherz(has_output.data(), has_instr_indices, check_has_output_mask);

                rd = rd | ((instr_in_buf + 64) & has_output_val_mask);
            }

            /* Use rd unless node is the non-conditional node of a if/else/while, in which case use instr_no */
            const m256i use_rd_mask = no_parent_mask | (~non_conditionals_mask & load_from_parents_mask);

            const m256i node_size_mapping_indices = (node_types * data_type_count) + result_types;
            const m256i instr_no = instr_in_buf + epi32::maskgatherz(node_size_mapping.data(), node_size_mapping_indices, non_conditionals_mask);

            const m256i data_prop_value = (rd & use_rd_mask) | (instr_no & non_conditionals_mask);

            // TODO this value is loaded a few times, is that needed, or are we being nicer to the compiler
            const m256i relative_offset = epi32::from_values(0, 1, 2, 3, 0, 1, 2, 3);

            /* Default to instr_in_buf */
            m256i instr_loc = instr_in_buf & has_instr_mask;

            // TODO is this faster than a set_epi32? could be
            const m256i relative_offset_1_mask = (relative_offset == 1);
            const m256i if_else_mask = relative_offset_1_mask & (node_types == epi32::from_enum(if_else_statement));
            const m256i while_mask = relative_offset_1_mask & (node_types == epi32::from_enum(while_statement));

            if (!epi32::is_zero(if_else_mask) || !epi32::is_zero(while_mask)) {
                m256i register_indices = indices * parent_idx_per_node;
                register_indices = register_indices + (1_m256i & if_else_mask) + (2_m256i & while_mask);

                const m256i load_from_reg_mask = if_else_mask | while_mask;

                /* Replace by registers if any are present */
                instr_loc = (instr_loc & ~load_from_reg_mask) | epi32::maskgatherz(registers.data(), register_indices, load_from_reg_mask);
            }

            /* func_decl or >=2 func_decl_dummy*/
            const m256i instr_in_buf_add_two_mask = ((node_types == epi32::from_enum(func_decl)) | ((relative_offset > 1) & (node_types == epi32::from_enum(func_decl_dummy))));
            const m256i func_call_arg_list_mask = relative_offset_1_mask & (node_types == epi32::from_enum(func_call_arg_list));

            /* Add 2 now and add child_idx later */
            instr_loc = instr_loc + (2_m256i & (instr_in_buf_add_two_mask | func_call_arg_list_mask));

            {
                /* We get -1, but this results in type 0 anyway, so it's ignored */
                const m256i prev_indices = indices - 1;
                const m256i prev_node_types = epi32::maskgatherz(this->node_types.data(), prev_indices, func_call_arg_list_mask);

                /* If the previous node is any kind of func call arg... */
                const m256i func_call_arg_mask = func_call_arg_list_mask & ((prev_node_types & epi32::from_enum(func_call_arg)) == epi32::from_enum(func_call_arg));
                if (!epi32::is_zero(func_call_arg_mask)) {
                    /* add child_idx to res */
                    const m256i prev_child_idx = epi32::maskgatherz(this->child_idx.data(), prev_indices, func_call_arg_mask);
                    instr_loc = instr_loc + (prev_child_idx & func_call_arg_mask);
                }
            }

            /* Gather actual instruction words */
            m256i instr_words = epi32::maskgatherz(instr_table.data(), has_instr_indices, has_instr_mask);

            /* Re-used for jt */
            const m256i instr_constant_indices = (node_types * 4) + relative_offset;

            // TODO node data is loaded once previously, not needed?
            /* Load all node data of nodes with calc_type != 0 */
            const m256i node_data = epi32::maskgatherz(this->node_data.data(), indices, has_instr_mask);
            {
                /* Calculate instruction constants */
                const m256i calc_type = epi32::maskgatherz(instr_constant_table.data(), instr_constant_indices, has_instr_mask);

                /* lui -> upper 20 bits of the constant, don't mode */
                instr_words = instr_words | ((node_data & epi32::from_value(0xFFFFF000)) & (calc_type == 1));
                /* addi -> lower 12 bits, shifted left by 20 */
                instr_words = instr_words | (_mm256_slli_epi32(node_data & epi32::from_value(0xFFF), 20) & (calc_type == 2));

                /* Multiple ones use node_data*4, pre-calculate */
                // TODO leftshift?
                const m256i node_data_mul4 = node_data * 4;
                /* func call arg list, (-4 * (node_data + 2)) << 20 -> (-1 * (node_data_mul4 + 8)) << 20 */
                instr_words = instr_words | (_mm256_slli_epi32((node_data_mul4 + 8) * -1, 20) & (calc_type == 3));
                /* Func arg on stack */
                instr_words = instr_words | (_mm256_slli_epi32(node_data_mul4, 20) & (calc_type == 4));
                /* func arg list */
                instr_words = instr_words | (_mm256_slli_epi32(node_data_mul4 * -1, 20) & (calc_type == 5));
            }

            /* Calculate jump targets*/
            m256i jump_targets = epi32::zero();
            {
                const m256i calc_type = epi32::maskgatherz(instr_jt_table.data(), instr_constant_indices, has_instr_mask);

                /* calc_types 1-5 load from registers array*/
                const m256i use_registers_mask = (~(calc_type > 5) & (calc_type > 0));
                
                /* idx * 3 + 1 for 1 and 2, +2 for 3 and 4 */
                const m256i register_indices = (indices * parent_idx_per_node) + 1 + (1_m256i & (calc_type > 2));
                if (!epi32::is_zero(register_indices)) {
                    m256i register_vals = epi32::maskgatherz(registers.data(), register_indices, use_registers_mask);

                    /* If calc_type is even (lowest bit is 0), add 1 */
                    register_vals = register_vals + (1_m256i & ((calc_type & 1) == epi32::zero()));

                    jump_targets = register_vals & use_registers_mask;
                }

                /* for 6 and 7, use func_ends and func_starts respectively */
                const m256i use_func_ends_mask = (calc_type == 6);
                if (!epi32::is_zero(use_func_ends_mask)) {
                    const m256i func_ends_vals = epi32::maskgatherz(this->func_ends.data(), node_data, use_func_ends_mask);

                    jump_targets = jump_targets | ((func_ends_vals - 6) & use_func_ends_mask);
                }

                const m256i use_func_starts_mask = (calc_type == 7);
                if (!epi32::is_zero(use_func_starts_mask)) {
                    const m256i func_starts_vals = epi32::maskgatherz(this->func_starts.data(), node_data, use_func_starts_mask);

                    jump_targets = jump_targets | (func_starts_vals & use_func_starts_mask);
                }
            }

            /* Calculate rs1 and rs2 */
            m256i rs1 = epi32::zero();
            m256i rs2 = epi32::zero();
            {
                // TODO convert to uint16_t array and merge gather calls?
                for (uint32_t i = 0; i < 2; ++i) {
                    const m256i operand_table_indices = (has_instr_indices * 2) + i;
                    const m256i calc_type = epi32::maskgatherz(operand_table.data(), operand_table_indices, has_instr_mask);

                    m256i instr_arg = epi32::zero();

                    // TODO reorder to be more efficient?
                    const m256i calc_type_1_mask = (calc_type == 1);
                    const m256i calc_type_4_mask = (calc_type == 4);
                    const m256i calc_type_5_mask = (calc_type == 5);
                    const m256i calc_type_6_mask = (calc_type == 6);
                    const m256i use_registers_mask = calc_type_1_mask | calc_type_4_mask | calc_type_5_mask | calc_type_6_mask;
                    if (!epi32::is_zero(use_registers_mask)) {
                        m256i register_indices = indices * parent_idx_per_node;
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

void rv_generator_avx::optimize() {
    avx_buffer<uint32_t> used_registers { instr.size() };
    for (size_t i = 0; i < instr.size_m256i(); ++i) {
        // TODO cmpeq 0 vs cmpgt -1 shouldn't matter too much, right?
        AVX_ALIGNED auto rs1_array = epi32::extract(epi32::load(this->rs1_avx.m256i(i)) - 64);
        AVX_ALIGNED auto rs2_array = epi32::extract(epi32::load(this->rs2_avx.m256i(i)) - 64);


        for (uint32_t j = 0; j < 8; ++j) {
            if (int idx = rs1_array[j]; idx >= 0) {
                used_registers[idx] = 0xFFFFFFFF;
            }
        }

        for (uint32_t j = 0; j < 8; ++j) {
            if (int idx = rs2_array[j]; idx >= 0) {
                used_registers[idx] = 0xFFFFFFFF;
            }
        }
    }

    bool cont = true;
    while (cont) {
        avx_buffer<uint32_t> new_used_registers = used_registers;

        /* Need to be processed in 2 steps to preserve consistency */
        for (uint32_t i = 0; i < instr.size_m256i(); ++i) {
            const m256i rd = epi32::load(this->rd_avx.m256i(i));

            /* If rd >= 64, rd - 64 is the instruction index */
            m256i can_remove_mask = (rd > 63);

            const m256i used_mask = epi32::maskgatherz(used_registers.data(), rd - 64, can_remove_mask);
            can_remove_mask = can_remove_mask & ~used_mask;

            const m256i rs1_src = epi32::load(this->rs1_avx.m256i(i)) - 64;
            const m256i rs2_src = epi32::load(this->rs2_avx.m256i(i)) - 64;

            /* Non-masked entries get set to -1, so won't be used in the scatter operation */
            m256i unset = (epi32::from_value(-1) & ~can_remove_mask);
            AVX_ALIGNED auto indices = epi32::extract((rs1_src & can_remove_mask) | unset);

            /* Scatter */
            for (uint32_t j = 0; j < 8; ++j) {
                if (int idx = indices[j]; idx >= 0) {
                    new_used_registers[idx] = 0;
                }
            }

            indices = epi32::extract((rs2_src & can_remove_mask) | unset);

            for (uint32_t j = 0; j < 8; ++j) {
                if (int idx = indices[j]; idx >= 0) {
                    new_used_registers[idx] = 0;
                }
            }
        }
            
        for (uint32_t i = 0; i < instr.size_m256i(); ++i) {
            const m256i rs1_src = epi32::load(this->rs1_avx.m256i(i)) - 64;
            const m256i rs2_src = epi32::load(this->rs2_avx.m256i(i)) - 64;

            const m256i instr = epi32::load(this->instr.m256i(i));
            const m256i instr_opcode_func3 = (instr & 0b0000000'00000'00000'111'00000'1111111);
            const m256i is_store_mask = (instr_opcode_func3 == 0b0000000'00000'00000'010'00000'0100011) | (instr_opcode_func3 == 0b0000000'00000'00000'010'00000'0100111);

            const m256i unset = (epi32::from_value(-1) & ~is_store_mask);
            AVX_ALIGNED auto indices = epi32::extract((rs1_src & is_store_mask) | unset);

            for (uint32_t j = 0; j < 8; ++j) {
                if (int idx = indices[j]; idx >= 0) {
                    new_used_registers[idx] = 0xFFFFFFFF;
                }
            }

            indices = epi32::extract((rs2_src & is_store_mask) | unset);

            for (uint32_t j = 0; j < 8; ++j) {
                if (int idx = indices[j]; idx >= 0) {
                    new_used_registers[idx] = 0xFFFFFFFF;
                }
            }
        }

        /* If nothing changed, we're done */
        if (used_registers == new_used_registers) {
            cont = false;
        }

        used_registers = std::move(new_used_registers);
    }

    used_instrs_avx = avx_buffer<uint32_t> { instr.size() };
    for (uint32_t i = 0; i < instr.size_m256i(); ++i) {
        const m256i rd = epi32::load(this->rd_avx.m256i(i));
        const m256i used_mask = epi32::load(used_registers.m256i(i));

        epi32::store(used_instrs_avx.m256i(i), ~(rd > 63) | used_mask);
    }
}

void rv_generator_avx::regalloc() {
    /* Process 1 instruction per function at a time, to at least have some parallelism */
    uint32_t max_func_size = std::ranges::max(function_sizes);
    uint32_t func_count = static_cast<uint32_t>(function_sizes.size());

    // TODO 2x 32-bit or 1x 64-bit?
    /* Maps physical (* func_count) to virtual register */
    auto register_state = avx_buffer<uint32_t>::fill(func_count * 64, -1);

    /* Current lifetime state (= used registers) of every function */
    auto lifetime_masks = avx_buffer<uint64_t>::fill(func_count, 0b00000000'00000000'00000000'00000000'00000000'00000000'00000000'00011111);

    /* Keeps track what registers need to be preserved if a function call occurs */
    avx_buffer<uint64_t> preserve_masks { func_count };

    /* For every possible virtual register within our program, keep track of the physical register it's in and whether it's swapped out currently */
    avx_buffer<uint32_t> symbol_registers { used_instrs_avx.size() };
    avx_buffer<uint32_t> symbol_swapped { used_instrs_avx.size() };

    /* For every instruction... */
    for (uint32_t i = 0; i < max_func_size; ++i) {
        avx_buffer<uint32_t> original_register_state = register_state;

        /* For every function... */
        for (uint32_t j = 0; j < function_sizes.size_m256i(); ++j) {
            const m256i func_indices = (8_m256i * j) + epi32::from_values(0, 1, 2, 3, 4, 5, 6, 7);
            const m256i func_starts = epi32::load(this->func_starts.m256i(j));
            const m256i func_sizes = epi32::load(this->function_sizes.m256i(j));

            /* All 1's, aka -1, if i > size */
            const m256i instr_offset = (func_starts + i) | (epi32::from_value(i) >= func_sizes);

            /* From lifetime_result for function x:
             *   mask is the lifetime mask after this instruction, gets assigned to lifetime_masks[x]
             *   reg_info gets combined into symb_data
             *     reg is the virtual register it belongs to, this is used to index into symbol_registers/symbol_swapped
             *     sym is the state of this register, corresponding to the previously mentioned arrays, this is written into symbol_registers/symbol_swapped
             *   For every positive element in swapped, the physical register at that index was sawpped out.
             *     Mark the symbols contained in these registers as swapped (and update in symbol_registers/symbol_swapped)
             *   registers is used to directly update this functions register_state
             */
            const m256i in_bounds_mask = ~(instr_offset == -1);
            /* For all valid indices, gather whether the instructions are enabled */
            const m256i valid_mask = in_bounds_mask | epi32::maskgatherz(used_instrs_avx.data(), instr_offset, in_bounds_mask);

            /* There's always 1 instruction enabled, else we wouldn't be here at all */
            const m256i instr = epi32::maskgatherz(this->instr.data(), instr_offset, valid_mask);

            // TODO opcode 1101111 (JAL) is never generated, shoudn't this be JALR?
            m256i is_call_mask = (instr == 0b0000000'00000'00000'000'00001'1101111);
            const m256i jt = epi32::maskgatherz(this->jt_avx.data(), instr_offset, is_call_mask);
            const m256i func_ends = func_starts + func_sizes;

            m256i lifetime_mask_lo = epi64::load(lifetime_masks.m256i(2 * j));
            m256i lifetime_mask_hi = epi64::load(lifetime_masks.m256i((2 * j) + 1));

            /* A call that jumps to outside the current function */
            is_call_mask = is_call_mask & ((jt < func_starts) | (jt >= func_ends));
            if (!epi32::is_zero(is_call_mask)) {
                std::cout << "wow\n";
                /* For calls, no rd, rs1, rs2 */
                //const m256i new_lifetime_mask_lo = ();
                const m256i new_lifetime_mask_lo = epi64::and256(lifetime_mask_lo, preserved_register_mask);
                const m256i new_lifetime_mask_hi = epi64::and256(lifetime_mask_hi, preserved_register_mask);

                const m256i spill_mask_lo = epi64::and256(lifetime_mask_lo, ~preserved_register_mask);
                const m256i spill_mask_hi = epi64::and256(lifetime_mask_hi, ~preserved_register_mask);

                /* For every register reprsented in the spill masks, if they are spilled, mark them as swapped 
                 *   Use the register number to index into original_reg_state to obtain the virtual register.
                 *     Set the symbol to swapped in symbol_swapped.
                 */
                for (uint32_t k = 0; k < 64; ++k) {
                    const m256i extract_reg_mask = _mm256_slli_epi64(epi64::from_value(1), k);
                    
                    /* Whether register k needs to be saved */
                    const m256i spilled_mask = epi32::pack64_32((spill_mask_lo & extract_reg_mask) > 0, (spill_mask_hi & extract_reg_mask) > 0);
 
                    /* new_lifetime_mask marks all saved registers at a call as free afterwards */
                    const m256i new_lifetime_mask = epi32::pack64_32(
                        (new_lifetime_mask_lo & extract_reg_mask) > 0, (new_lifetime_mask_hi & extract_reg_mask) > 0);

                    /* Indices into flattened array */
                    const m256i register_state_indices = (func_indices * 64) + k;

                    /* If the current physical register is spilled,
                     * obtain the current virtual register residing in it from original_register_state
                     */
                    // TODO this could be if'd out, but then again function calls are somewhat rare
                    const m256i virtual_regs = epi32::maskgatherz(
                        original_register_state.data(),
                        register_state_indices,
                        spilled_mask | new_lifetime_mask
                    );
                    const m256i swapped_regs = virtual_regs - 64;

                    /* For all swapped virtual registers, set swapped to true */
                    const m256i swapped_indices = (swapped_regs & spilled_mask) | (epi32::from_value(-1) & ~spilled_mask);

                    /* For all spilled reigsters, set swapped to 1 */
                    AVX_ALIGNED auto swapped_indices_array = epi32::extract(swapped_indices);
                    for (uint32_t l = 0; l < 8; ++l) {
                        /* Mark swapped symbols as swapped out */
                        if (int idx = swapped_indices_array[l]; idx >= 0) {
                            symbol_swapped[idx] = 0xFFFFFFFF;
                        }
                    }

                    /* If the register is not free after this instruction, transfer the mapping, else mark as unmapped */
                    const m256i transferred_regs = (virtual_regs & new_lifetime_mask) | (epi32::from_value(-1) & ~new_lifetime_mask);

                    AVX_ALIGNED auto transferred_regs_array = epi32::extract(transferred_regs);
                    AVX_ALIGNED auto register_state_indices_array = epi32::extract(register_state_indices);
                    for (uint32_t l = 0; l < 8; ++l) {
                        register_state[register_state_indices_array[l]] = transferred_regs_array[l];
                    }
                }
            }

            const m256i non_call_mask = ~is_call_mask & valid_mask;
            if (!epi32::is_zero(non_call_mask)) {
                /* rs1 and rs2 are used to query current register state */
                const m256i rs1 = epi32::maskgatherz(this->rs1_avx.data(), instr_offset, non_call_mask);
                const m256i rs1_virtual_mask = (rs1 > 63);

                const m256i rs2 = epi32::maskgatherz(this->rs2_avx.data(), instr_offset, non_call_mask);
                const m256i rs2_virtual_mask = (rs2 > 63);

                /* Use virtual registers to query current physical register and swap state */
                const m256i rs1_adjusted = rs1 - 64;
                m256i rs1_registers = epi32::maskgatherz(symbol_registers.data(), rs1_adjusted, rs1_virtual_mask);
                m256i rs1_swapped = epi32::maskgatherz(symbol_swapped.data(), rs1_adjusted, rs1_virtual_mask);

                const m256i rs2_adjusted = rs1 - 64;
                m256i rs2_registers = epi32::maskgatherz(symbol_registers.data(), rs2_adjusted, rs2_virtual_mask);
                m256i rs2_swapped = epi32::maskgatherz(symbol_swapped.data(), rs2_adjusted, rs2_virtual_mask);

                /* For bot rs1 and rs2, if they're currently swapped, allocate a new register to load them in */
                if (!epi32::is_zero(rs1_swapped)) {
                    /* FCVT.W.S needs a float source register, and only has rs1, so if the instr matches, require a float */
                    const m256i rs1_fcvt_w_s_mask = (instr == 0b1100000'00000'00000'111'00000'1010011);
                    
                    /* Or if the instruction opcode is that of floats */
                    const m256i rs1_needs_float_mask = rs1_fcvt_w_s_mask | ((instr & 0b1111111) == 0b1010011);

                    /* All swapped registers, 5 if int, 37 if float */
                    rs1_registers = (rs1_registers & ~rs1_swapped)
                        | (5_m256i & (rs1_swapped & ~rs1_needs_float_mask))
                        | (37_m256i & (rs1_swapped & rs1_needs_float_mask));
                }

                if (!epi32::is_zero(rs2_swapped)) {
                    /* rs2 is only ever float in a "normal" float instruction */
                    const m256i rs2_needs_float_mask = ((instr & 0b1111111) == 0b1010011);

                    rs2_registers = (rs2_registers & ~rs2_swapped)
                        | (6_m256i & (rs2_swapped & ~rs2_needs_float_mask))
                        | (38_m256i & (rs2_swapped & rs2_needs_float_mask));
                }

                /* The architecture is read-once, so mark our newly calculated rs1 and rs2 as ready for re-use */
                const m256i one = epi64::from_value(1);
                m256i clear_register_mask_lo = _mm256_sllv_epi64(one, epi32::expand32_64_lo(rs1_registers));
                clear_register_mask_lo = clear_register_mask_lo & _mm256_sllv_epi64(one, epi32::expand32_64_lo(rs2_registers));

                /* Never free up register 0*/
                clear_register_mask_lo = clear_register_mask_lo & epi64::from_value(-2);

                m256i clear_register_mask_hi = _mm256_sllv_epi64(one, epi32::expand32_64_hi(rs1_registers));
                clear_register_mask_hi = clear_register_mask_hi & _mm256_sllv_epi64(one, epi32::expand32_64_hi(rs2_registers));
                clear_register_mask_hi = clear_register_mask_hi & epi64::from_value(-2);

                const m256i cleared_lifetime_mask_lo = lifetime_mask_lo & ~clear_register_mask_lo;
                const m256i cleared_lifetime_mask_hi = lifetime_mask_hi & ~clear_register_mask_hi;

                /* Allocate rd */
                m256i rd = epi32::maskgatherz(this->rd_avx.data(), instr_offset, non_call_mask);
                const m256i original_rd = rd;
                const m256i rd_virtual_mask = (rd > 63);
                if (!epi32::is_zero(rd_virtual_mask)) {
                    /* FCVT.S.W and any generic float instructions need a float rd */
                    const m256i rd_float_mask = (instr == 0b1101000'00000'00000'111'00000'1010011) | ((instr & 0b1111111) == 0b1010011);
                    
                    /* If a float is required, mask all integer registers to 1's, else mask all float registers to 1's */
                    const m256i float_mask_lo = epi32::expand32_64_lo(rd_float_mask);
                    const m256i float_mask_hi = epi32::expand32_64_hi(rd_float_mask);
                    const m256i adjust_mask_lo = (float_mask_lo & epi64::from_value(0xFFFFFFFF)) | (~float_mask_lo & epi64::from_value(0xFFFFFFFFull << 32));
                    const m256i adjust_mask_hi = (float_mask_hi & epi64::from_value(0xFFFFFFFF)) | (~float_mask_hi & epi64::from_value(0xFFFFFFFFull << 32));
                   
                    const m256i adjusted_lifetime_mask_lo = cleared_lifetime_mask_lo | adjust_mask_lo;
                    const m256i adjusted_lifetime_mask_hi = cleared_lifetime_mask_hi | adjust_mask_hi;

                    AVX_ALIGNED auto rd_virtual_mask_array = epi32::extract(rd_virtual_mask);
                    AVX_ALIGNED std::array<uint64_t, 8> lifetime_mask_array {};
                    epi64::maskstore(&lifetime_mask_array[0], epi32::expand32_64_lo(rd_virtual_mask), adjusted_lifetime_mask_lo);
                    epi64::maskstore(&lifetime_mask_array[4], epi32::expand32_64_hi(rd_virtual_mask), adjusted_lifetime_mask_hi);
                    
                    for (uint32_t k = 0; k < 8; ++k) {
                        /* For every virtual register in rd, obtain a physical register to assign, and store into rd */
                        if (rd_virtual_mask_array[k]) {
                            const m256i mask = epi32::from_values((k == 0) ? 0 : -1, (k == 1) ? 0 : -1, (k == 2) ? 0 : -1, (k == 3) ? 0 : -1,
                                (k == 4) ? 0 : -1, (k == 5) ? 0 : -1, (k == 6) ? 0 : -1, (k == 7) ? 0 : -1);
                            rd = (rd & mask) | (~mask & ptilopsis::ffz(lifetime_mask_array[k]));
                            //std::cerr << format<32, true>(mask) << "\n";
                        }
                    }

                    /* If a register is 64, no register was found, so allocate a temporary based on type */
                    const m256i rd_overflow_mask = (rd == 64);
                    rd = (rd & ~rd_overflow_mask) | (37_m256i & (rd_overflow_mask & rd_float_mask)) | (5_m256i & (rd_overflow_mask & ~rd_float_mask));
                    rd = rd & non_call_mask;
                }
                
                /* If rs1 or rs2 were in use before this, they need to be marked as swapped */
                const m256i non_call_mask_lo = epi32::expand32_64_lo(non_call_mask);
                const m256i non_call_mask_hi = epi32::expand32_64_hi(non_call_mask);
                // TODO this actual double work w.r.t. clearing the registers
                const m256i rs1_registers_lo = epi32::expand32_64_lo(rs1_registers);
                const m256i rs1_registers_hi = epi32::expand32_64_hi(rs1_registers);

                const m256i rs1_mask_lo = _mm256_sllv_epi64(one, rs1_registers_lo) & non_call_mask_lo;
                const m256i rs1_mask_hi = _mm256_sllv_epi64(one, rs1_registers_hi) & non_call_mask_hi;

                /* If a register is not zero and it was in use previously, it has been swapped */
                const m256i rs1_swapped_mask_lo = ~(epi64::cmpeq(rs1_registers_lo, epi32::zero()) | epi64::cmpeq(lifetime_mask_lo & rs1_mask_lo, epi32::zero()));
                const m256i rs1_swapped_mask_hi = ~(epi64::cmpeq(rs1_registers_hi, epi32::zero()) | epi64::cmpeq(lifetime_mask_hi & rs1_mask_hi, epi32::zero()));
                const m256i rs1_swapped_mask = epi32::pack64_32(rs1_swapped_mask_lo, rs1_swapped_mask_hi);

                /* Isolate the swapped registers */
                const m256i rs1_swapped_registers = (rs1_registers & rs1_swapped_mask);

                /* Same but for rs2 */
                const m256i rs2_registers_lo = epi32::expand32_64_lo(rs2_registers);
                const m256i rs2_registers_hi = epi32::expand32_64_hi(rs2_registers);
                
                const m256i rs2_mask_lo = _mm256_sllv_epi64(one, rs2_registers_lo) & non_call_mask_lo;
                const m256i rs2_mask_hi = _mm256_sllv_epi64(one, rs2_registers_hi) & non_call_mask_hi;

                const m256i rs2_swapped_mask_lo = ~(epi64::cmpeq(rs2_registers_lo, epi32::zero()) | epi64::cmpeq(lifetime_mask_lo & rs2_mask_lo, epi32::zero()));
                const m256i rs2_swapped_mask_hi = ~(epi64::cmpeq(rs2_registers_hi, epi32::zero()) | epi64::cmpeq(lifetime_mask_hi & rs2_mask_hi, epi32::zero()));
                const m256i rs2_swapped_mask = epi32::pack64_32(rs2_swapped_mask_lo, rs2_swapped_mask_hi);

                const m256i rs2_swapped_registers = (rs2_registers & rs2_swapped_mask);

                /* Similar for rd, except use the cleared_lifetime_mask, since rs1 and rs2 are free at the moment rd is used */
                const m256i rd_registers_lo = epi32::expand32_64_lo(rd);
                const m256i rd_registers_hi = epi32::expand32_64_hi(rd);
                const m256i rd_mask_lo = _mm256_sllv_epi64(one, rd_registers_lo) & non_call_mask_lo;
                const m256i rd_mask_hi = _mm256_sllv_epi64(one, rd_registers_hi) & non_call_mask_hi;

                const m256i rd_swapped_mask_lo = ~(epi64::cmpeq(rd_registers_lo, epi32::zero()) | epi64::cmpeq(cleared_lifetime_mask_lo & rd_mask_lo, epi32::zero()));
                const m256i rd_swapped_mask_hi = ~(epi64::cmpeq(rd_registers_hi, epi32::zero()) | epi64::cmpeq(cleared_lifetime_mask_hi & rd_mask_hi, epi32::zero()));
                const m256i rd_swapped_mask = epi32::pack64_32(rd_swapped_mask_lo, rd_swapped_mask_hi);

                const m256i rd_swapped_registers = (rd & rd_swapped_mask);

                /* For rs1_swapped, rs2_swapped and rd_swapped, use the register numbers to gather the virtual register contained from original_register_state */
                // TODO reordering this is probably useful...
                if (!epi32::is_zero(rs1_swapped_mask)) {
                    const m256i virtual_rs1_regs = epi32::maskgather(epi32::from_value(-1), original_register_state.data(), rs1_swapped_registers, rs1_swapped_mask);
                    const m256i rs1_swapped_indices = virtual_rs1_regs - 64;
                    AVX_ALIGNED auto rs1_swapped_indices_array = epi32::extract(rs1_swapped_indices);
                    std::cerr << lifetime_mask_lo << '\n';
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int idx = rs1_swapped_indices_array[k]; idx >= 0) {
                            std::cerr << "swapping rs1 " << idx << '\n';
                            std::cerr << rs1_registers << '\n';
                            std::cerr << virtual_rs1_regs << '\n';
                            symbol_swapped[idx] = 0xFFFFFFFF;
                        }
                    }
                }

                if (!epi32::is_zero(rs2_swapped_mask)) {
                    const m256i virtual_rs2_regs = epi32::maskgather(epi32::from_value(-1), original_register_state.data(), rs2_swapped_registers, rs2_swapped_mask);
                    const m256i rs2_swapped_indices = virtual_rs2_regs - 64;
                    AVX_ALIGNED auto rs2_swapped_indices_array = epi32::extract(rs2_swapped_indices);
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int idx = rs2_swapped_indices_array[k]; idx >= 0) {
                            std::cerr << "swapping rs2 " << idx << '\n';
                            symbol_swapped[idx] = 0xFFFFFFFF;
                        }
                    }
                }

                if (!epi32::is_zero(rd_swapped_mask)) {
                    const m256i virtual_rd_regs = epi32::maskgather(epi32::from_value(-1), original_register_state.data(), rd_swapped_registers, rd_swapped_mask);
                    const m256i rd_swapped_indices = virtual_rd_regs - 64;
                    AVX_ALIGNED auto rd_swapped_indices_array = epi32::extract(rd_swapped_indices);
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int idx = rd_swapped_indices_array[k]; idx >= 0) {
                            std::cerr << "swapping rd " << idx << '\n';
                            symbol_swapped[idx] = 0xFFFFFFFF;
                        }
                    }
                }

                /* Set rd to be in use and store in the lifetime mask and store */
                epi64::maskstore(lifetime_masks.m256i(2 * j), non_call_mask_lo, cleared_lifetime_mask_lo | rd_mask_lo);
                epi64::maskstore(lifetime_masks.m256i((2 * j) + 1), non_call_mask_hi, cleared_lifetime_mask_hi | rd_mask_hi);

                /* Mark the physical registers rd as mapping to their virtual registers */
                {
                    AVX_ALIGNED auto rd_array = epi32::extract(rd);
                    AVX_ALIGNED auto original_rd_array = epi32::extract(original_rd);
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (original_rd_array[k]) {
                            //std::cout << original_rd_array[k] << " -> " << rd_array[k] << "\n\n";
                            register_state[(k * 64) + rd_array[k]] = original_rd_array[k];
                        }
                    }
                }

                /* Mask rs1 and rs2 as free if they're not rd */
                {
                    const m256i rs1_free_mask = (rs1_registers != rd) & non_call_mask;
                    AVX_ALIGNED auto rs1_free_array = epi32::extract((rs1_registers & rs1_free_mask) | (epi32::from_value(-1) & ~rs1_free_mask));
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int reg = rs1_free_array[k]; reg >= 0) {
                            register_state[(k * 64) + reg] = -1;
                        }
                    }
                }

                {
                    const m256i rs2_free_mask = (rs2_registers != rd) & non_call_mask;
                    AVX_ALIGNED auto rs2_free_array = epi32::extract((rs2_registers & rs2_free_mask) | (epi32::from_value(-1) & ~rs2_free_mask));
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int reg = rs2_free_array[k]; reg >= 0) {
                            register_state[(k * 64) + reg] = -1;
                        }
                    }
                }

                /* For virtual (>63) registers rs1, rs2 and rd, set symbol_registers to the physical reg and symbol_swapped to false */
                {
                    const m256i rs1_virtual_mask = (rd > 63);

                    AVX_ALIGNED auto rs1_array = epi32::extract(((rs1 - 64) & rs1_virtual_mask) | (epi32::from_value(-1) & ~rs1_virtual_mask));
                    AVX_ALIGNED auto rs1_registers_array = epi32::extract(rs1_registers);
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int idx = rs1_array[k]; idx >= 0) {
                            symbol_registers[idx] = rs1_registers_array[k];
                            symbol_swapped[idx] = 0;
                        }
                    }
                }

                {
                    const m256i rs2_virtual_mask = (rd > 63);

                    AVX_ALIGNED auto rs2_array = epi32::extract(((rs2 - 64) & rs2_virtual_mask) | (epi32::from_value(-1) & ~rs2_virtual_mask));
                    AVX_ALIGNED auto rs2_registers_array = epi32::extract(rs2_registers);
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int idx = rs2_array[k]; idx >= 0) {
                            symbol_registers[idx] = rs2_registers_array[k];
                            symbol_swapped[idx] = 0;
                        }
                    }
                }

                {
                    const m256i rd_virtual_mask = (original_rd > 63);

                    AVX_ALIGNED auto rd_array = epi32::extract(((original_rd - 64) & rd_virtual_mask) | (epi32::from_value(-1) & ~rd_virtual_mask));
                    AVX_ALIGNED auto rd_registers_array = epi32::extract(rd);
                    for (uint32_t k = 0; k < 8; ++k) {
                        if (int idx = rd_array[k]; idx >= 0) {
                            symbol_registers[idx] = rd_registers_array[k];
                            symbol_swapped[idx] = 0;
                        }
                    }
                }
            }
        }
    }

    avx_buffer<uint32_t> reverse_func_id_map { instr.size() };
    /* Scatter, so scalar... */
    for (uint32_t start : func_starts) {
        reverse_func_id_map[start] = 1;
    }

    avx_buffer<uint32_t> func_start_bools = reverse_func_id_map;

    /* Vectorized prefix sum again */
    m256i offset = epi32::zero();
    for (uint32_t i = 0; i < reverse_func_id_map.size_m256i(); ++i) {
        m256i vals = epi32::load(reverse_func_id_map.m256i(i));

        vals = vals + _mm256_slli_si256_dual<4>(vals);
        vals = vals + _mm256_slli_si256_dual<8>(vals);
        vals = vals + _mm256_slli_si256_dual<16>(vals);

        vals = vals + offset;

        offset = _mm256_permutevar8x32_epi32(vals, 7_m256i);

        epi32::store(reverse_func_id_map.m256i(i), vals);
    }

    /* Whether the virtual register was swapped */
    avx_buffer<uint32_t> spill_offsets { symbol_registers.size() };
    for (uint32_t i = 0; i < spill_offsets.size_m256i(); ++i) {
        m256i swapped_mask = epi32::load(symbol_swapped.m256i(i));
        epi32::store(spill_offsets.m256i(i), 1_m256i & swapped_mask);
    }

    // TODO avx segmented scan
    for (size_t i = 0; i < symbol_registers.size(); ++i) {
        /* Segmented scan: restart counting at every function */
        if (!func_start_bools[i]) {
            spill_offsets[i] = i > 0 ? spill_offsets[i - 1] : 0;
        }
    }

    avx_buffer<uint32_t> instr_counts { instr.size() };
    for (uint32_t i = 0; i < instr_counts.size_m256i(); ++i) {
        const m256i enabled = epi32::load(this->used_instrs_avx.m256i(i));

        const m256i rs1 = epi32::maskload(this->rs1_avx.m256i(i), enabled);
        const m256i rs1_swapped = epi32::maskgatherz(symbol_swapped.data(), rs1 - 64, rs1 > 63);

        const m256i rs2 = epi32::maskload(this->rs2_avx.m256i(i), enabled);
        const m256i rs2_swapped = epi32::maskgatherz(symbol_swapped.data(), rs2 - 64, rs2 > 63);

        const m256i rd = epi32::maskload(this->rd_avx.m256i(i), enabled);
        const m256i rd_swapped = epi32::maskgatherz(symbol_swapped.data(), rd - 64, rd > 63);

        const m256i base = 1_m256i & enabled;
        m256i res = base;
        res = res + (base & rs1_swapped) + (base & rs2_swapped) + (base & rd_swapped);

        epi32::maskstore(instr_counts.m256i(i), enabled, res);
    }

    /* Only preserve caller-saved registers */
    const m256i nonscratch = epi64::from_value(nonscratch_registers);
    for (uint32_t i = 0; i < preserve_masks.size_m256i(); ++i) {
        const m256i masks = nonscratch & epi64::load(preserve_masks.m256i(i));
        AVX_ALIGNED auto elements = epi64::extract<uint64_t>(masks);
        AVX_ALIGNED auto iota = epi64::extract<uint64_t>(_mm256_add_epi64(epi64::from_value(4 * i), epi64::from_values(0, 1, 2, 3)));
        for (uint32_t j = 0; j < 4; ++j) {
            uint64_t func_id = iota[j];
            if (func_id < func_starts.size()) {
                uint32_t preserved = std::popcount(elements[j]);
                
                instr_counts[func_starts[func_id] + 5] += preserved;
                instr_counts[func_starts[func_id] + function_sizes[func_id] - 6] += preserved;
            }
        }
    }

    /* Exclusive scan on the newly calculated offsets */
    offset = epi32::zero();
    avx_buffer<uint32_t> instr_offsets { instr.size() };
    for (uint32_t i = 0; i < instr_counts.size_m256i(); ++i) {
        m256i vals = epi32::loadu(&instr_counts[i * 8] - 1);

        vals = vals + _mm256_slli_si256_dual<4>(vals);
        vals = vals + _mm256_slli_si256_dual<8>(vals);
        vals = vals + _mm256_slli_si256_dual<16>(vals);

        vals = vals + offset;

        offset = _mm256_permutevar8x32_epi32(vals, 7_m256i);

        epi32::store(instr_offsets.m256i(i), vals);
    }

    uint32_t new_instr_count = (instr_offsets.size() == 0) ? 0 : (instr_offsets.back() + 1);

    avx_buffer<uint32_t> new_instr { new_instr_count };
    avx_buffer<uint32_t> new_rd { new_instr_count };
    avx_buffer<uint32_t> new_rs1 { new_instr_count };
    avx_buffer<uint32_t> new_rs2 { new_instr_count };
    avx_buffer<uint32_t> new_jt { new_instr_count };

    stack_sizes = avx_buffer<uint32_t> { func_count };

    for (uint32_t i = 0; i < instr.size_m256i(); ++i) {
        const m256i used_mask = epi32::load(this->used_instrs_avx.m256i(i));
        if (!epi32::is_zero(used_mask)) {
            /* Starting offsets for the instructions */
            const m256i base_offset = epi32::maskload(instr_offsets.m256i(i), used_mask);
            const m256i instr = epi32::maskload(this->instr.m256i(i), used_mask);

            const m256i rs1 = epi32::maskload(this->rs1_avx.m256i(i), used_mask);
            const m256i rs1_virtual_mask = (rs1 > 63);
            const m256i rs1_adjusted = rs1 - 64;
            m256i rs1_load_offset = epi32::from_value(-1);
            m256i rs1_stack_offset = epi32::zero();
            m256i allocated_rs1 = rs1 & ~rs1_virtual_mask;
            if (!epi32::is_zero(rs1_virtual_mask)) {
                const m256i swapped_mask = epi32::maskgatherz(symbol_swapped.data(), rs1_adjusted, rs1_virtual_mask);

                /* rs1 load offset is at the very start, keep all non-swapped ones at -1 */
                rs1_load_offset = ~swapped_mask | base_offset;
                rs1_stack_offset = epi32::maskgatherz(spill_offsets.data(), rs1_adjusted, rs1_virtual_mask);

                /* If rs1 was swapped, load into t0 or ft5 */
                const m256i need_float_mask = (instr == 0b1100000'00000'00000'111'00000'1010011) | ((instr & 0b1111111) == 0b1010011);

                const m256i actual_regs = epi32::maskgatherz(symbol_registers.data(), rs1_adjusted, rs1_virtual_mask & ~swapped_mask);

                /* Nonvirtual -> rs1 
                 * Virtual non-swapped -> actual_regs
                 * Virtual swapped int -> 5
                 * Virtual swapped float -> 37
                 * 
                 * swapped_mask is equivalent to swapped_mask & rs1_virtual_mask
                 */
                allocated_rs1 = allocated_rs1 | actual_regs | (5_m256i & (swapped_mask & ~need_float_mask));
                allocated_rs1 = allocated_rs1 | (37_m256i & (swapped_mask & need_float_mask));
            }

            const m256i rs2 = epi32::maskload(this->rs2_avx.m256i(i), used_mask);
            const m256i rs2_virtual_mask = (rs2 > 63);
            const m256i rs2_adjusted = rs2 - 64;
            m256i rs2_load_offset = epi32::from_value(-1);
            m256i rs2_stack_offset = epi32::zero();
            m256i allocated_rs2 = rs2 & ~rs2_virtual_mask;
            if (!epi32::is_zero(rs2_virtual_mask)) {
                const m256i swapped_mask = epi32::maskgatherz(symbol_swapped.data(), rs2_adjusted, rs2_virtual_mask);

                rs2_load_offset = ~swapped_mask | (base_offset + (1_m256i & (rs2_load_offset > 0)));
                rs2_stack_offset = epi32::maskgatherz(spill_offsets.data(), rs2_adjusted, rs2_virtual_mask);

                /* rs2 is never float except with generic float instructions */
                const m256i need_float_mask = ((instr & 0b1111111) == 0b1010011);

                const m256i actual_regs = epi32::maskgatherz(symbol_registers.data(), rs2_adjusted, rs2_virtual_mask & ~swapped_mask);
                allocated_rs2 = allocated_rs2 | actual_regs | (6_m256i & (swapped_mask & ~need_float_mask));
                allocated_rs2 = allocated_rs2 | (38_m256i & (swapped_mask & need_float_mask));
            }

            const m256i main_instr_offset = (base_offset + (1_m256i & (rs1_load_offset > 0)) + (1_m256i & (rs2_load_offset > 0)));

            const m256i rd = epi32::maskload(this->rd_avx.m256i(i), used_mask);
            const m256i rd_virtual_mask = (rd > 63);
            const m256i rd_adjusted = rd - 64;
            m256i rd_offset = epi32::from_value(-1);
            m256i rd_stack_offset = epi32::zero();
            m256i allocated_rd = rd & ~rd_virtual_mask;
            if (!epi32::is_zero(rd_virtual_mask)) {
                const m256i swapped_mask = epi32::maskgatherz(symbol_swapped.data(), rd_adjusted, rd_virtual_mask);

                rd_offset = ~swapped_mask | (main_instr_offset + 1);
                rd_stack_offset = epi32::maskgatherz(spill_offsets.data(), rd_adjusted, rd_virtual_mask);

                /* rd doesn't need to be loaded in beforehand */
                allocated_rd = allocated_rd | epi32::maskgatherz(symbol_registers.data(), rd_adjusted, rd_virtual_mask);
            }

            //const m256i func_id = epi32::maskload(reverse_func_id_map.m256i(i), used_mask) - 1;
            
            /* Construct the rs1 and rs2 loads */
            const m256i rs1_instr = (((rs1_stack_offset - 1) * 4) << 20) | 0b0000000'00000'00000'010'00000'0000011;
            const m256i rs2_instr = (((rs2_stack_offset - 1) * 4) << 20) | 0b0000000'00000'00000'010'00000'0000011;

            /* And the rd store */
            m256i rd_instr = 0b0000000'00000'00000'010'00000'0100011_m256i;
            {
                const m256i imm = ((rd_stack_offset - 1) * -4);
                rd_instr = rd_instr | ((imm & 0x1f) << 7) | ((imm & 0xfe) << 19);
            }

            /* rs1 */
            {
                AVX_ALIGNED auto rs1_load_offset_array = epi32::extract(rs1_load_offset);
                AVX_ALIGNED auto rs1_instr_array = epi32::extract(rs1_instr);
                for (uint32_t j = 0; j < 8; ++j) {
                    if (int idx = rs1_load_offset_array[j]; idx >= 0) {
                        new_instr[idx] = rs1_instr_array[j];
                        new_rd[idx] = 5;
                    }
                }
            }

            /* rs2 */
            {
                AVX_ALIGNED auto rs2_load_offset_array = epi32::extract(rs2_load_offset);
                AVX_ALIGNED auto rs2_instr_array = epi32::extract(rs2_instr);
                for (uint32_t j = 0; j < 8; ++j) {
                    if (int idx = rs2_load_offset_array[j]; idx >= 0) {
                        new_instr[idx] = rs2_instr_array[j];
                        new_rd[idx] = 6;
                    }
                }
            }

            /* Main instruction */
            {
                AVX_ALIGNED auto main_instr_offset_array = epi32::extract(main_instr_offset);
                AVX_ALIGNED auto allocated_rd_array = epi32::extract(allocated_rd);
                AVX_ALIGNED auto allocated_rs1_array = epi32::extract(allocated_rs1);
                AVX_ALIGNED auto allocated_rs2_array = epi32::extract(allocated_rs2);
                for (uint32_t j = 0; j < 8; ++j) {
                    if (int idx = main_instr_offset_array[j]; idx >= 0) {
                        new_instr[idx] = this->instr[(8 * i) + j];
                        new_rd[idx] = allocated_rd_array[j];
                        new_rs1[idx] = allocated_rs1_array[j];
                        new_rs2[idx] = allocated_rs2_array[j];
                        new_jt[idx] = instr_offsets[jt_avx[(8 * i) + j]];
                    }
                }
            }

            /* rd store */
            {
                AVX_ALIGNED auto rd_offset_array = epi32::extract(rd_offset);
                AVX_ALIGNED auto rd_instr_array = epi32::extract(rd_instr);
                AVX_ALIGNED auto allocated_rd_array = epi32::extract(allocated_rd);
                for (uint32_t j = 0; j < 8; ++j) {
                    if (int idx = rd_offset_array[j]; idx >= 0) {
                        new_instr[idx] = rd_instr_array[j];
                        new_rs1[idx] = allocated_rd_array[j];
                    }
                }
            }
        }
    }

    instr = new_instr;
    rd_avx = new_rd;
    rs1_avx = new_rs1;
    rs2_avx = new_rs2;
    jt_avx = new_jt;

    dump_instrs();
}
