#include "codegen/rv_generator.hpp"

#include "codegen/rv_nodetype.hpp"

#include "utils.hpp"
#include "disassembler.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <codegen/depthtree.hpp>

#include <iomanip>
#include <limits>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <bit>
#include <bitset>

#include <cmath>

#include <immintrin.h>

#pragma warning(disable: 26451)

rv_generator::rv_generator(const DepthTree& tree)
    : nodes          { tree.filledNodes() }
    , max_depth      { tree.maxDepth()    }
    , node_types     { nodes }
    , result_types   { nodes}
    , parents        { nodes }
    , depth          { nodes }
    , child_idx      { nodes }
    , node_data      { nodes }
    , node_sizes     { nodes }
    , node_locations { nodes } {

    std::ranges::transform(tree.getNodeTypes(), node_types.begin(), [](uint8_t v) {
        return static_cast<rv_node_type>(pareas_to_rv_nodetype[v]);
    });

    std::ranges::transform(tree.getResultingTypes(), result_types.begin(), [](uint8_t v) {
        return static_cast<DataType>(v);
    });

    std::copy_n(tree.getParents().begin(),  nodes, parents.begin());
    std::copy_n(tree.getDepth().begin(),    nodes, depth.begin());
    std::copy_n(tree.getChildren().begin(), nodes, child_idx.begin());
    std::copy_n(tree.getNodeData().begin(), nodes, node_data.begin());

    std::fill_n(node_sizes.begin(), nodes, 0);
}

std::ostream& rv_generator::print(std::ostream& os) const {
    std::ios_base::fmtflags f { os.flags() };
    os << std::setfill(' ');

    auto max_digits = [nodes=nodes]<typename T>(const avx_buffer<T>& buf) {
        return static_cast<size_t>(
            1 + log10l(*std::max_element(buf.begin(), buf.end())));
    };

    auto node_digits = static_cast<size_t>(log10l(static_cast<long double>(nodes)) + 1);
    size_t depth_digits = max_digits(depth);
    size_t size_digits = max_digits(node_sizes);
    size_t data_digits = max_digits(node_data);
    size_t location_digits = max_digits(node_locations);

    for (size_t i = 0; i < nodes; ++i) {
        fmt::print(os, "Node {:>{}}", i, node_digits);
        fmt::print(os, ", type = {:>3}", as_index(node_types[i]));
        fmt::print(os, ", result = {:>9}", result_types[i]);
        fmt::print(os, ", parent = {:>{}}", parents[i], node_digits);
        fmt::print(os, ", depth = {:>{}}", depth[i], depth_digits);
        fmt::print(os, ", child_idx = {:>{}}", child_idx[i], node_digits + 1);
        fmt::print(os, ", size = {:>{}}", node_sizes[i], size_digits);
        fmt::print(os, ", data = {:>{}}", node_data[i], data_digits);
        fmt::print(os, ", loc = {:>{}}", node_locations[i], location_digits);

        os << '\n';
    }

    if (instructions) {
        rvdisasm::disassemble(os, instructions, 0);
    }

    os.flags(f);

    return os;
}

void rv_generator_st::process() {
    preprocess();
    isn_cnt();
    isn_gen();
}

void rv_generator_st::preprocess() {
    using enum rv_node_type;

    /* Function argument node preprocessing */
    for (size_t i = 0; i < nodes; ++i) {
        /* transform only func_args and func_call_args */
        if ((node_types[i] & func_arg) == func_arg) {
            /* node_data is the argument index, for this specific type,
             * child_idx is the index of the argument overall
             */

            /* Lower bit set and popcnt > 1 means either float_ref or float*/
            if ((result_types[i] & 0b001) && (result_types[i] & 0b110) && (node_data[i] < 8)) {
                /* 8 floating point registers are available for arguments,
                 * meaning the first 8 are always passed in these
                 */

                /* Adjust node data to represent actual register index */
                node_data[i] += 10;

            } else {
                /* Index of the integer register in which to place this argument */
                int32_t preceding_integers;
                if ((result_types[i] & 0b001) && (result_types[i] & 0b110)) {
                    /* All preceding non-float arguments are integer */
                    preceding_integers = child_idx[i] - node_data[i];
                    
                } else {
                    preceding_integers = node_data[i];
                }

                int32_t preceding_floats = child_idx[i] - preceding_integers;

                /* Integer Register index, logical integer register to place this value in */
                int32_t ir_idx = preceding_integers + std::max<int32_t>(preceding_floats - 8, 0);

                if (ir_idx < 8) {
                    /* Same register is used for integers and float-in-int-register */
                    node_data[i] = ir_idx + 10;

                    /* This argument is placed in an integer register */
                    if ((result_types[i] & 0b001) && (result_types[i] & 0b110)) {
                        /* A float in an integer register */
                        node_types[i] |= 0b01;// rv_node_type::func_arg_float_as_int;
                    }

                    /* Integer in integer register is a normal func_arg, so don't touch */

                } else {
                    /* Register overflow, argument is on the stack */
                    node_types[i] |= 0b10;// rv_node_type::func_arg_on_stack;

                    /* Every argument past the 8th takes 4 bytes */
                    node_data[i] = (ir_idx - 8) * 4;
                }
            }
        }
    }

    /* Comparison operator preprocessing */
    for (size_t i = 0; i < nodes; ++i) {
        // TODO: convert to a mapping operation (write in a separate memory buffer and then apply)
        /* If the parent of this node is a comparison operator and this node is a float */
        if ((node_types[parents[i]] & eq_expr) == eq_expr
            && result_types[i] == DataType::FLOAT) {
            result_types[parents[i]] = DataType::FLOAT;
        }
    }
}

void rv_generator_st::isn_cnt() {
    using enum rv_node_type;

    /* S(n) = S_b(n) + A(n) + C(n) */
    auto node_count = [this](uint32_t i) -> uint32_t {
        uint32_t base = node_size_mapping[as_index(node_types[i])][as_index(result_types[i])];
        uint32_t delta = 0;

        /* Previous node is last argument, child index + 1 is number of nodes */
        if (i > 0
            && node_types[i] == func_call_arg_list) {
            delta = 1;

            if ((node_types[i - 1] & func_call_arg) == func_call_arg) {
                delta += child_idx[i - 1] + 1;
            }
        }

        if (node_types[i] != invalid) {
            auto parent_type = node_types[parents[i]];

            if (child_idx[i] == 0 && (parent_type == if_statement || parent_type == if_else_statement)) {
                /* if and if_else, first child, this is the conditional node */
                delta += 1;
            } else if (child_idx[i] == 1 && (parent_type == if_else_statement || parent_type == while_statement)) {
                /* for if_else, this is the unconditional jump to jump over the second part,
                 * for while, this is the first unconditional jump
                 */
                delta += 1;
            }
        }
        return base + delta;
    };

    std::ranges::transform(std::views::iota(uint32_t{ 0 }, nodes), node_sizes.begin(), node_count);

    /* Compute the actual instruction locations with an exclusive prefix sum */
    std::exclusive_scan(node_sizes.begin(), node_sizes.end(), node_locations.begin(), 0);

    /* not needed as we can implement an exclusive prefix sum directly */
    //std::ranges::rotate(node_locations, node_locations.end() - 1);

    //node_locations[0] = 0;

    /* instr_count_fix
     * TODO probably not needed anymore
     */
    for (size_t i = 0; i < nodes; ++i) {
        if (node_types[i] == invalid) {
            // std::cout << "invalid node " << i << '\n';
            node_locations[i] = 0xFFFFFFFF;
        }
    }

    /* instr_count_fix_post */
    avx_buffer<int32_t> fix_idx { nodes };
    avx_buffer<uint32_t> fix_offsets { nodes };

    for (size_t i = 0; i < nodes; ++i) {
        if (parents[i] == -1 /* INVALID_NODE_IDX */) {
            fix_idx[i] = -1;
            fix_offsets[i] = 0;
        } else {
            auto parent_type = node_types[parents[i]];

            /* Any conditional statement, lower bit represents the child index of the conditional node */
            if (((parent_type & if_statement) == if_statement) && child_idx[i] == static_cast<int32_t>(parent_type & 1)) {
                fix_idx[i] = parents[i];
                fix_offsets[i] = node_locations[i] + node_size_mapping[as_index(node_types[i])][as_index(result_types[i])];

            } else if ((node_types[i] & func_call_arg) == func_call_arg) {
                fix_idx[i] = static_cast<int32_t>(i);

                /* instr_call_arg_offset */
                fix_offsets[i] = node_locations[parents[i]] + child_idx[i] + 1;
            } else {
                fix_idx[i] = -1;
                fix_offsets[i] = 0;
            }
        }
    }

    /* scatter */
    for (size_t i = 0; i < fix_idx.size(); ++i) {
        if (fix_idx[i] >= 0) {
            node_locations[fix_idx[i]] = fix_offsets[i];
        }
    }

    /* Function table generation */
    auto func_offsets = avx_buffer<uint32_t>::iota(nodes);

    avx_buffer<uint32_t> func_decls {
        std::views::filter(avx_buffer<uint32_t>::iota(nodes), [this](uint32_t i) {
            return node_types[i] == func_decl;
        })
    };

    avx_buffer<uint32_t> function_ids { func_decls.size() };
    std::ranges::transform(func_decls, function_ids.begin(), [this](uint32_t i) {
        return node_data[i];
    });

    avx_buffer<uint32_t> offsets { func_decls.size() };
    std::ranges::transform(func_decls, offsets.begin(), [this](uint32_t i) {
        return node_locations[i] + 6;
    });

    avx_buffer rotated_offsets { offsets };
    std::ranges::rotate(rotated_offsets, rotated_offsets.end() - 1);
    rotated_offsets[0] = 0;

    auto function_sizes = avx_buffer<uint32_t>::iota(func_decls.size());
    std::ranges::transform(function_sizes, function_sizes.begin(), [&offsets](uint32_t i) {
        if (i == 0) {
            return offsets[0];
        }

        return offsets[i] - offsets[i - 1];
    });

    return;
    for (size_t i = 0; i < func_decls.size(); ++i) {
        std::cout << "Function " << function_ids[i]
                  << " of node " << func_decls[i]
                  << " at " << rotated_offsets[i]
                  << " of size " << function_sizes[i] << '\n';
    }
}

void rv_generator_st::isn_gen() {
    size_t word_count = node_locations.back();

    // TODO Use i.e. radix sort for parallel sorting
    avx_buffer<uint32_t> idx_array { std::views::iota(uint32_t { 0 }, nodes) };

    /* Sort in separate array to obtain a mapping to the original */
    std::ranges::sort(idx_array, std::ranges::less{}, [this](uint32_t i) {
        return depth[i];
    });

    /* Find the indices at which a new level starts */
    auto depth_starts = avx_buffer<uint32_t>::iota(nodes);

    auto removed = std::ranges::remove_if(depth_starts, [this, &idx_array](uint32_t i) {
        return !(i == 0 || depth[idx_array[i]] != depth[idx_array[i - 1]]);
    });

    depth_starts.shrink_to(std::distance(depth_starts.begin(), removed.begin()));

    /* Data propagation (ancillary) buffer */
    auto registers = avx_buffer<int64_t>::zero(nodes * parent_idx_per_node);

    //std::cout << idx_array << '\n';

    instructions = avx_buffer<uint32_t>::zero(word_count);
    rd = avx_buffer<int64_t>::zero(word_count);
    rs1 = avx_buffer<int64_t>::zero(word_count);
    rs2 = avx_buffer<int64_t>::zero(word_count);
    jt = avx_buffer<uint32_t>::zero(word_count);

    /* Retrieve the index to which we can write our results in the ancillary buffer (registers) */
    auto get_parent_arg_idx = [this](uint32_t node, uint32_t instr_offset) -> int64_t {
        auto parent_arg_idx = [&] {
            /* par. 3.3.1 -> n * x + i */
            return parents[node] * parent_idx_per_node + child_idx[node];
        };

        /* Use the lookup */
        auto sub = [&]() -> int64_t {
            // TODO make enum
            /* Retrieve the type of lookup we use */
            int8_t calc_type = parent_arg_idx_lookup[as_index(node_types[node])][instr_offset][as_index(result_types[node])];

            switch (calc_type) {
                case 1:
                    /* 1 = just use the child index as a relative index */
                    return parent_arg_idx();

                case 2:
                    /* 2 = use child index as relative index if the current node has a return value/type */
                    if (as_index(result_types[node]) > 1) {
                        return parent_arg_idx();
                    }

                    return -1;

                default:
                    return -1;
            }
        };

        if (parents[node] == -1) {
            /* a top-level node, use defeault method */
            return sub();
        } else {
            auto parent = parents[node];

            /* Current node is the non-conditional node of a control flow statement, place their results
             * in their corresponding relative indices of the parent
             */
            if (child_idx[node] > 0 && (node_types[parent] == rv_node_type::if_statement || node_types[parent] == rv_node_type::if_else_statement)) {
                return parent_arg_idx();
            } else if ((child_idx[node] == 0 || child_idx[node] == 2) && node_types[parent] == rv_node_type::while_statement) {
                return parent_arg_idx();
            } else {
                /* Any other node: use default lookup */
                return sub();
            }
        }
    };

    auto get_data_prop_value = [this](uint32_t node, int64_t rd, int64_t instr_offset) {
        if (parents[node] == -1) {
            /* no parent node */
            return rd;
        } else {
            auto parent_type = node_types[parents[node]];

            /* current node's instruction offset + the number of instructions this node should require
             * this is the current instruction's unique result register in our infinite register architecture
             * (is this not the past-the-end position of the last related instruction?)
             */
            auto instr_no = instr_offset + node_size_mapping[as_index(node_types[node])][as_index(result_types[node])];

            if (parent_type == rv_node_type::if_statement || parent_type == rv_node_type::if_else_statement) {
                /* if the current node is the non-conditional child */
                if (child_idx[node] == 1 || child_idx[node] == 2) {
                    return instr_no;
                } else {
                    return rd;
                }
            } else if (parent_type == rv_node_type::while_statement) {
                /* same as with if/if_else */
                if (child_idx[node] == 0 || child_idx[node] == 2) {
                    return instr_no;
                } else {
                    return rd;
                }
            } else {
                return rd;
            }
        }
    };

    // TODO more parallel
    /* For every level*/
    for (size_t i = 0; i < max_depth + 1; ++i) {
        size_t current_depth = max_depth - i;

        uint32_t start_index = depth_starts[current_depth];
        uint32_t end_index = (current_depth == max_depth)
            ? static_cast<uint32_t>(nodes)
            : depth_starts[current_depth + 1];

        //std::cout << current_depth << " (" << start_index << ", " << end_index << "): " << idx_array.slice(start_index, end_index) << '\n';

        uint32_t level_size = end_index - start_index;

        auto instruction_indices = avx_buffer<int32_t>::zero(level_size * 4);
        auto parent_indices = avx_buffer<int32_t>::zero(level_size * 4);
        auto current_instructions = avx_buffer<uint32_t>::zero(level_size * 4);
        auto new_regs = avx_buffer<int64_t>::zero(level_size * 4);

        uint32_t instr_idx = 0;

        /* Iterate over all indices in this level */
        for (size_t idx : idx_array.slice(start_index, end_index)) {
            /* Compile each node */
            uint32_t instr_offset = node_locations[idx];

            /* Every node generates at most 4 instructions
             * has_instr_mapping[node_type][offset][result_type]
             *   Whether a node of the specific node_type has an instruction at the specified offset in [0, 3],
             *     given it's result_type is as specified.
             *   The order for types is: [Invalid, Void, Int, Float, Int_ref, Float_ref].
             *      For example, all literals cost 2 instructions, except floats, which can cost 3.
             *  
             */

            /* compile_node returns a tuple (idx, parent_idx, instrs, new_regs)
             * types (int, int, instruction, int)
             * where idx scatters instrs and parent_idx scatters new_regs
             * instrs contains opcodes, func fields and immediates
             * new_regs for propagation
             *  
             * 
             * These operations happen for each level
             * 
             * Each instruction has at most 4 instructions, for every possible instruction check whether
             *  an actual instruction is placed at that relative index.
             * 
             */

            // registers is the propagation buffer, a copy is used for each compile_node, and then it's updated
            // after each level
            for (uint32_t i = 0; i < 4; ++i, ++instr_idx) {
                if (has_instr_mapping[as_index(node_types[idx])][i][as_index(result_types[i])]) {
                    //std::cerr << "instruction at " << instr_offset << "\n";
                    uint32_t instr_in_buf = instr_idx + i;
                    auto node_type = node_types[idx];
                    auto data_type = result_types[idx];
                    auto get_output_register = [this, idx, i, instr_in_buf](rv_node_type node_type, DataType resulting_type) -> int64_t {
                        // TODO enum
                        /* the type of calculation required to get the output register */
                        auto calc_type = get_output_table[as_index(node_type)][i][as_index(resulting_type)];
                        switch (calc_type) {
                            case 1: return node_data[idx] + 10 /* func call arg: int
                                                                * func call arg float as int: all 
                                                                * -> so all args in int registers
                                                                */;
                            case 2: return node_data[idx] + 42; /* float call argument */
                            case 3: return 32; /* return values: float */
                            case 4: return 10; /* return values: invalid, void, int, int_ref, float_ref */
                            default:
                                if (as_index(resulting_type) > 1) {
                                    return instr_in_buf + 64;
                                }
                        }
                    };

                    auto rd = get_output_register(node_type, result_types[idx]);

                    auto get_instr_loc = [this](uint32_t node, uint32_t instr_in_buf, uint32_t relative_offset, avx_buffer<int64_t>& registers) -> int64_t {
                        if (relative_offset == 1 && (node_types[node] == rv_node_type::if_else_statement)) {
                            return registers[node * parent_idx_per_node + 1];
                        } else if (relative_offset == 1 && node_types[node] == rv_node_type::while_statement) {
                            return registers[node * parent_idx_per_node + 2];
                        } else if ((relative_offset >= 2 && node_types[node] == rv_node_type::func_decl_dummy) || node_types[node] == rv_node_type::func_decl) {
                            return instr_in_buf + 2;
                        } else if (relative_offset == 1 && (node_types[node] == rv_node_type::func_call_arg_list)) {
                            if (node == 0) {
                                return instr_in_buf + 2;
                            } else {
                                auto prev_node = node - 1;
                                auto res = instr_in_buf + 2;

                                if (node_types[prev_node] == rv_node_type::func_call_arg
                                    || node_types[prev_node] == rv_node_type::func_call_arg_float_as_int
                                    || node_types[prev_node] == rv_node_type::func_call_arg_on_stack) {
                                    res += child_idx[prev_node];
                                }
                            }
                        } else {
                            return instr_in_buf;
                        }
                    };

                    auto extend = [](uint32_t x) {
                        int32_t signed_x = x;
                        /* sign extend by arithmetic right-shift */
                        return static_cast<uint32_t>((signed_x << 20) >> 20);
                    };

                    auto instr_constant = [extend](rv_node_type type, uint32_t node_data, int64_t relative_offset) -> uint32_t {
                        auto calc_type = instr_constant_table[as_index(type)][relative_offset];

                        // TODO these are logical, explain
                        switch (calc_type) {
                            case 1: return (node_data - (extend(node_data & 0xFFF))) & 0xFFFFF000;
                            case 2: return (node_data & 0xFFF) << 20;
                            case 3: return (-(4 * (node_data + 2))) << 20;
                            case 4: return (4 * node_data) << 20;
                            case 5: return (-(4 * node_data)) << 20;
                            default: return 0;
                        }
                    };

                    // TODO per-node copy registers
                    //std::cout << "rd for " << idx << ": " << rd << '\n';
                    //std::cout <<  << " for " << idx << '\n';
                    auto instr_loc = get_instr_loc(idx, node_locations[idx] + i, i, registers);

                    instruction_indices[instr_idx] = instr_loc;
                    parent_indices[instr_idx] = get_parent_arg_idx(idx, i);
                    current_instructions[instr_idx] = instr_table[as_index(node_type)][i][as_index(data_type)];
                    current_instructions[instr_idx] |= instr_constant(node_type, node_data[idx], i);

                    std::cout << "node " << idx << ": " << std::bitset<32>(current_instructions[instr_idx])
                            << " -> " << rvdisasm::instruction(current_instructions[instr_idx]) << '\n';
                    new_regs[instr_idx] = get_data_prop_value(idx, rd, node_locations[idx] + i);
                } else if (i == 0) {
                    /* if no instruction present, propagate */
                    //std::cerr << "start\n";
                    instruction_indices[instr_idx] = -1;
                    parent_indices[instr_idx] = get_parent_arg_idx(idx, 0);
                    current_instructions[instr_idx] = 0;
                    new_regs[instr_idx] = get_data_prop_value(idx, 0, instr_idx);
                    //std::cerr << "parent_index: " << parent_indices[instr_idx] << " for " << idx << ": " << new_regs[instr_idx] << '\n';
                } else {
                    // std::cerr << "empty\n";
                    instruction_indices[instr_idx] = -1;
                    parent_indices[instr_idx] = -1;
                    current_instructions[instr_idx] = 0;
                    new_regs[instr_idx] = 0;
                }
            }
        }
    }
}
