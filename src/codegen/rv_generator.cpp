#include "codegen/rv_generator.hpp"

#include "codegen/rv_nodetype.hpp"

#include "utils.hpp"
#include "disassembler.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <range/v3/view/zip.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/range/conversion.hpp>

#include <codegen/depthtree.hpp>

#include <iomanip>
#include <limits>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <bit>
#include <bitset>
#include <sstream>
#include <vector>

#include <cmath>

#include <immintrin.h>

#ifdef _MSC_VER
#   include <intrin.h>
#   pragma warning(disable: 26451)
#endif

using namespace magic_enum::bitwise_operators;
using namespace magic_enum::ostream_operators;

rv_generator::rv_generator(const DepthTree& tree)
    : nodes          { tree.filledNodes() }
    , max_depth      { tree.maxDepth()    }
    , node_types     { nodes }
    , result_types   { nodes }
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
        return static_cast<rv_data_type>(v);
    });

    std::copy_n(tree.getParents().begin(),  nodes, parents.begin());
    std::copy_n(tree.getDepth().begin(),    nodes, depth.begin());
    std::copy_n(tree.getChildren().begin(), nodes, child_idx.begin());
    std::copy_n(tree.getNodeData().begin(), nodes, node_data.begin());

    std::fill_n(node_sizes.begin(), nodes, 0);
}

[[nodiscard]] static constexpr uint32_t extend(uint32_t x) {
    int32_t signed_x = x;
    /* sign extend by arithmetic right-shift */
    return static_cast<uint32_t>((signed_x << 20) >> 20);
}

std::ostream& rv_generator::print(std::ostream& os, bool disassemble) const {
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

#if 1
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
#endif

    if (disassemble && instructions) {
        std::vector<uint64_t> func_starts { this->func_starts.begin(), this->func_starts.end() };
        rvdisasm::disassemble(os, instructions, 0, func_starts);
    }

    os.flags(f);

    return os;
}

std::ostream& rv_generator::to_binary(std::ostream& os) const {
    return os.write(reinterpret_cast<const char*>(instructions.data()), instructions.size() * 4);
}

std::ostream& rv_generator::to_asm(std::ostream& os) const {
    for (uint32_t instr : instructions) {
        os << rvdisasm::instruction(instr) << '\n';
    }

    return os;
}

void rv_generator_st::process() {
    std::vector<std::pair<std::string_view, std::chrono::nanoseconds>> durations;
    using pair_type = decltype(durations)::value_type;
    durations.reserve(7);

    auto time = [this, &durations](std::string_view name, void(rv_generator_st::* func)()) {
        auto begin = std::chrono::steady_clock::now();
        (this->*func)();
        auto end = std::chrono::steady_clock::now();

        durations.emplace_back(name, end - begin);
    };

    time("preprocess", &rv_generator_st::preprocess);
    time("isn_cnt", &rv_generator_st::isn_cnt);
    time("isn_gen", &rv_generator_st::isn_gen);
    time("optimize", &rv_generator_st::optimize);
    time("regalloc", &rv_generator_st::regalloc);
    time("fix_jumps", &rv_generator_st::fix_jumps);
    time("postprocess", &rv_generator_st::postprocess);

    std::chrono::nanoseconds total = ranges::accumulate(durations | std::views::transform(&pair_type::second), std::chrono::nanoseconds {0});

    size_t name_length = 1 + std::ranges::max(durations | std::views::transform(&pair_type::first) | std::views::transform(&std::string_view::size));

    auto to_time_str = [](std::chrono::nanoseconds ns) {
        std::stringstream ss;
        ss << ns;
        return ss.str();
    };

    std::string total_str = to_time_str(total);

    std::vector<std::string> time_strings = durations | std::views::transform(&pair_type::second) | std::views::transform(to_time_str) | ranges::to_vector;
    size_t time_length = 1 + std::max<size_t>(total_str.size(), std::ranges::max(time_strings | std::views::transform(&std::string::size)));

    auto to_percent_str = [total](std::chrono::nanoseconds ns) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << ((ns.count() * 100.) / total.count()) << '%';
        return ss.str();
    };

    std::vector<std::string> percent_strings = durations | std::views::transform(&pair_type::second) | std::views::transform(to_percent_str) | ranges::to_vector;
    size_t percent_length = std::max(size_t { 12 }, 1 + std::ranges::max(percent_strings | std::views::transform(&std::string::size)));

    std::cerr << std::setw(name_length) << std::setfill(' ') << std::left << "Stage" << ' ' << std::setw(time_length) << std::setfill(' ') << "Duration" << " % of total\n"
        << std::string(name_length + time_length + 12, '-') << '\n';

    for (const auto& [pair, time, percent] : ranges::views::zip(durations, time_strings, percent_strings)) {
        std::cerr
            << rvdisasm::color::instr << std::setw(name_length)    << std::setfill(' ') << std::left  << pair.first
            << rvdisasm::color::extra << std::setw(time_length)    << std::setfill(' ') << std::right << time
                                      << std::setw(percent_length) << std::setfill(' ') << std::right << percent
            << rvdisasm::color::white << '\n';
    }

    std::cerr << std::string(name_length + time_length + 12, '-') << '\n'
        << rvdisasm::color::instr << std::setw(name_length) << std::setfill(' ') << std::left << "Total"
        << rvdisasm::color::extra << std::setw(time_length) << std::setfill(' ') << std::right << total_str
        << rvdisasm::color::white << '\n';
}

void rv_generator_st::dump_instrs() {
    std::cout << rvdisasm::color::extra << " == " << instructions.size() << " instructions ==             rd rs1 rs2 jt\n" << rvdisasm::color::white;
    size_t digits = static_cast<size_t>(std::log10(instructions.size()) + 1);
    for (size_t i = 0; i < instructions.size(); ++i) {
        std::string instr = rvdisasm::instruction(instructions[i], true);
        std::cout << rvdisasm::color::extra << std::dec << std::setw(digits) << std::setfill(' ') << i << rvdisasm::color::white << ": " << instr;
        int64_t pad = std::max<int64_t>(0, 32 - static_cast<int64_t>(instr.size()));
        for (int64_t j = 0; j < pad; ++j) {
            std::cout << ' ';
        }

        std::cout << std::setw(2) << std::setfill(' ') << rd[i] << ' '
            << std::setw(2) << std::setfill(' ') << rs1[i] << ' '
            << std::setw(2) << std::setfill(' ') << rs2[i] << ' '
            << std::setw(2) << std::setfill(' ') << jt[i] << '\n';
    }
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
                //node_data[i] += 10;

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
                    node_data[i] = ir_idx;

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
                    node_data[i] = (ir_idx - 8);
                }
            }
        }
    }

    /* replace_arg_lists */
    for (size_t i = 0; i < nodes; ++i) {
        if (i > 0 && node_types[i] == func_call_arg_list) {
            size_t prev = i - 1;
            int64_t stack_args = 0;
            if (node_types[prev] == func_call_arg){
                if (result_types[prev] == rv_data_type::FLOAT) {
                    int64_t int_args = static_cast<int64_t>(child_idx[prev]) - node_data[prev];
                    stack_args = std::max<int64_t>(int_args - 8, 0);
                }
            } else if (node_types[prev] == func_call_arg_on_stack) {
                stack_args = node_data[prev] + 1;
            }

            node_data[i] = static_cast<uint32_t>(stack_args);
        }
    }

    /* Comparison operator preprocessing */
    for (size_t i = 0; i < nodes; ++i) {
        // TODO: convert to a mapping operation (write in a separate memory buffer and then apply)
        /* If the parent of this node is a comparison operator and this node is a float */
        if (parents[i] >= 0
            && (node_types[parents[i]] & eq_expr) == eq_expr
            && result_types[i] == rv_data_type::FLOAT) {
            result_types[parents[i]] = rv_data_type::FLOAT;
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
            /* Parent-less top-level node is a statement list */
            auto parent_type = (parents[i] >= 0) ? node_types[parents[i]] : rv_node_type::statement_list;

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
    auto src_buf = avx_buffer<uint32_t>::iota(nodes);
    //std::vector<uint32_t> src = ;
    avx_buffer<uint32_t> func_decls(
        std::views::filter(src_buf, [this](uint32_t i) {
            return node_types[i] == func_decl;
        })
    );

    function_ids = avx_buffer<uint32_t>::zero(func_decls.size());
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

    function_sizes = avx_buffer<uint32_t>::iota(func_decls.size());
    std::ranges::transform(function_sizes, function_sizes.begin(), [&offsets](uint32_t i) {
        if (i == 0) {
            return offsets[0];
        }

        return offsets[i] - offsets[i - 1];
    });

    func_starts = rotated_offsets;
    func_ends = rotated_offsets;
    for (size_t i = 0; i < func_ends.size(); ++i) {
        func_ends[i] += function_sizes[i];
    }

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
    avx_buffer<uint32_t> idx_array = avx_buffer<uint32_t>::iota(nodes);

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

                    [[fallthrough]];

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
    for (size_t j = 0; j < max_depth + 1; ++j) {
        size_t current_depth = max_depth - j;

        uint32_t start_index = depth_starts[current_depth];
        uint32_t end_index = (current_depth == max_depth)
            ? static_cast<uint32_t>(nodes)
            : depth_starts[current_depth + 1];

        //std::cout << current_depth << " (" << start_index << ", " << end_index << "): " << idx_array.slice(start_index, end_index) << '\n';

        uint32_t level_size = end_index - start_index;

        uint32_t instr_idx = 0;

        auto instruction_indices = avx_buffer<int32_t>::zero(level_size * 4);
        auto parent_indices = avx_buffer<int32_t>::zero(level_size * 4);
        auto current_instructions = avx_buffer<uint32_t>::zero(level_size * 4);
        auto new_regs = avx_buffer<int64_t>::zero(level_size * 4);
        auto current_rd = avx_buffer<int64_t>::zero(level_size * 4);
        auto current_rs1 = avx_buffer<int64_t>::zero(level_size * 4);
        auto current_rs2 = avx_buffer<int64_t>::zero(level_size * 4);
        auto current_jt = avx_buffer<int64_t>::zero(level_size * 4);

        /* Iterate over all indices in this level
         * Note: this corresponds to the inner `let` of compile_tree, containing the mapping to compile_node
         */
        for (size_t idx : idx_array.slice(start_index, end_index)) {
            /* Compile each node */
            // uint32_t instr_offset = node_locations[idx];

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
                auto local_registers = registers;
                /* This corresponds to compile_node */
                if (has_instr_mapping[as_index(node_types[idx])][i][as_index(result_types[idx])]) {
                    //if (node_types[idx] == rv_node_type::func_decl_dummy) {
                    //    std::cerr << "instruction at " << (instr_offset + i) << "\n";
                    //}
                    uint32_t instr_in_buf = node_locations[idx] + i;
                    auto node_type = node_types[idx];
                    auto data_type = result_types[idx];
                    auto get_output_register = [this, idx, i, get_parent_arg_idx](rv_node_type node_type, rv_data_type resulting_type, uint32_t instr_in_buf) -> int64_t {
                        // TODO enum

                        /* The type of calculation required to get the output register */
                        auto calc_type = get_output_table[as_index(node_type)][i][as_index(resulting_type)];
                        switch (calc_type) {
                            case 1: return node_data[idx] + 10 /* func call arg: int
                                                                * func call arg float as int: all 
                                                                * -> so all args in int registers
                                                                */;
                            case 2: return node_data[idx] + 42; /* float call argument */
                            case 3: return 32; /* return values: float */
                            case 4: return 10; /* return values: invalid, void, int, int_ref, float_ref */
                            default: {
                                bool has_output_val = (get_parent_arg_idx(static_cast<uint32_t>(idx), i) != -1)
                                    || has_output[as_index(node_type)][i][as_index(resulting_type)];
                                if (has_output_val) {
                                    return instr_in_buf + 64;
                                }
                                return 0;
                            }
                        }
                    };

                    auto rd = get_output_register(node_type, result_types[idx], instr_in_buf);

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

                                return res;
                            }
                        } else {
                            return instr_in_buf;
                        }
                    };

                    auto instr_constant = [](rv_node_type type, uint32_t node_data, int64_t relative_offset) -> uint32_t {
                        auto calc_type = instr_constant_table[as_index(type)][relative_offset];

                        // TODO these are logical, explain
                        switch (calc_type) {
                            /* lui -> only upper 20 bits, in-place */
                            case 1: return (node_data - (extend(node_data & 0xFFF))) & 0xFFFFF000;
                            /* addi -> lower 12 bits, in the top 12 bits of the instruction word */
                            case 2: return (node_data & 0xFFF) << 20;
                            /* func call arg list, so setting up the stackframe for the call (?) */
                            case 3: return (-(4 * (static_cast<int32_t>(node_data) + 2))) << 20;
                            /* func arg on stack, so stack offset to above the current sp */
                            case 4: return (4 * node_data) << 20;
                            /* func arg list, so also an offset, but to below the current sp */
                            case 5: return (-(4 * static_cast<int32_t>(node_data))) << 20;
                            default: return 0;
                        }
                    };

                    // TODO per-node copy registers
                    //std::cout << "rd for " << idx << ": " << rd << '\n';
                    //std::cout <<  << " for " << idx << '\n';
                    auto instr_loc = get_instr_loc(static_cast<uint32_t>(idx), instr_in_buf, i, local_registers);
                    //std::cout << instr_loc << '\n';
                    instruction_indices[instr_idx] = static_cast<int32_t>(instr_loc);
                    parent_indices[instr_idx] = static_cast<int32_t>(get_parent_arg_idx(static_cast<uint32_t>(idx), i));
                    current_instructions[instr_idx] = instr_table[as_index(node_type)][i][as_index(data_type)];
                    current_instructions[instr_idx] |= instr_constant(node_type, node_data[idx], i);

                    auto node_get_instr_arg = [this](uint32_t node, avx_buffer<int64_t>& registers,
                        int64_t arg_no, int64_t instr_in_buf, int64_t relative_offset) -> int64_t {
                        auto calc_type = operand_table[as_index(node_types[node])][relative_offset][as_index(result_types[node])][arg_no];

                        switch (calc_type) {
                            // TODO make these make sense
                            /* simple relative args */
                            case 1: return registers[node * parent_idx_per_node + arg_no];
                            /* float as int arg and if-statement int_ref arg 2 */
                            case 2: return node_data[node] + 10;
                            /* if-statement float_ref arg 2 */
                            case 3: return node_data[node] + 42;
                            case 4: return registers[node * parent_idx_per_node + 1];
                            case 5: return registers[node * parent_idx_per_node + 1 - arg_no];
                            case 6: return registers[node * parent_idx_per_node];
                            case 7: return instr_in_buf + 64 - 1;
                            default: return 0;
                        }
                    };

                    auto instr_jt = [this](uint32_t node, int64_t relative_offset, avx_buffer<int64_t>& registers) -> int64_t {
                        auto calc_type = instr_jt_table[as_index(node_types[node])][relative_offset];

                        switch (calc_type) {
                            case 1: return registers[node * parent_idx_per_node + 1];
                            case 2: return registers[node * parent_idx_per_node + 1] + 1;
                            case 3: return registers[node * parent_idx_per_node + 2];
                            case 4: return registers[node * parent_idx_per_node + 2] + 1;
                            case 5: return registers[node * parent_idx_per_node];
                            case 6: return func_ends[node_data[node]] - 6;
                            case 7: return func_starts[node_data[node]];
                            default: return 0;
                        }
                    };

                    current_rd[instr_idx] = rd;
                    current_rs1[instr_idx] = node_get_instr_arg(static_cast<uint32_t>(idx), local_registers, 0, instr_in_buf, i);
                    current_rs2[instr_idx] = node_get_instr_arg(static_cast<uint32_t>(idx), local_registers, 1, instr_in_buf, i);
                    current_jt[instr_idx] = instr_jt(static_cast<uint32_t>(idx), i, local_registers);

                    /*std::cout << "node ";
                    if (idx < 10) {
                        std::cout << ' ';
                    }
                    std::cout << idx << ": "
                        << "rd: " << current_rd[instr_idx]
                        << ", rs1: " << current_rs1[instr_idx]
                        << ", rs2: " << current_rs2[instr_idx]
                        << ", jt: " << current_jt[instr_idx]
                        << " -> " << rvdisasm::instruction(current_instructions[instr_idx]) << '\n';*/
                    new_regs[instr_idx] = get_data_prop_value(static_cast<uint32_t>(idx), rd, instr_in_buf);
                } else if (i == 0) {
                    /* if no instruction present, propagate */
                    //std::cerr << "start\n";
                    instruction_indices[instr_idx] = -1;
                    parent_indices[instr_idx] = static_cast<int32_t>(get_parent_arg_idx(static_cast<uint32_t>(idx), 0));
                    current_instructions[instr_idx] = 0;
                    new_regs[instr_idx] = get_data_prop_value(static_cast<uint32_t>(idx), 0, node_locations[idx]);
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

        /* scatter data idx instrs */
        for (size_t i = 0; i < (level_size * 4); ++i) {
            if (instruction_indices[i] >= 0 && instruction_indices[i] < static_cast<int64_t>(instructions.size())) {
                instructions[instruction_indices[i]] = current_instructions[i];
                rd[instruction_indices[i]] = current_rd[i];
                rs1[instruction_indices[i]] = current_rs1[i];
                rs2[instruction_indices[i]] = current_rs2[i];
                jt[instruction_indices[i]] = static_cast<uint32_t>(current_jt[i]);
            }
        }

        /* scatter registers parent_idx new_regs */
        for (size_t i = 0; i < (level_size * 4); ++i) {
            if (parent_indices[i] >= 0 && parent_indices[i] < static_cast<int64_t>(registers.size())) {
                registers[parent_indices[i]] = new_regs[i];
            }
        }
    }
    //std::cout << func_ends << '\n';
    //dump_instrs();
}

void rv_generator_st::optimize() {
    //dump_instrs();
    auto initial_used_registers_length = 2 * instructions.size();
    avx_buffer<int64_t> initial_used_registers { initial_used_registers_length };
    for (size_t i = 0; i < instructions.size(); ++i) {
        initial_used_registers[2 * i] = rs1[i] - 64;
        initial_used_registers[2 * i + 1] = rs2[i] - 64;
    }
    auto used_registers = avx_buffer<bool>::zero(instructions.size());
    // TODO bounds checking?
    for (size_t i = 0; i < initial_used_registers_length; ++i) {
        if (initial_used_registers[i] >= 0 && initial_used_registers[i] < static_cast<int64_t>(instructions.size())) {
            used_registers[initial_used_registers[i]] = true;
        }
    }

    // TODO explain
    auto can_remove = [this](uint32_t instr, std::span<bool> used) {
        if (rd[instr] < 64) {
            return false;
        } else {
            return !used[rd[instr] - 64];
        }
    };

    auto has_side_effect = [](uint32_t instr_word) {
        switch (instr_word & 0b0000000'00000'00000'111'00000'1111111) {
            /* SW and FSW */
            case 0b0000000'00000'00000'010'00000'0100011:
            case 0b0000000'00000'00000'010'00000'0100111:
                return true;

            default:
                return false;
        }
    };

    bool cont = true;
    while (cont) {
        auto used_registers_cpy = used_registers;
        avx_buffer<int64_t> newly_used { instructions.size() * 2 };
        for (size_t i = 0; i < instructions.size(); ++i) {
            if (can_remove(static_cast<uint32_t>(i), used_registers_cpy)) {
                newly_used[2 * i] = rs1[i] - 64;
                newly_used[2 * i + 1] = rs2[i] - 64;
            } else {
                newly_used[2 * i] = -1;
                newly_used[2 * i + 1] = -1;
            }
        }

        auto new_used_registers = used_registers;
        for (size_t i = 0; i < newly_used.size(); ++i) {
            if (newly_used[i] >= 0 && newly_used[i] < static_cast<int64_t>(used_registers.size())) {
                new_used_registers[newly_used[i]] = false;
            }
        }

        avx_buffer<int64_t> side_effect_correct { instructions.size() * 2 };
        for (size_t i = 0; i < instructions.size(); ++i) {
            if (has_side_effect(instructions[i])) {
                side_effect_correct[2 * i] = rs1[i] - 64;
                side_effect_correct[2 * i + 1] = rs2[i] - 64;
            } else {
                side_effect_correct[2 * i] = -1;
                side_effect_correct[2 * i + 1] = -1;
            }
        }

        //std::cout << "sideeffects: " << side_effect_correct << '\n';

        auto result = new_used_registers;
        for (size_t i = 0; i < side_effect_correct.size(); ++i) {
            if (side_effect_correct[i] >= 0 && side_effect_correct[i] < static_cast<int64_t>(result.size())) {
                result[side_effect_correct[i]] = true;
            }
        }
        
        if (used_registers_cpy == result) {
            cont = false;
        }

        used_registers = result;
    }
    
    used_instrs = avx_buffer<bool> { instructions.size() };
    for (size_t i = 0; i < instructions.size(); ++i) {
        used_instrs[i] = rd[i] < 64 || used_registers[i];
    }
   // dump_instrs();
}

void rv_generator_st::regalloc() {
    //dump_instrs();
    uint32_t func_count = static_cast<uint32_t>(function_sizes.size());
    uint32_t max_func_size = std::ranges::max(function_sizes);
    stack_sizes = avx_buffer<uint32_t>::zero(func_count);
    /* Which register is currently used. Mark zero, ra, sp, gp, tp as used. */
    auto lifetime_masks = avx_buffer<uint64_t>::fill(func_count, 0b00000000'00000000'00000000'00000000'00000000'00000000'00000000'00011111);
    auto preserve_masks = avx_buffer<uint64_t>::zero(func_count);

    /* Map virtual reg to physical reg. If swapped, store the value on the stack after write */
    struct symbol_data {
        uint8_t reg;
        bool swapped;
    };
    auto symbol_registers = std::vector<symbol_data>(used_instrs.size());//) < symbol_data > ::fill(used_instrs.size(), {});
    /* Map physical to virtual registers */
    std::vector<avx_buffer<int64_t>> register_state(func_count, avx_buffer<int64_t>::fill(64, -1));

    auto current_func_offset = [](int64_t i, uint32_t size, uint32_t start) -> uint32_t {
        if (i >= size) {
            return 0xFFFFFFFF;
        } else {
            return static_cast<uint32_t>(i) + start;
        }
    };

    struct register_info {
        int64_t reg;
        symbol_data sym;
    };

    /* Lifetime analysis state at a specific instruction */
    struct lifetime_result {
        /* Usage mask after this instruction */
        uint64_t mask;
        /* rd, rs1 and rs2 for this instruction */
        std::array<register_info, 3> reg_info;
        /* Which physical registers were swapped. This acts as an array: every >=0 value is a physical register */
        std::array<int64_t, 64> swapped;
        /* Maps physical registers to the virtual registers currently occupying them */
        std::array<int64_t, 64> registers;
    };

    auto get_symbol_data = [](std::span<symbol_data> symbol_registers, int64_t reg) {
        if (reg < 64) {
            /* A predetermined register that can't be reassigned*/
            return symbol_data { .reg = static_cast<uint8_t>(reg), .swapped = false };
        } else {
            /* Data for the virtual register reg */
            return symbol_registers[reg - 64];
        }
    };

    auto needs_float_reg = [](uint32_t instr, uint32_t offset) {
        /* offset: rd = 0, rs1 = 1, rs2 = 2*/
        if (instr == 0b1100000'00000'00000'111'00000'1010011) {
            /* FCVT.W.S, float is needed for the source register */
            return offset != 0;
        } else if (instr == 0b1101000'00000'00000'111'00000'1010011) {
            /* FCVT.S.W, float is needed for the destination register */
            return offset == 0;
        } else {
            /* Any other float instr requires all float registers */
            return (instr & 0b1111111) == 0b1010011;
        }
    };

    auto lifetime_analyze_valid = [this, get_symbol_data, needs_float_reg](std::span<symbol_data> symbol_registers, uint32_t instr_offset, uint64_t lifetime_mask,
        std::span<int64_t> register_state, uint32_t func_start, uint32_t func_size) -> lifetime_result {

        auto is_call = [](uint32_t instr, uint32_t start, uint32_t size, uint32_t jt) {
            uint32_t end = start + size;
            return instr == 0b0000000'00000'00000'000'00001'1101111 && (jt < start || jt >= end);
        };
        uint32_t instr = instructions[instr_offset];
        if (is_call(instr, func_start, func_size, jt[instr_offset])) {
            /* Call instructions are a special case */
            std::array<register_info, 3> reg_info { register_info { -1, {} }, register_info { -1, {} }, register_info { -1, {} } };

            /* The following registers may be used:
             *  zero, ra, sp, gp, tp, s0-s11, fs0-fs11
             * All other registers should be marked as free before the call.
             */
            auto new_lifetime_mask = lifetime_mask & preserved_register_mask;

            /* Any of these callee-saved registers that are in used must be saved */
            auto spilled_register_mask = lifetime_mask & ~preserved_register_mask;

            lifetime_result res { .mask = new_lifetime_mask, .reg_info = reg_info };
            for (size_t i = 0; i < 64; ++i) {
                if ((spilled_register_mask & (1ull << i)) != 0) {
                    /* Callee-saved register that is in-use will be swapped out before this call,
                     * and marked as swapped out afterwards.
                     */
                    res.swapped[i] = i;
                } else {
                    res.swapped[i] = -1;
                }

                if ((new_lifetime_mask & (1ull << i)) != 0) {
                    /* Copy the virtual-to-physical register mapping of the registers that don't need to be saved */
                    res.registers[i] = register_state[i];
                } else {
                    res.registers[i] = -1;
                }
            }

            return res;
        } else {

            /* Get the current state of rs1 and rs2 */
            auto old_rs1_data = get_symbol_data(symbol_registers, rs1[instr_offset]);
            auto old_rs2_data = get_symbol_data(symbol_registers, rs2[instr_offset]);

            /* If swapped = true, then the value was moved to the stack after it was previously written to,
             * so we need to load it in a temporary register, t0 and t1 for integers and f5 and f6 for floats.
             * If it wasn't swapped, just use the physical register it's in
             */
            uint8_t rs1_reg = old_rs1_data.reg;
            uint8_t rs2_reg = old_rs2_data.reg;
            if (old_rs1_data.swapped) {
                if (needs_float_reg(instructions[instr_offset], 1)) {
                    rs1_reg = 37;
                } else {
                    rs1_reg = 5;
                }
            }

            if (old_rs2_data.swapped) {
                if (needs_float_reg(instructions[instr_offset], 2)) {
                    rs2_reg = 38;
                } else {
                    rs2_reg = 6;
                }
            }

            auto clear_reg = [](int32_t reg, uint64_t mask) -> uint64_t {
                if (reg == 0) {
                    return mask;
                } else {
                    return mask & ~(1ull << reg);
                }
            };

            /* Mark the actual rs1 and rs2 as clear */
            auto cleared_lifetime_mask = clear_reg(rs2_reg, clear_reg(rs1_reg, lifetime_mask));

            auto find_free_reg = [](bool float_reg, uint64_t mask) -> uint32_t {
                /* If float, mark all int registers as occupied and vice versa */
                auto fixed_regs = mask | (0xFFFFFFFFull << (float_reg ? 0 : 32));

#ifdef _MSC_VER
                /* The first 1 in the NOT'ed mask means the first zero in the original */
                unsigned long r = 0;
                bool nonzero = _BitScanForward64(&r, ~fixed_regs);
                /* ~fixed_regs == 0 means all registers are taken*/
                if (!nonzero) {
                    return 64;
                }

                return r;
#else
                int res = __builtin_ffs(~fixed_regs);
                /* Not found */
                if (res == 0) {
                    return 64;
                }

                return res - 1;
#endif
            };

            /* The virtual destination register */
            int64_t rd_register = rd[instr_offset];
            if (rd_register >= 64) {
                /* Not predetermined, so it needs to be allocated */
                auto float_reg = needs_float_reg(instructions[instr_offset], 0);

                /* Use the mask where rs1 and rs2 are marked as clear, since rd can be rs1 or rs2 */
                auto rd_tmp = find_free_reg(float_reg, cleared_lifetime_mask);
                if (rd_tmp == 64) {
                    /* No registers available, so we'll use t0 and f5 again*/
                    if (float_reg) {
                        rd_register = 37;
                    } else {
                        rd_register = 5;
                    }
                } else {
                    /* New register found */
                    rd_register = rd_tmp;
                }
            }

            /* For rd, rs1 and rs2, if they were in use before this instruction,
             * mark them to be swapped out before this instruction, so the next
             * usage knows they are swapped out
             */
            int64_t swap_rd = -1;
            if (rd_register != 0 && ((cleared_lifetime_mask & (1ull << rd_register)) != 0)) {
                swap_rd = rd_register;
            }
            int64_t swap_rs1 = -1;
            if (rs1_reg != 0 && ((lifetime_mask & (1ull << rs1_reg)) != 0)) {
                swap_rs1 = rs1_reg;
            }
            int64_t swap_rs2 = -1;
            if (rs2_reg != 0 && ((lifetime_mask & (1ull << rs2_reg)) != 0)) {
                swap_rs2 = rs2_reg;
            }

            /* Set rd to be in use after this instruction */
            auto new_lifetime_mask = cleared_lifetime_mask | (1ull << rd_register);
            lifetime_result res { .mask = new_lifetime_mask };

            /* For rd, rs1 and rs2, store the virtual register they correspond to and the destination physical regsiter
             * If the register is predetermined, there is no virtual register so store -1
             */
            res.reg_info[0].reg = ( rd[instr_offset] >= 64) ? ( rd[instr_offset] - 64) : -1;
            res.reg_info[0].sym = { .reg = static_cast<uint8_t>(rd_register), .swapped = false };
            res.reg_info[1].reg = (rs1[instr_offset] >= 64) ? (rs1[instr_offset] - 64) : -1;
            res.reg_info[1].sym = { .reg = rs1_reg, .swapped = false };
            res.reg_info[2].reg = (rs2[instr_offset] >= 64) ? (rs2[instr_offset] - 64) : -1;
            res.reg_info[2].sym = { .reg = rs2_reg, .swapped = false };

            std::ranges::copy(register_state, res.registers.begin());

            /* Store which virtual registers are held by the physical registers used by rd, rs1 and rs2 */
            res.registers[rd_register] = rd[instr_offset];

            /* If rs1 and rs2 are not rd, mark them as free after this */
            if (rs1_reg != rd_register) {
                res.registers[rs1_reg] = -1;
            }
            if (rs2_reg != rd_register) {
                res.registers[rs2_reg] = -1;
            }

            std::ranges::fill(res.swapped, -1);
            res.swapped[0] = swap_rd;
            res.swapped[1] = swap_rs1;
            res.swapped[2] = swap_rs2;
            return res;
        };
    };

    auto lifetime_analyze = [this, lifetime_analyze_valid](std::span<symbol_data> symbol_registers, std::span<bool> enabled,
                                   uint32_t instr_offset, uint64_t lifetime_mask, std::span<int64_t> register_state, uint32_t func_start, uint32_t func_size) -> lifetime_result {
        std::array<register_info, 3> reg_info { register_info { -1, {} }, register_info { -1, {} }, register_info { -1, {} } };

        if (instr_offset == 0xFFFFFFFF || !enabled[instr_offset]) {
            /* If there's no valid instruction (past the end or optimized away) at the current position... */
            lifetime_result res { .mask = lifetime_mask, .reg_info = reg_info };

            /* No registers are swapped and all mappings stay the same */
            for (size_t i = 0; i < 64; ++i) {
                res.swapped[i] = -1;
                res.registers[i] = register_state[i];
            }

            return res;
        } else {
            /* Perform the actual analysis */
            return lifetime_analyze_valid(symbol_registers, instr_offset, lifetime_mask, register_state, func_start, func_size);
        }
    };

    /* Process 1 instruction per function */
    for (int64_t i = 0; i < max_func_size; ++i) {
        /* For every function calculate the instruction offset relative to the start we're currently manipulating, or 0xFFFFFFFF if we're past the end */
        auto old_offsets = avx_buffer<uint32_t>::zero(func_count);
        for (size_t j = 0; j < func_count; ++j) {
            old_offsets[j] = current_func_offset(i, function_sizes[j], func_starts[j]);
        }

        /* Analyze all instructions we need to analyze */

        /* Store a copy of the virtual to physical register mapping from before this instruction */
        auto reg_state_copy = register_state;

        /* At most rd, rs1 and rs2 are updated every instruction */
        std::vector<std::array<register_info, 3>> updated_symbols(func_count);

        std::vector<std::array<int64_t, 64>> swapped_registers(func_count);
        for (size_t j = 0; j < func_count; ++j) {
            auto res = lifetime_analyze(symbol_registers, used_instrs, old_offsets[j], lifetime_masks[j], register_state[j], func_starts[j], function_sizes[j]);

            lifetime_masks[j] = res.mask;
            updated_symbols[j] = res.reg_info;
            swapped_registers[j] = res.swapped;
            register_state[j] = res.registers;
        }

        std::vector<int64_t> swap_data_regs(func_count * 64, -1);
        std::vector<symbol_data> swap_data_sym(func_count * 64);
        for (size_t k = 0; k < func_count; ++k) {
            /* For every physical register */
            for (size_t j = 0; j < 64; ++j) {
                auto reg = swapped_registers[k][j];
                if (reg < 0) {
                    /* Not swapped out, do nothing */
                    swap_data_regs[(k * func_count) + j] = -1;
                } else {
                    /* Physical register j is swapped during this instruction */
                    
                    /* The virtual register being swapped */
                    swap_data_regs[(k * func_count) + j] = reg_state_copy[k][reg] - 64;

                    /* Reverse lookup for the physical register it's contained in */
                    swap_data_sym[(k * func_count) + j] = get_symbol_data(symbol_registers, reg_state_copy[k][reg]);

                    /* Mark it as swapped out*/
                    swap_data_sym[(k * func_count) + j].swapped = true;
                }
            }
        }

        /* All registers that were used */
        std::vector<register_info> symb_data(func_count * 3);
        for (size_t j = 0; j < func_count; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                symb_data[(j * 3) + k] = updated_symbols[j][k];
            }
        }
        std::vector<int64_t> symbol_offsets(swap_data_regs.size() + symb_data.size(), -1);
        std::vector<symbol_data> all_symbol_data(swap_data_sym.size() + symb_data.size());

        /* Gather all registers touched by being swapped */
        for (size_t j = 0; j < swap_data_regs.size(); ++j) {
            /* Virtual register being swapped */
            symbol_offsets[j] = swap_data_regs[j];

            /* The physical reg the virtual register is in */
            all_symbol_data[j] = swap_data_sym[j];
        }

        /* And all registers touched by being used */
        for (size_t j = 0; j < symb_data.size(); ++j) {
            symbol_offsets[swap_data_regs.size() + j] = symb_data[j].reg;
            all_symbol_data[swap_data_regs.size() + j] = symb_data[j].sym;
        }

        /* Update the mask keeping track of all the used registers in this function */
        for (size_t j = 0; j < preserve_masks.size(); ++j) {
            preserve_masks[j] |= lifetime_masks[j];
        }

        /* scatter symbol_registers symbol_offsets all_symbol_data */
        for (size_t j = 0; j < symbol_offsets.size(); ++j) {
            if (symbol_offsets[j] >= 0 && symbol_offsets[j] < static_cast<int64_t>(symbol_registers.size())) {
                /* Update the touched virtual register's info */
                symbol_registers[symbol_offsets[j]] = all_symbol_data[j];
            }
        }
    }

    for (size_t i = 0; i < preserve_masks.size(); ++i) {
        // TODO surely this should include s0?
        /* s1-s11, fs0-fs11 are callee-saved, they won't be considered "used" */
        preserve_masks[i] &= nonscratch_registers;
    }

    /* Actually bool, int used for scan operation */
    auto func_start_bools = avx_buffer<int16_t>::zero(instructions.size());
    for (auto start : func_starts) {
        func_start_bools[start] = 1;
    }

    /* Fore very instruction, what function index it belongs to */
    auto reverse_func_id_map = avx_buffer<int64_t>::zero(instructions.size());
    std::inclusive_scan(func_start_bools.begin(), func_start_bools.end(), reverse_func_id_map.begin());
    
    /* For every virtual reg, store whether it was swapped */
    auto spill_offsets = avx_buffer<int64_t>::zero(symbol_registers.size());
    for (size_t i = 0; i < symbol_registers.size(); ++i) {
        spill_offsets[i] = symbol_registers[i].swapped;
    }

    /* For every function, add all swapped registers */
    for (size_t i = 0; i < symbol_registers.size(); ++i) {
        /* Segmented scan: restart counting at every function */
        if (!func_start_bools[i]) {
            spill_offsets[i] = i > 0 ? spill_offsets[i - 1] : 0;
        }
    }

    auto count_instr = [this, get_symbol_data](uint32_t instr, std::span<symbol_data> symb_data, std::span<bool> enabled) -> int32_t {
        if (enabled[instr]) {
            /* The instruction itself */
            int32_t res = 1;

            /* Possible register swaps for all 3 registers */
            res += (get_symbol_data(symb_data, rd[instr]).swapped ? 1 : 0);
            res += (get_symbol_data(symb_data, rs1[instr]).swapped ? 1 : 0);
            res += (get_symbol_data(symb_data, rs2[instr]).swapped ? 1 : 0);
            return res;
        } else {
            return 0;
        }
    };

    auto count_instr_add_preserve = [this, func_count](std::span<uint64_t> preserve_masks, std::span<int64_t> counts) {
        for (size_t i = 0; i < func_count; ++i) {
            /* Number of registers that need to be preserved*/
            auto preserved = std::popcount(preserve_masks[i]);
            auto start = func_starts[i] + 5;
            auto end = func_starts[i] + function_sizes[i] - 6;

            /* Add space for the required loads and stores */
            counts[start] += preserved;
            counts[end] += preserved;
        }
    };

    /* For every instruction, calculate how many instructions are actually needed,
     * including instructions to load and store the arguments and the result
     */
    auto instr_counts = avx_buffer<int64_t>::zero(instructions.size());
    for (size_t i = 0; i < instructions.size(); ++i) {
        instr_counts[i] = count_instr(static_cast<uint32_t>(i), symbol_registers, used_instrs);
    }
    /* Add space for the instructions to store callee-saved registers */
    count_instr_add_preserve(preserve_masks, instr_counts);

    /* New instruction offsets that take spills into account */
    auto instr_offsets = avx_buffer<int64_t>::zero(instructions.size());
    std::exclusive_scan(instr_counts.begin(), instr_counts.end(), instr_offsets.begin(), 0ll);

    /* Total number of instructions */
    int64_t new_instr_count = (instr_offsets.size() == 0) ? 0 : (instr_offsets.back() + 1);

    auto new_instr = avx_buffer<uint32_t>::zero(new_instr_count);
    auto new_rd = avx_buffer<int64_t>::zero(new_instr_count);
    auto new_rs1 = avx_buffer<int64_t>::zero(new_instr_count);
    auto new_rs2 = avx_buffer<int64_t>::zero(new_instr_count);
    auto new_jt = avx_buffer<uint32_t>::zero(new_instr_count);

    auto new_indices = avx_buffer<int64_t>::zero(instructions.size() * 4);
    auto temp_instr = avx_buffer<uint32_t>::zero(instructions.size() * 4);
    auto temp_rd = avx_buffer<int64_t>::zero(instructions.size() * 4);
    auto temp_rs1 = avx_buffer<int64_t>::zero(instructions.size() * 4);
    auto temp_rs2 = avx_buffer<int64_t>::zero(instructions.size() * 4);
    auto temp_jt = avx_buffer<uint32_t>::zero(instructions.size() * 4);

    for (size_t i = 0; i < instructions.size(); ++i) {
        if (used_instrs[i]) {
            /* Instruction wasn't optimized out */

            /* New index of this instruction */
            auto base_offset = instr_offsets[i];
            int64_t rs1_load_offset = -1;
            uint32_t rs1_stack_offset = 0;
            /* If rs1 isn't predetermined and was swapped,, it must be loaded */
            if (rs1[i] >= 64 && symbol_registers[rs1[i] - 64].swapped) {
                rs1_load_offset = base_offset;
                rs1_stack_offset = static_cast<uint32_t>(spill_offsets[rs1[i] - 64]);
            }

            int64_t rs2_load_offset = -1;
            uint32_t rs2_stack_offset = 0;
            /* Same for rs2, but offset it by 1 if rs1 also had to be loaded */
            if (rs2[i] >= 64 && symbol_registers[rs2[i] - 64].swapped) {
                rs2_load_offset = base_offset + ((rs1_load_offset > 0) ? 1 : 0);
                rs2_stack_offset = static_cast<uint32_t>(spill_offsets[rs2[i] - 64]);
            }

            /* The relative position of the actual instruction */
            int64_t main_instr_offset = base_offset + ((rs1_load_offset > 0) ? 1 : 0) + ((rs2_load_offset > 0) ? 1 : 0);
            int64_t rd_offset = -1;
            uint32_t rd_stack_offset = 0;

            /* If rd needs to be stored we also need a store after the main instruction */
            if (rd[i] >= 64 && symbol_registers[rd[i] - 64].swapped) {
                rd_offset = main_instr_offset + 1;
                rd_stack_offset = static_cast<uint32_t>(spill_offsets[rd[i] - 64]);
            }

            /* Function index */
            auto func_id = reverse_func_id_map[i] - 1;

            /* If rd is predetermined, use that, else retrieve the correct physical reg */
            int64_t allocated_rd = rd[i];
            if (rd[i] >= 64) {
                allocated_rd = symbol_registers[rd[i] - 64].reg;
            }

            /* Same as with rd */
            int64_t allocated_rs1 = rs1[i];
            if (rs1[i] >= 64) {
                /* If rs1 was swapped it'll be loaded into either t0 or ft5 */
                if (symbol_registers[rs1[i] - 64].swapped) {
                    if (needs_float_reg(instructions[i], 1)) {
                        allocated_rs1 = 37;
                    } else {
                        allocated_rs1 = 5;
                    }
                } else {
                    /* Not swapped, already in a register */
                    allocated_rs1 = symbol_registers[rs1[i] - 64].reg;
                }
            }

            int64_t allocated_rs2 = rs2[i];
            if (rs2[i] >= 64) {
                /* If rs2 was swapped it'll be loaded into either t1 or ft6*/
                if (symbol_registers[rs2[i] - 64].swapped) {
                    if (needs_float_reg(instructions[i], 2)) {
                        allocated_rs2 = 38;
                    } else {
                        allocated_rs2 = 6;
                    }
                } else {
                    /* Already in a register */
                    allocated_rs2 = symbol_registers[rs2[i] - 64].reg;
                }
            }

            auto make_load = [](int64_t dest_reg, uint32_t stack_offset, uint32_t& instr, int64_t& rd, int64_t& rs1, int64_t& rs2, uint32_t& jt) {
                /* Construct immediate */
                uint32_t imm = (4 * (stack_offset - 1)) << 20; 
                
                /* lw */
                instr = 0b0000000'00000'00000'010'00000'0000011 | imm;
                rd = dest_reg;
                rs1 = 0;
                rs2 = 0;
                jt = 0;
            };

            auto make_store = [](int64_t src_reg, uint32_t stack_offset, uint32_t& instr, int64_t& rd, int64_t& rs1, int64_t& rs2, uint32_t& jt) {
                uint32_t imm = (-4 * (stack_offset - 1));
                uint32_t lower = (imm & 0x1f) << 7;
                uint32_t upper = (imm & 0xfe) << 19;

                /* sw */
                instr = 0b0000000'00000'00000'010'00000'0100011 | lower | upper;
                rd = 0;
                rs1 = src_reg;
                rs2 = 0;
                jt = 0;
            };

            /* load into temp regs */
            /* If needed, make an instruction to load rs1 */
            new_indices[(4 * i) + 0] = rs1_load_offset;
            /* lw into t0 */
            make_load(5, rs1_stack_offset + stack_sizes[func_id], temp_instr[4 * i], temp_rd[4 * i], temp_rs1[4 * i], temp_rs2[4 * i], temp_jt[4 * i]);

            /* lw into t1 */
            new_indices[(4 * i) + 1] = rs2_load_offset;
            make_load(6, rs2_stack_offset + stack_sizes[func_id],
                temp_instr[(4 * i) + 1], temp_rd[(4 * i) + 1], temp_rs1[(4 * i) + 1], temp_rs2[(4 * i) + 1], temp_jt[(4 * i) + 1]);

            /* Fields for the main instruction */
            new_indices[(4 * i) + 2] = main_instr_offset;
            temp_instr[(4 * i) + 2] = instructions[i];
            temp_rd[(4 * i) + 2] = allocated_rd;
            temp_rs1[(4 * i) + 2] = allocated_rs1;
            temp_rs2[(4 * i) + 2] = allocated_rs2;
            /* Translate the jt to the new offset */
            temp_jt[(4 * i) + 2] = static_cast<uint32_t>(instr_offsets[jt[i]]);

            //std::cout << i << ": " << rd[i] << ", " << rs1[i] << ", " << rs2[i] << '\n';
            //std::cout << "    " << allocated_rd << ", " << allocated_rs1 << ", " << allocated_rs2 << '\n';

            /* Store from the result */
            new_indices[(4 * i) + 3] = rd_offset;
            make_store(allocated_rd, rd_stack_offset + stack_sizes[func_id],
                temp_instr[(4 * i) + 3], temp_rd[(4 * i) + 3], temp_rs1[(4 * i) + 3], temp_rs2[(4 * i) + 3], temp_jt[(4 * i) + 3]);
        } else {
            new_indices[(4 * i) + 0] = -1;
            new_indices[(4 * i) + 1] = -1;
            new_indices[(4 * i) + 2] = -1;
            new_indices[(4 * i) + 3] = -1;
        }
    }

    /* Scatter */
    for (size_t i = 0; i < new_indices.size(); ++i) {
        if (new_indices[i] >= 0 && new_indices[i] < new_instr_count) {
            new_instr[new_indices[i]] = temp_instr[i];
            new_rd[new_indices[i]] = temp_rd[i];
            new_rs1[new_indices[i]] = temp_rs1[i];
            new_rs2[new_indices[i]] = temp_rs2[i];
            new_jt[new_indices[i]] = temp_jt[i];
        }
    }

    auto overflows = avx_buffer<uint32_t>::zero(func_count);

    //dump_instrs();

    instructions = new_instr;
    rd = new_rd;
    rs1 = new_rs1;
    rs2 = new_rs2;
    jt = new_jt;

    fix_func_tab(instr_offsets);

    //dump_instrs();

    auto scatter_indices = avx_buffer<int64_t>::zero(func_count * 4ull);
    auto scatter_opcodes = avx_buffer<uint32_t>::zero(func_count * 4ull);
    for (size_t i = 0; i < func_count; ++i) {
        uint32_t preserve_count = std::popcount(preserve_masks[i]);
        uint32_t stack_size = (stack_sizes[i] + overflows[i] + preserve_count + 2) * 4;
        uint32_t lower = (stack_size & 0xFFF) << 20;
        uint32_t upper = (stack_size - (extend(stack_size & 0xFFF))) & 0xFFFFF000;

        scatter_indices[(i * 4) + 0] = func_starts[i] + 2;
        scatter_opcodes[(i * 4) + 0] = 0b0000000'00000'00000'000'01000'0110111 | upper; /* lui x8, upper */
        scatter_indices[(i * 4) + 1] = func_starts[i] + 3;
        scatter_opcodes[(i * 4) + 1] = 0b0000000'00000'01000'000'01000'0010011 | lower; /* addi x8, x8, lower */
        scatter_indices[(i * 4) + 2] = func_starts[i] + function_sizes[i] - 6;
        scatter_opcodes[(i * 4) + 2] = 0b0000000'00000'00000'000'01000'0110111 | upper; /* lui x8, upper */
        scatter_indices[(i * 4) + 3] = func_starts[i] + function_sizes[i] - 5;
        scatter_opcodes[(i * 4) + 3] = 0b0000000'00000'01000'000'01000'0010011 | lower; /* addi x8, x8, lower */
    }

    auto preserve_indices = avx_buffer<int64_t>::zero(func_count * 64ull);
    auto preserve_opcodes = avx_buffer<uint32_t>::zero(func_count * 64ull);
    for (size_t i = 0; i < func_count; ++i) {
        auto preserve_stack_offset = (stack_sizes[i] + overflows[i] + 2) * 4;
        auto p_mask = preserve_masks[i];

        for (uint32_t j = 0; j < 64; ++j) {
            int64_t leading = std::popcount(p_mask & ((1ull << j) - 1));
            int64_t offset = -(preserve_stack_offset + leading * 4);
            uint32_t offset_high = (offset & 0xFE0u) << 25;
            uint32_t offset_low = (offset & 0x1Fu) < 7;
            uint32_t src = (j % 32) << 20;
            uint32_t store_const = offset_high | offset_low | src;
            /* fsw or sw */
            uint32_t opcode = (j >= 32) ? 0b0000000'00000'01000'010'00000'0100111 : 0b0000000'00000'01000'010'00000'0100011;
            if (p_mask & (1ull << j)) {
                preserve_indices[(i * 64) + j] = func_starts[i] + leading + 6;
                preserve_opcodes[(i * 64) + j] = opcode | store_const;
            } else {
                preserve_indices[(i * 64) + j] = -1;
            }
        }
    }

    auto load_indices = avx_buffer<int64_t>::zero(func_count * 64ull);
    auto load_opcodes = avx_buffer<uint32_t>::zero(func_count * 64ull);
    for (size_t i = 0; i < func_count; ++i) {
        auto preserve_stack_offset = (stack_sizes[i] + overflows[i] + 2) * 4;
        auto p_mask = preserve_masks[i];
        
        for (uint32_t j = 0; j < 64; ++j) {
            int64_t leading = std::popcount(p_mask & (((1ull << j)) - 1));
            int64_t offset = -(preserve_stack_offset + leading * 4);

            uint32_t load_offset = static_cast<uint32_t>(offset) << 20;
            uint32_t load_dst = (j % 32) << 7;
            uint32_t load_const = load_offset | load_dst;

            /* flw or lw */
            uint32_t opcode = (j >= 32) ? 0b0000000'00000'01000'010'00000'0000111 : 0b0000000'00000'01000'010'00000'0000011;

            if (p_mask & (1ull << j)) {
                load_indices[(i * 64) + j] = func_starts[i] + function_sizes[i] + leading - 7;
                load_opcodes[(i * 64) + j] = opcode | load_const;
            } else {
                load_indices[(i * 64) + j] = -1;
            }
        }
    }

    auto all_indices = avx_buffer<int64_t>::zero(func_count * 132);
    auto all_opcodes = avx_buffer<uint32_t>::zero(func_count * 132);
    for (size_t i = 0; i < (func_count * 4); ++i) {
        all_indices[i] = scatter_indices[i];
        all_opcodes[i] = scatter_opcodes[i];
    }

    for (size_t i = 0; i < (func_count * 64); ++i) {
        all_indices[(func_count * 4) + i] = preserve_indices[i];
        all_indices[(func_count * 68) + i] = load_indices[i];

        all_opcodes[(func_count * 4) + i] = preserve_opcodes[i];
        all_opcodes[(func_count * 68) + i] = load_opcodes[i];
    }
    for (size_t i = 0; i < all_indices.size(); ++i) {
        if (all_indices[i] >= 0 && all_indices[i] < static_cast<int64_t>(instructions.size())) {
            instructions[all_indices[i]] = all_opcodes[i];
            rd[all_indices[i]] = 0;
            rs1[all_indices[i]] = 0;
            rs2[all_indices[i]] = 0;
            jt[all_indices[i]] = 0;
        }
    }
    //dump_instrs();
}

void rv_generator_st::fix_func_tab(std::span<int64_t> instr_offsets) {
    /* Re-calculate function table based on new offsets */
    for (size_t i = 0; i < func_starts.size(); ++i) {
        uint32_t func_start = static_cast<uint32_t>(instr_offsets[func_starts[i]]);
        uint64_t func_end_loc = func_starts[i] + function_sizes[i];
        uint32_t func_end = static_cast<uint32_t>((func_end_loc >= instr_offsets.size()) ? (instr_offsets.back() + 1) : instr_offsets[func_end_loc]);
        auto func_size = func_end - func_start;

        func_starts[i] = func_start;
        func_ends[i] = func_end;
        function_sizes[i] = func_size;
    }
}

void rv_generator_st::fix_jumps() {
    //dump_instrs();
    auto is_jump = [](uint32_t instr) {
        /* jalr but not a return-from-function (jalr zero, 0(ra))*/
        return ((instr & 0b1111111) == 0b1100111) && (instr != 0b0000000'00000'00001'000'00000'1100111);
    };

    auto is_branch = [](uint32_t instr) {
        /* beq and friends */
        return (instr & 0b1111111) == 0b1100011;
    };

    auto instr_sizes = avx_buffer<int64_t>::zero(instructions.size());
    for (size_t i = 0; i < instructions.size(); ++i) {
        if (is_jump(instructions[i])) {
            instr_sizes[i] = 2;
        } else if (instructions[i] == 0) {
            /* Filter out any leftover null instructions */
            instr_sizes[i] = 0;
        } else {
            instr_sizes[i] = 1;
        }
    }
    auto instr_offsets = avx_buffer<int64_t>::zero(instructions.size());
    std::exclusive_scan(instr_sizes.begin(), instr_sizes.end(), instr_offsets.begin(), 0ll);
    int64_t instr_count = instr_sizes.back() + instr_offsets.back();

    auto new_instr = avx_buffer<uint32_t>::zero(instr_count);
    auto new_rd = avx_buffer<int64_t>::zero(instr_count);
    auto new_rs1 = avx_buffer<int64_t>::zero(instr_count);
    auto new_rs2 = avx_buffer<int64_t>::zero(instr_count);
    auto new_jt = avx_buffer<uint32_t>::zero(instr_count);

    for (size_t i = 0; i < instr_offsets.size(); ++i) {
        new_instr[instr_offsets[i]] = instructions[i];
        new_rd[instr_offsets[i]] = rd[i];
        new_rs1[instr_offsets[i]] = rs1[i];
        new_rs2[instr_offsets[i]] = rs2[i];
        new_jt[instr_offsets[i]] = jt[i];
    }

    for (int64_t i = 0; i < instr_count; ++i) {
        new_jt[i] = static_cast<uint32_t>(instr_offsets[new_jt[i]]);
    }

    auto offsets = avx_buffer<int64_t>::zero(instructions.size() * 2);
    auto opcodes = avx_buffer<uint32_t>::zero(instructions.size() * 2);
    auto temp_rd = avx_buffer<int64_t>::zero(instructions.size() * 2);
    auto temp_rs1 = avx_buffer<int64_t>::zero(instructions.size() * 2);
    auto temp_rs2 = avx_buffer<int64_t>::zero(instructions.size() * 2);

    for (size_t i = 0; i < instructions.size(); ++i) {
        int64_t new_index = instr_offsets[i];
        if (is_jump(new_instr[new_index])) {
            int64_t target = new_jt[new_index] * 4;
            uint32_t delta = static_cast<uint32_t>(target - (new_index * 4));
            uint32_t upper = (delta & 0xFFFFF000) >> 12;
            uint32_t lower = delta & 0xFFF;
            uint32_t sign = (delta >> 11) & 1;
            uint32_t upper_constant = upper + sign;

            /* auipc x1 */
            offsets[(2 * i) + 0] = new_index;
            opcodes[(2 * i) + 0] = 0b00001'0010111 | (upper_constant << 12);
            //std::cout << new_index << ": " << rvdisasm::instruction(opcodes[2 * i]) << "  ->  " << rd[new_index] << '\n';

            /* The actual jump */
            offsets[(2 * i) + 1] = new_index + 1;
            opcodes[(2 * i) + 1] = new_instr[new_index] | (lower << 20) | (0b00001u << 15);
        } else if (is_branch(new_instr[new_index])) {
            int64_t target = new_jt[new_index] * 4;
            uint32_t delta = static_cast<uint32_t>(target - (new_index * 4));
            uint32_t sign     = (delta >> 12) & 1;
            uint32_t bit_11   = (delta >> 11) & 1;
            uint32_t bit_10_5 = (delta >>  5) & 0b11111;
            uint32_t bit_1_4  = (delta >>  1) & 0b1111;

            offsets[2 * i] = new_index;
            opcodes[2 * i] = new_instr[new_index] | (sign << 31) | (bit_11 << 7) | (bit_10_5 << 25) | (bit_1_4 << 8);
            temp_rd[2 * i] = new_rd[new_index];
            temp_rs1[2 * i] = new_rs1[new_index];
            temp_rs2[2 * i] = new_rs2[new_index];

            offsets[(2 * i) + 1] = -1;
        } else {
            offsets[(2 * i) + 0] = -1;
            offsets[(2 * i) + 1] = -1;
        }
    }

    for (size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i] >= 0 && offsets[i] < instr_count) {
            new_instr[offsets[i]] = opcodes[i];
            new_jt[offsets[i]] = 0;
            new_rd[offsets[i]] = temp_rd[i];
            new_rs1[offsets[i]] = temp_rs1[i];
            new_rs2[offsets[i]] = temp_rs2[i];
        }
    }

    instructions = new_instr;
    rd = new_rd;
    rs1 = new_rs1;
    rs2 = new_rs2;
    jt = new_jt;

    fix_func_tab(instr_offsets);

    //dump_instrs();
}

void rv_generator_st::postprocess() {
    for (size_t i = 0; i < instructions.size(); ++i) {
        uint32_t rd  = static_cast<uint32_t>(this->rd[i])  & 0b11111;
        uint32_t rs1 = static_cast<uint32_t>(this->rs1[i]) & 0b11111;
        uint32_t rs2 = static_cast<uint32_t>(this->rs2[i]) & 0b11111;

        instructions[i] |= (rd << 7) | (rs1 << 15) | (rs2 << 20);
    }
}
