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

#include <cmath>

#include <immintrin.h>

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

    /* S_b(n) */
    //for (size_t i = 0; i < nodes; ++i) {
    //    node_sizes[i] = node_size_mapping[as_index(node_types[i])][as_index(result_types[i])];
    //}
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
            /* if/if_else */
            if (child_idx[i] == 0 && (node_types[parents[i]] & 1) == 0) {  // NOLINT(bugprone-branch-clone)
                delta += 1;

                /* if_else: unconditional jump to jump over else block
                 * while: conditional jump in conditional
                 */
            } else if (child_idx[i] == 1 && (node_types[parents[i]] & 0b11) != 0) {
                delta += 1;
            }
        }

        return base + delta;
    };

    
    std::ranges::transform(std::views::iota(uint32_t{ 0 }, nodes), node_sizes.begin(), node_count);

    /*
     * A(n)
     * A register move is a single instruction:
     *  - mv rd, rs
     *  - fmv.s rd, rs
     *  - fmv.x.w rd, rs1
     *
     * A store is also a single instruction:
     *  - sw rs2, rs1 + offset
     *  - fsw rs2, rs1 + offset
     *
     * When encountering a func_arg_list, check if the previous node is a func_call_arg,
     *  if so add the number of arguments contained in the list to the call's node size.
     *
     *  Also add 1 instruction to allocate stack space for the arguments (always allocated, may be 0 bytes)
     */
    /*for (size_t i = 0; i < nodes; ++i) {
        if (node_types[i] == func_call_arg_list
            && (node_types[i - 1] & func_call_arg) == func_call_arg) {
            
            node_sizes[i] += (child_idx[i - 1] + 1 + 1);
        }
    }*/

    /*
     * C(n)
     *
     * Check if the parent node is a branching instruction
     */
    //for (size_t i = 0; i < nodes; ++i) {
    //    if ((node_types[parents[i]] & if_statement) == if_statement

            /*
             * if, if/else and while are specified so that the lower bit indicates the index of the conditional,
             * meaning this checks whether the current node is the conditional node belonging to the parent branch op.
             */
    //        && child_idx[i] == as_index(node_types[parents[i]] & 1)) {
            /* This node is the conditional for a if, if_else or while */
    //        node_sizes[i] += 1;
    //    }
    //}

    /* Add padding for an unconditional jump after the if-block of an if_else */
    //for (size_t i = 0; i < nodes; ++i) {
    //    if (node_types[parents[i]] == if_else_statement
    //        && child_idx[i] == 1) {
    //        node_sizes[i] += 1;
    //    }
    //}

    /* Compute the actual instruction locations with an exclusive prefix sum */
    std::exclusive_scan(node_sizes.begin(), node_sizes.end(), node_locations.begin(), 0);

    std::ranges::rotate(node_locations, node_locations.end() - 1);

    node_locations[0] = 0;
}

void rv_generator_st::isn_gen() {
    size_t word_count = node_locations.back();

    instructions = avx_buffer<uint32_t>::zero(word_count);

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

    auto registers = avx_buffer<uint32_t>::zero(nodes * registers_per_node);

    for (uint32_t v : depth_starts) {
        std::cout << v << '\n';
    }
}
