#include "codegen/rv_generator.hpp"

#include "codegen/rv_nodetype.hpp"

#include "utils.hpp"

#include <codegen/depthtree.hpp>

#include <iomanip>
#include <limits>
#include <ranges>
#include <algorithm>

#include <cmath>

#include <immintrin.h>

rv_generator::rv_generator(const DepthTree& tree)
    : nodes { tree.filledNodes() }
    , node_types   { make_buffer<rv_node_type>(nodes) }
    , result_types { make_buffer<DataType>(nodes)     }
    , parents      { make_buffer<int32_t>(nodes)      }
    , depth        { make_buffer<int32_t>(nodes)      }
    , child_idx    { make_buffer<int32_t>(nodes)      }
    , node_data    { make_buffer<uint32_t>(nodes)     } {

    std::ranges::transform(tree.getNodeTypes(), node_types.get(), [](uint8_t v) {
        return static_cast<rv_node_type>(pareas_to_rv_nodetype[v]);
    });

    std::ranges::transform(tree.getResultingTypes(), result_types.get(), [](uint8_t v) {
        return static_cast<DataType>(v);
    });

    std::copy_n(tree.getParents().begin(),        nodes, parents.get());
    std::copy_n(tree.getDepth().begin(),          nodes, depth.get());
    std::copy_n(tree.getChildren().begin(),       nodes, child_idx.get());
    std::copy_n(tree.getNodeData().begin(),       nodes, node_data.get());
}

std::ostream& rv_generator::print(std::ostream& os) const {
    std::ios_base::fmtflags f { os.flags() };
    os << std::setfill(' ');

    auto digits = static_cast<size_t>(log10l(static_cast<long double>(nodes)) + 1);

    for (size_t i = 0; i < nodes; ++i) {
        // Why must GCC not have std::format support yet...
        os << "Node "          << std::setw(digits) << i
           << ", type = "      << std::setw(2) << static_cast<uint32_t>(node_types[i])
           << ", result = "    << std::setw(9) << result_types[i]
           << ", parent = "    << std::setw(2) << parents[i]
           << ", depth = "     << std::setw(2) << depth[i]
           << ", child_idx = " << std::setw(2) << child_idx[i]
           << ", data = "      << std::setw(2) << node_data[i]
           << '\n';
    }

    os << "end\n";

    os.flags(f);

    return os;
}



void rv_generator_st::process() {
    preprocess();
}

void rv_generator_st::preprocess() {
    // TODO: Absolute or relative register indices?
    for (size_t i = 0; i < nodes; ++i) {
        /* transform only func_args */
        if (node_types[i] == rv_node_type::func_arg) {
            /* node_data is the argument index, for this specific type,
             * child_idx is the index of the argument overall
             */
            if (result_types[i] == DataType::FLOAT_REF
                && node_data[i] < 8) {
                /* 8 floating point registers are available for arguments,
                 * meaning the first 8 are always passed in the corresponding registers
                 */
                node_types[i] = rv_node_type::func_arg;
                /* Adjust node data to represent actual register index */
                node_data[i] += 10;

            } else {
                /* Index of the integer register in which to place this argument */
                int32_t preceding_integers;
                if (result_types[i] == DataType::FLOAT_REF) {
                    /* All preceding non-float arguments are integer */
                    preceding_integers = child_idx[i] - node_data[i];
                    
                } else {
                    preceding_integers = node_data[i];
                }

                int32_t preceding_floats = child_idx[i] - preceding_integers;
                int32_t ir_idx = preceding_integers + std::max<int32_t>(preceding_floats - 8, 0);

                if (ir_idx < 8) {
                    /* This argument is placed in an integer register */
                    if (result_types[i] == DataType::FLOAT_REF) {
                        /* A float in an integer register */
                        node_types[i] = rv_node_type::func_arg_float_as_int;

                        /* Set the node data to the index of the integer register this argument gets placed in */
                        node_data[i] = ir_idx + 10;
                    } else {
                        /* An integer in an integer register */
                        node_types[i] = rv_node_type::func_arg;
                        /* Node data is still the register index, so adjust */
                        node_data[i] = ir_idx + 10;
                    }
                } else {
                    /* Register overflow, argument is on the stack */
                    node_types[i] = rv_node_type::func_arg_on_stack;

                    /* Every argument past the 8th takes 4 bytes */
                    node_data[i] = (ir_idx - 8) * 4;
                }
            }
        }
    }
}
