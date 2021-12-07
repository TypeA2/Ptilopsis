#include "codegen/rv_generator.hpp"

#include <codegen/depthtree.hpp>

#include <iomanip>
#include <limits>
#include <cmath>

#include <immintrin.h>

rv_generator::rv_generator(const DepthTree& tree)
    : nodes { tree.filledNodes() }
    , node_types { nodes }, result_types { nodes }, parents   { nodes }
    , depth      { nodes }, child_idx    { nodes }, node_data { nodes } {
    
    std::copy_n(tree.getNodeTypes(),      nodes, node_types->data());
    std::copy_n(tree.getResultingTypes(), nodes, result_types->data());
    std::copy_n(tree.getParents(),        nodes, parents->data());
    std::copy_n(tree.getDepth(),          nodes, depth->data());
    std::copy_n(tree.getChildren(),       nodes, child_idx->data());
    std::copy_n(tree.getNodeData(),       nodes, node_data->data());
}

void rv_generator::preprocess() {
    size_t remaining = nodes;
}

std::ostream& rv_generator::print(std::ostream& os) const {
    std::ios_base::fmtflags f { os.flags() };
    os << std::setfill(' ');

    /* Defined in astnode.cpp */
    extern const char* NODE_NAMES[];

    /* Defined in datatype.cpp */
    extern const char* TYPE_NAMES[];

    size_t digits = log10l(nodes) + 1;

    for (size_t i = 0; i < nodes; ++i) {
        // Why must GCC not have std::format support yet...
        os << "Node " << std::setw(digits) << i
            << ", type = " << std::setw(2) << static_cast<uint32_t>(node_types[i])
            << ", result = "    << std::setw(9) << TYPE_NAMES[result_types[i]]
            << ", parent = "    << std::setw(2) << parents[i]
            << ", depth = "     << std::setw(2) << depth[i]
            << ", child_idx = " << std::setw(2) << child_idx[i]
            << ", data = "      << std::setw(2) << node_data[i]
            << " | " << NODE_NAMES[node_types[i]]
            << '\n';
    }

    os.flags(f);

    return os;
}
