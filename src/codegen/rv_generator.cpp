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
    , node_types { nodes }, result_types { nodes }, parents   { nodes }
    , depth      { nodes }, child_idx    { nodes }, node_data { nodes } {
    
    auto types = tree.getNodeTypes();
    std::ranges::transform(types, node_types->get(), [](uint8_t v) {
        return static_cast<uint8_t>(pareas_to_rv_nodetype[v]);
    });
    std::copy_n(tree.getResultingTypes().begin(), nodes, result_types->get());
    std::copy_n(tree.getParents().begin(),        nodes, parents->get());
    std::copy_n(tree.getDepth().begin(),          nodes, depth->get());
    std::copy_n(tree.getChildren().begin(),       nodes, child_idx->get());
    std::copy_n(tree.getNodeData().begin(),       nodes, node_data->get());
}

void rv_generator::preprocess() {
    //size_t remaining = nodes;
    //__m256i vec = _mm256_load_si256(reinterpret_cast<__m256i*>(node_types.in().get()));
    //__m256i cmp = _mm256_set1_epi8(ptilopsis::to_integral(rv_node_type::func_arg));
}

std::ostream& rv_generator::print(std::ostream& os) const {
    std::ios_base::fmtflags f { os.flags() };
    os << std::setfill(' ');

    /* Defined in datatype.cpp */
    // ReSharper disable once CppTooWideScope
    // ReSharper disable once CppInconsistentNaming
    extern const char* TYPE_NAMES[];

    auto digits = static_cast<size_t>(log10l(static_cast<long double>(nodes)) + 1);

    for (size_t i = 0; i < nodes; ++i) {
        // Why must GCC not have std::format support yet...
        os << "Node "          << std::setw(digits) << i
           << ", type = "      << std::setw(2) << static_cast<uint32_t>(node_types[i])
           << ", result = "    << std::setw(9) << TYPE_NAMES[result_types[i]]
           << ", parent = "    << std::setw(2) << parents[i]
           << ", depth = "     << std::setw(2) << depth[i]
           << ", child_idx = " << std::setw(2) << child_idx[i]
           << ", data = "      << std::setw(2) << node_data[i]
           << '\n';
    }

    os.flags(f);

    return os;
}
