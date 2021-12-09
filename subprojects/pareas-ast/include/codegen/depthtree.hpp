#ifndef _PAREAS_CODEGEN_DEPTHTREE_HPP
#define _PAREAS_CODEGEN_DEPTHTREE_HPP

#include <cstdint>
#include <cstddef>
#include <iosfwd>
#include <unordered_map>
#include <span>

class ASTNode;

class DepthTree {
    private:
        uint8_t* node_types;
        uint8_t* resulting_types;
        int32_t* parents;
        int32_t* depth;
        int32_t* child_idx;
        uint32_t* node_data;
        size_t max_nodes;
        size_t filled_nodes;
        size_t max_depth;

        void construct(ASTNode*);
        void setElement(size_t, ASTNode*, size_t, size_t);
    public:
        DepthTree(ASTNode*);
        ~DepthTree();

        void print(std::ostream&) const;

        inline std::span<const uint8_t> getNodeTypes() const {
            return { this->node_types, filled_nodes };
        }
        inline std::span<const uint8_t> getResultingTypes() const {
            return { this->resulting_types, filled_nodes };
        }
        inline std::span<const int32_t> getParents() const {
            return { this->parents, filled_nodes };
        }
        inline std::span<const int32_t> getDepth() const {
            return { this->depth, filled_nodes };
        }
        inline std::span<const int32_t> getChildren() const {
            return { this->child_idx, filled_nodes };
        }
        inline std::span<const uint32_t> getNodeData() const {
            return { this->node_data, filled_nodes };
        }
        inline size_t maxNodes() const {
            return this->max_nodes;
        }
        inline size_t maxDepth() const {
            return this->max_depth;
        }
        inline size_t filledNodes() const {
            return this->filled_nodes;
        }
};

std::ostream& operator<<(std::ostream&, const DepthTree&);

#endif
