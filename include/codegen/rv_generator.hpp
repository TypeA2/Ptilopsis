#ifndef PTILOPSIS_RV_GENERATOR_HPP
#define PTILOPSIS_RV_GENERATOR_HPP

#include <span>
#include <vector>
#include <thread>
#include <ostream>

#include <cstdint>

/* Utility vector to automatically swap 2 buffers */
template <typename T>
class swap_vector {
    std::vector<T> vec1;
    std::vector<T> vec2;
    bool active = false;

    public:
    swap_vector() = delete;
    swap_vector(size_t size) : vec1(size), vec2(size) { }

    void swap() {
        active = !active;
    }

    auto& cur() {
        return active ? vec2 : vec1;
    }

    const auto& cur() const {
        return active ? vec2 : vec1;
    }

    std::vector<T>* operator->() {
        return &cur();
    }

    auto& operator[](size_t i) {
        return cur()[i];
    }

    const auto& operator[](size_t i) const {
        return cur()[i];
    }

    auto& in() {
        return cur();
    }

    auto& out() {
        return (&cur() == &vec1) ? vec2 : vec1;
    }
};

class DepthTree;

/* RISC-V machine code generator */
class rv_generator {
    size_t nodes;

    swap_vector<uint8_t> node_types;
    swap_vector<uint8_t> result_types;
    swap_vector<int32_t> parents;
    swap_vector<int32_t> depth;
    swap_vector<int32_t> child_idx;
    swap_vector<uint32_t> node_data;
    public:
    rv_generator(const DepthTree& tree);

    void preprocess();

    std::ostream& print(std::ostream& os) const;
};

inline std::ostream& operator<<(std::ostream& os, const rv_generator& gen) {
    return gen.print(os);
}

#endif /* PTILOPSIS_RV_GENERATOR_HPP */
