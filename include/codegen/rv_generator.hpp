#ifndef PTILOPSIS_RV_GENERATOR_HPP
#define PTILOPSIS_RV_GENERATOR_HPP

#include <span>
#include <memory>
#include <thread>
#include <ostream>

#include <cstdint>

#include "simd.hpp"

/* Utility vector to automatically swap 2 buffers */
template <typename T>
class swap_buffer {
    size_t _size;
    std::unique_ptr<T[]> vec1;
    std::unique_ptr<T[]> vec2;
    bool active = false;

    struct buffer_deleter {
        void operator()(T* ptr) const {
            operator delete(ptr, std::align_val_t{ AVX_ALIGNMENT });
        }
    };

    public:
    swap_buffer() = delete;
    swap_buffer(size_t size)
        : _size{ size }
        /* Align to 64 bytes, for theoretical AVX-512 support */
        , vec1{ static_cast<T*>(operator new[](sizeof(T) * _size, std::align_val_t{ AVX_ALIGNMENT })), {} }
        , vec2{ static_cast<T*>(operator new[](sizeof(T) * _size, std::align_val_t{ AVX_ALIGNMENT })), {} } { }

    size_t size() const {
        return _size;
    }

    void swap() {
        active = !active;
    }

    auto& cur() {
        return active ? vec2 : vec1;
    }

    const auto& cur() const {
        return active ? vec2 : vec1;
    }

    std::unique_ptr<T[]>* operator->() {
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

    swap_buffer<uint8_t> node_types;
    swap_buffer<uint8_t> result_types;
    swap_buffer<int32_t> parents;
    swap_buffer<int32_t> depth;
    swap_buffer<int32_t> child_idx;
    swap_buffer<uint32_t> node_data;
    public:
    rv_generator(const DepthTree& tree);

    void preprocess();

    std::ostream& print(std::ostream& os) const;
};

inline std::ostream& operator<<(std::ostream& os, const rv_generator& gen) {
    return gen.print(os);
}

#endif /* PTILOPSIS_RV_GENERATOR_HPP */
